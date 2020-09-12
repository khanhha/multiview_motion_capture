import numpy as np
import cv2
import pickle
from typing import Dict, List, Optional
from pathlib import Path
import json
import scipy.stats as st
import os
from easydict import EasyDict
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from dataclasses import dataclass, field
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorlayer as tl
import copy
from tqdm import tqdm
import imageio
from scipy.optimize import linear_sum_assignment
from typing import Tuple
from pose_def import Pose, KpsType, KpsFormat, get_pose_bones_index, conversion_openpose_25_to_coco
from mv_math_util import (Calib, calc_epipolar_error, triangulate_point_groups_from_multiple_views_linear,
                          project_3d_points_to_image_plane_without_distortion, unproject_uv_to_rays,
                          points_to_lines_distances)
from pose_viz import plot_poses_3d
from enum import Enum
from inverse_kinematics import PoseParam, Skeleton

matplotlib.use('Qt5Agg')


@dataclass
class FrameData:
    frame_idx: int
    poses: Dict[int, Pose]
    calib: Calib
    view_id: int


class PoseAssociation:

    def __init__(self,
                 cam: Calib,
                 frame_idx: int,
                 view_id: int,
                 id_obj: Tuple[int, Pose],
                 use_weighted_kps_score: bool,
                 min_triangulate_kps_score: float,
                 match_threshold: float = 200):
        """
        :param cam:
        :param frame_idx:
        :param view_id:
        :param use_weighted_kps_score:
        :param min_triangulate_kps_score:
        :param match_threshold:
        """
        self.frame_idx = frame_idx
        self.cams = [cam]
        self.view_ids = [view_id]
        self.id_poses = [id_obj]
        self.threshold = match_threshold
        self.use_weighted_kps_score = use_weighted_kps_score
        self.min_triangulate_kps_score = min_triangulate_kps_score
        self.cur_pose_3d = None

    def __len__(self):
        return len(self.id_poses)

    @property
    def poses(self):
        return [obj[1] for obj in self.id_poses]

    @property
    def cam_projs(self):
        return [c.P for c in self.cams]

    def calc_epipolar_error(self, cam_o: Calib, pose_o: Pose):
        too_wrong = False  # if true we cannot join {other} with this
        total_error = 0
        for pose, cam in zip(self.poses, self.cams):
            error = calc_epipolar_error(cam1=cam, keypoints_1=pose.keypoints, scores_1=pose.keypoints_score,
                                        cam2=cam_o, keypoints_2=pose_o.keypoints, scores_2=pose_o.keypoints_score,
                                        score_weighted=self.use_weighted_kps_score)
            total_error += error

            # TODO: hard threshold for cost is variant to image resolution
            if total_error > self.threshold:
                too_wrong = True

        return total_error / len(self.poses), too_wrong

    def merge(self, cam: Calib, id_obj: Tuple[int, Pose], view_id: int):
        self.cams.append(cam)
        self.view_ids.append(view_id)
        self.id_poses.append(id_obj)

    def triangulate(self, min_kps_score=None):
        min_kps_score = min_kps_score if min_kps_score is not None else self.min_triangulate_kps_score
        if len(self.poses) >= 2:
            proj_mats = np.array([c.P for c in self.cams])
            poses_2d = [np.concatenate([p.keypoints, p.keypoints_score.reshape((-1, 1))], axis=1) for p in self.poses]
            self.cur_pose_3d = triangulate_point_groups_from_multiple_views_linear(proj_mats,
                                                                                   poses_2d,
                                                                                   min_kps_score)
            return self.cur_pose_3d
        else:
            raise ValueError('not enough 2d poses for triangulation')

    def calc_reprojection_error(self, update_triangulate):
        if self.cur_pose_3d is None or update_triangulate:
            self.triangulate()

        kps_avg_err = None
        for p2d, cam in zip(self.poses, self.cams):
            cam_p = cam.P
            reproj_p2d = project_3d_points_to_image_plane_without_distortion(cam_p, self.cur_pose_3d)
            diff = reproj_p2d - p2d.keypoints
            if kps_avg_err is None:
                kps_avg_err = np.linalg.norm(diff, axis=1)
            else:
                kps_avg_err += np.linalg.norm(diff, axis=1)

        return kps_avg_err / len(self.poses)

    def debug_get_association_viz(self, view_imgs: Dict[int, np.ndarray], h=256):
        all_crops = []
        for idx in range(len(self.view_ids)):
            v_id = self.view_ids[idx]
            pose = self.id_poses[idx][1]
            img = view_imgs[v_id]
            x1, y1, x2, y2 = pose.box.astype(np.int)
            crop = img[y1:y2, x1:x2]
            c_h, c_w = crop.shape[:2]
            new_h = h
            new_w = int((c_w / c_h) * new_h)
            crop = cv2.resize(crop, dsize=(new_w, new_h))
            cv2.putText(crop, f'{v_id}', (int(0.5 * new_w), 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
            all_crops.append(crop)

        # view_id increasing from left to right
        sorted_view_ids = np.argsort(self.view_ids)
        all_crops = [all_crops[idx] for idx in sorted_view_ids]

        viz = np.concatenate(all_crops, axis=1)
        cv2.putText(viz, f'{self.frame_idx}', (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
        return viz


def match_objects_across_views(frame_idx: int,
                               view_frames: List[FrameData],
                               use_kps_weighted_score: bool,
                               match_threshold,
                               min_triangulate_kps_score) -> List[PoseAssociation]:
    view_pose_counts = [len(frm.poses) for frm in view_frames]
    init_view_idx = int(np.argmax(view_pose_counts))
    init_frame = view_frames[init_view_idx]
    init_cam = view_frames[init_view_idx].calib

    mv_obj_grps = [PoseAssociation(cam=init_cam,
                                   frame_idx=frame_idx,
                                   view_id=init_frame.view_id,
                                   id_obj=(obj_id, obj),
                                   use_weighted_kps_score=use_kps_weighted_score,
                                   min_triangulate_kps_score=min_triangulate_kps_score,
                                   match_threshold=match_threshold)
                   for obj_id, obj in init_frame.poses.items()]

    other_view_idxs = [i for i in range(len(view_frames)) if i != init_view_idx]
    for vi in other_view_idxs:
        frame = view_frames[vi]
        if len(frame.poses) < 1:
            continue

        poses_obj_ids = [(obj_id, pose) for obj_id, pose in frame.poses.items()]
        obj_ids, poses = list(zip(*poses_obj_ids))

        # calculate cost matrix
        n_hyp = len(mv_obj_grps)
        n_poses = len(poses)
        cost_mat = np.zeros((n_hyp, n_poses))
        mask_mat = np.zeros_like(cost_mat).astype(np.int32)
        for obj_idx, pose in enumerate(poses):
            for hid, h in enumerate(mv_obj_grps):
                c, too_bad = h.calc_epipolar_error(frame.calib, pose)
                cost_mat[hid, obj_idx] = c
                if too_bad:
                    mask_mat[hid, obj_idx] = 1

        rows, cols = linear_sum_assignment(cost_mat)

        # create new, or merge object into groups
        matched_pids = set()
        for hid, obj_idx in zip(rows, cols):
            is_masked = mask_mat[hid, obj_idx] == 1
            matched_pids.add(obj_idx)
            if is_masked:
                # even the closest other person is
                # too far away (> threshold)
                mv_obj_grps.append(PoseAssociation(
                    frame_idx=frame_idx,
                    cam=frame.calib,
                    view_id=frame.view_id,
                    id_obj=(obj_ids[obj_idx], frame.poses[obj_ids[obj_idx]]),
                    use_weighted_kps_score=use_kps_weighted_score,
                    min_triangulate_kps_score=min_triangulate_kps_score,
                    match_threshold=match_threshold))
            else:
                mv_obj_grps[hid].merge(cam=frame.calib,
                                       view_id=frame.view_id,
                                       id_obj=(obj_ids[obj_idx], frame.poses[obj_ids[obj_idx]]))

        # add the remaining poses that are not matched yet
        for obj_idx, pose in enumerate(poses):
            if obj_idx not in matched_pids:
                mv_obj_grps.append(PoseAssociation(
                    frame_idx=frame_idx,
                    cam=frame.calib,
                    view_id=frame.view_id,
                    id_obj=(obj_ids[obj_idx], frame.poses[obj_ids[obj_idx]]),
                    min_triangulate_kps_score=min_triangulate_kps_score,
                    use_weighted_kps_score=use_kps_weighted_score,
                    match_threshold=match_threshold))

    return mv_obj_grps


def load_calib(cpath: Path):
    if 'pkl' in cpath.suffix:
        with open(str(cpath), 'rb') as file:
            data = pickle.load(file)
            mat_k = np.array(data["K"]).reshape((3, 3))
            mat_rt = np.concatenate(
                [np.array(data["R"]).reshape((3, 3)), np.array(data["t"]).reshape((3, 1))], axis=1)
            mat_p = mat_k @ mat_rt
            data["P"] = mat_p

            kr_inv = mat_rt[:3, :3].transpose() @ np.linalg.inv(mat_k)
            return Calib(K=mat_k, Rt=mat_rt, P=mat_p, Kr_inv=kr_inv)
    else:
        raise ValueError(f'unsupported calibration format. {cpath.name}')


@dataclass
class TrackletMatch:
    # view_idx, p_id
    view_idxs: List[int] = field(default_factory=list)
    pose_ids: List[int] = field(default_factory=list)

    def __len__(self):
        return len(self.view_idxs)


class TrackState(Enum):
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """
    Tentative = 1
    Confirmed = 2
    Dead = 3


def triangulate_util(cam_poses: List[Pose], cam_projs: List[np.ndarray], min_triangulate_score=0.01):
    p_type = cam_poses[0].pose_type
    poses = [np.concatenate([pose.keypoints, pose.keypoints_score], axis=-1) for pose in cam_poses]

    p_3d = triangulate_point_groups_from_multiple_views_linear(np.array(cam_projs), poses,
                                                               min_score=min_triangulate_score,
                                                               post_optimize=True)
    pose_3d = Pose(p_type, p_3d[:, :-1], p_3d[:, -1][:, np.newaxis], box=None)
    return pose_3d


class MvTracklet:
    def __init__(self,
                 frm_idx: int,
                 cam_poses_2d: List[Pose],
                 cam_projs: List[np.ndarray],
                 n_inits: int = 3,
                 max_age: int = 3):
        self.frame_idxs: List[int] = [frm_idx]
        self.poses_3d: List[Pose] = [triangulate_util(cam_poses_2d, cam_projs)]
        self.cam_poses_2d: List[List[Pose]] = [cam_poses_2d]
        self.cam_projs: List[List[np.ndarray]] = [cam_projs]

        self.bone_lengs: np.ndarray
        self.poses: List[PoseParam]

        self.time_since_update = 0
        self.hits = 1
        self.state = TrackState.Tentative
        self.max_age = max_age
        self.n_inits = n_inits

    @property
    def last_pose_3d(self):
        return self.poses_3d[-1]

    def __len__(self):
        return len(self.frame_idxs)

    def predict(self):
        self.time_since_update += 1

    def update(self, frm_idx: int, match: TrackletMatch, frames: List[FrameData]):
        cam_projs = [frames[v_idx].calib.P for v_idx in match.view_idxs]
        cam_poses = [frames[v_idx].poses[p_id] for v_idx, p_id in zip(match.view_idxs, match.pose_ids)]
        pose_3d = triangulate_util(cam_poses, cam_projs)

        self.frame_idxs.append(frm_idx)
        self.cam_poses_2d.append(cam_poses)
        self.cam_projs.append(cam_projs)
        self.poses_3d.append(pose_3d)

        self.time_since_update = 0
        self.hits += 1

        if self.is_tentative() and self.hits >= self.n_inits:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Dead
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Dead

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_dead(self):
        return self.state == TrackState.Dead


class MvTracker:
    def __init__(self):
        self.tracklets: List[MvTracklet] = []
        self.dead_tracklets: List[MvTracklet] = []

    @classmethod
    def tracklet_to_pose_2d_cost(cls, tlet: MvTracklet, pose_2d: Pose, calib: Calib):
        kps_rays = unproject_uv_to_rays(pose_2d.keypoints, calib)
        cam_locs = np.repeat(calib.cam_loc.reshape((1, 3)), len(kps_rays), 0)
        kps_dsts = points_to_lines_distances(tlet.last_pose_3d.keypoints, cam_locs, kps_rays)
        return np.mean(kps_dsts)

    @classmethod
    def tracklet_to_poses_association(cls, tracklets, d_frm: FrameData, max_dst=0.1):
        n_tlets = len(tracklets)
        n_poses = len(d_frm.poses)
        if n_tlets and n_poses:
            cost_mat = np.zeros((n_tlets, n_poses), dtype=np.float)
            invalid_mask = np.full_like(cost_mat, fill_value=False)
            p_ids, poses = list(zip(*[(p_id, pose) for p_id, pose in d_frm.poses.items()]))

            for t_idx, tlet in enumerate(tracklets):
                for p_idx, pose in enumerate(poses):
                    cost_mat[t_idx, p_idx] = cls.tracklet_to_pose_2d_cost(tlet, pose, d_frm.calib)
                    if cost_mat[t_idx, p_idx] > max_dst:
                        invalid_mask[t_idx, p_idx] = True

            rows, cols = linear_sum_assignment(cost_mat)
            matches = [(t_idx, p_ids[p_idx]) for t_idx, p_idx in zip(rows, cols) if not invalid_mask[t_idx, p_idx]]
            return matches
        else:
            return []

    def update(self, frm_idx: int, d_frames: List[FrameData]):
        n_views = len(d_frames)
        for tlet in self.tracklets:
            tlet.predict()

        # only do association with alive tracks
        cur_tracklets = [tlet for tlet in self.tracklets if not tlet.is_dead()]

        tlet_matches = [TrackletMatch() for _ in range(len(cur_tracklets))]
        for vi in range(n_views):
            matches = self.tracklet_to_poses_association(cur_tracklets, d_frames[vi])
            for t_idx, p_id in matches:
                tlet_matches[t_idx].view_idxs.append(vi)
                tlet_matches[t_idx].pose_ids.append(p_id)

        # update tracks with associated detections
        for tlet, tlet_match in zip(cur_tracklets, tlet_matches):
            if len(tlet_match) >= 2:
                tlet.update(frm_idx, tlet_match, d_frames)
            else:
                tlet.mark_missed()

        # now handle un-matched poses
        view_matched_pids = [[] for _ in range(n_views)]
        for tlet_match in tlet_matches:
            for v_idx, p_id in zip(tlet_match.view_idxs, tlet_match.pose_ids):
                view_matched_pids[v_idx].append(p_id)

        # run multi-view 2d pose matching on undetected poses
        new_d_frames = [copy.copy(frm) for frm in d_frames]
        for v_idx, frm in enumerate(new_d_frames):
            for p_id in view_matched_pids[v_idx]:
                del frm.poses[p_id]

        pose_grps = match_objects_across_views(frame_idx=frm_idx, view_frames=new_d_frames,
                                               use_kps_weighted_score=True, match_threshold=12,
                                               min_triangulate_kps_score=0.1)
        for grp in pose_grps:
            if len(grp) >= 2:
                p_3d_co = grp.triangulate()
                p_3d = Pose(grp.poses[0].pose_type, p_3d_co[:, :3], p_3d_co[:, -1][:, np.newaxis], box=None)
                tlet = MvTracklet(frm_idx, cam_poses_2d=grp.poses, cam_projs=grp.cam_projs)
                self.tracklets.append(tlet)

        # filter out dead tracks
        dead_tlets = [tlet for tlet in self.tracklets if tlet.is_dead()]
        self.dead_tracklets.extend(dead_tlets)
        self.tracklets = [tlet for tlet in self.tracklets if not tlet.is_dead()]


def draw_poses(frm_img: np.ndarray, frm_data: FrameData):
    for p_id, pose in frm_data.poses.items():
        for i in range(len(pose.keypoints)):
            cv2.circle(frm_img, (int(pose.keypoints[i, 0]), int(pose.keypoints[i, 1])),
                       radius=2, color=(255, 0, 0), thickness=2, lineType=cv2.FILLED)
    return frm_img


def parse_openpose_kps(js_path: Path):
    with open(js_path, 'rt') as file:
        data = json.load(file)
    people = data["people"]
    poses = {}
    for p_id, person in enumerate(people):
        kps = np.array(person["pose_keypoints_2d"]).reshape((-1, 3))
        coco_kps = conversion_openpose_25_to_coco(kps)
        poses[p_id] = Pose(KpsFormat.COCO, keypoints=coco_kps[:, :2], keypoints_score=coco_kps[:, -1][:, np.newaxis],
                           box=None)
    return poses


def extract_frame_data_from_openpose(in_dir: Path, calib_dir: Path, out_data_dir: Path):
    # sorted based on camera number
    cam_opn_dirs = sorted([ddir for ddir in in_dir.glob('*') if ddir.is_dir()], key=lambda path: path.stem)
    calibs = [load_calib(calib_dir / f'{vpath.stem}.pkl') for vpath in cam_opn_dirs]

    cam_kps_paths = []
    for kps_dir in cam_opn_dirs:
        kps_paths = sorted([kpath for kpath in kps_dir.glob('*.json')], key=lambda path: int(path.stem.split('_')[1]))
        cam_kps_paths.append(kps_paths)

    n_frms = min([len(kps_paths) for kps_paths in cam_kps_paths])
    for frm_idx in tqdm(range(n_frms), desc='parsing openpose json'):
        js_paths = [kps_paths[frm_idx] for kps_paths in cam_kps_paths]
        cam_poses = [parse_openpose_kps(js_path) for js_path in js_paths]
        d_frames = [FrameData(frm_idx, poses, calib, view_id=v_idx + 1)
                    for v_idx, (poses, calib) in enumerate(zip(cam_poses, calibs))]
        with open(out_data_dir / f'{str(frm_idx).zfill(6)}.pkl', 'wb') as file:
            pickle.dump(obj=d_frames, file=file)


def load_pickle(fpath: Path, mode):
    with open(fpath, mode) as file:
        return pickle.load(file)


def test_ik_tlet(tlet: MvTracklet):
    from inverse_kinematics import run_test_ik
    n = len(tlet)
    for i in range(n):
        p_3d = tlet.poses_3d[i]
        cam_poses = tlet.cam_poses_2d[i]
        cam_projs = tlet.cam_projs[i]
        run_test_ik(p_3d.keypoints, [p.to_kps_array() for p in cam_poses], cam_projs)


def run_main(video_dir: Path, pose_dir: Path, out_dir: Path):
    vpaths = sorted([vpath for vpath in video_dir.glob('*.*')], key=lambda path: path.stem)
    vreaders = [cv2.VideoCapture(str(vpath)) for vpath in vpaths]
    frm_pose_paths = sorted([ppath for ppath in pose_dir.glob('*.pkl')], key=lambda path: int(path.stem))
    vpath = '/media/F/thesis/data/debug/test.mp4'
    db_writer = imageio.get_writer(vpath)

    tracker = MvTracker()
    frm_idx = 0
    n_test = 10
    with tqdm(total=len(frm_pose_paths), desc='tracking') as bar:
        while True:
            bar.update()
            bar.set_description(
                f'tracking. n_dead = {len(tracker.dead_tracklets)}. n_tracks = {len(tracker.tracklets)}')
            # oks_frames = [vreader.read() for vreader in vreaders]
            # is_oks, org_frames = list(zip(*oks_frames))
            # if not all(is_oks) or frm_idx >= len(frm_pose_paths):
            #     break

            d_frames = load_pickle(frm_pose_paths[frm_idx], 'rb')

            # for frm_img, frm_data in zip(org_frames, d_frames):
            #     draw_poses(frm_img, frm_data)
            #
            # for i in range(len(d_frames)):
            #     cv2.imshow(f'{i}', frm_img)
            # cv2.waitKeyEx(1)

            tracker.update(frm_idx, d_frames)
            frm_idx += 1

            if frm_idx >= n_test:
                break

    db_writer.close()

    all_tlets = tracker.tracklets + tracker.dead_tracklets
    all_tlets = sorted(all_tlets, key=lambda tlet: -len(tlet.poses_3d))
    all_tlets = all_tlets[:2]

    test_ik_tlet(all_tlets[0])

    poses_3d = [p for p in all_tlets[0].poses_3d]
    plot_poses_3d(poses_3d, '/media/F/thesis/data/test_mv/2_pp/test_tracklet.mp4')


if __name__ == "__main__":
    run_main(video_dir=Path('/media/F/thesis/data/test_mv/2_pp/videos'),
             pose_dir=Path('/media/F/thesis/data/test_mv/2_pp/d_frames_coco'),
             out_dir=Path('/media/F/thesis/data/test_mv/2_pp_heatmaps'))
    # extract_frame_data_from_videos(
    #     in_dir=Path('/media/F/thesis/data/test_mv/2_pp/videos'),
    #     calib_dir=Path('/media/F/thesis/data/test_mv/2_pp/calibs'),
    #     out_pose_dir=Path('/media/F/thesis/data/test_mv/2_pp/poses'))

    # extract_frame_data_from_openpose(in_dir=Path('/media/F/thesis/data/test_mv/2_pp/openpose_keypoints'),
    #                                  calib_dir=Path('/media/F/thesis/data/test_mv/2_pp/calibs'),
    #                                  out_data_dir=Path('/media/F/thesis/data/test_mv/2_pp/d_frames_coco'))
