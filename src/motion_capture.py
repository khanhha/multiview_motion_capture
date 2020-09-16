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
import argparse
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
from pose_def import (Pose, KpsType, KpsFormat, get_pose_bones_index, conversion_openpose_25_to_coco,
                      map_to_common_keypoints)
from mv_math_util import (Calib, calc_epipolar_error, triangulate_point_groups_from_multiple_views_linear,
                          project_3d_points_to_image_plane_without_distortion, unproject_uv_to_rays,
                          points_to_lines_distances)
from pose_viz import plot_poses_3d
from enum import Enum
from inverse_kinematics import PoseShapeParam, Skeleton, PoseSolver, load_skeleton

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
            if pose.box is not None:
                x1, y1, x2, y2 = pose.box.astype(np.int)
            else:
                valid_kps = pose.keypoints[pose.keypoints_score.flatten() > 0.1, :]
                bmin, bmax = np.min(valid_kps, axis=0), np.max(valid_kps, axis=0)
                x1, y1, x2, y2 = np.concatenate([bmin, bmax]).astype(np.int)

            if y2 > y1 + 5 and x2 > x1 + 5:
                crop = img[y1:y2, x1:x2]
                c_h, c_w = crop.shape[:2]
                new_h = h
                new_w = int((c_w / c_h) * new_h)
                crop = cv2.resize(crop, dsize=(new_w, new_h))
                cv2.putText(crop, f'{v_id}', (int(0.5 * new_w), 128),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
                all_crops.append(crop)

        # view_id increasing from left to right
        if all_crops:
            # sorted_view_ids = np.argsort(self.view_ids)
            # all_crops = [all_crops[idx] for idx in sorted_view_ids]

            viz = np.concatenate(all_crops, axis=1)
            cv2.putText(viz, f'{self.frame_idx}', (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
            return viz
        else:
            return np.zeros((128, 128, 3), dtype=np.uint8)


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
            return Calib(K=mat_k, Rt=mat_rt, P=mat_p, Kr_inv=kr_inv, img_wh_size=(1920, 1080))
    elif 'js' in cpath.suffix:
        with open(str(cpath), 'r') as file:
            js_data = json.load(file)
            mat_k = np.array(js_data["K"]).reshape((3, 3))
            mat_rt = np.array(js_data["RT"]).reshape((3, 4))
            mat_p = mat_k @ mat_rt
            img_size = js_data["imgSize"]
            kr_inv = mat_rt[:3, :3].transpose() @ np.linalg.inv(mat_k)
            return Calib(K=mat_k, Rt=mat_rt, P=mat_p, Kr_inv=kr_inv, img_wh_size=img_size)
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
                 skel: Skeleton,
                 n_inits: int = 3,
                 max_age: int = 3):
        self.frame_idxs: List[int] = [frm_idx]
        self.cam_poses_2d: List[List[Pose]] = [cam_poses_2d]
        self.cam_projs: List[List[np.ndarray]] = [cam_projs]
        self.skel = skel
        solver = PoseSolver(skel, init_pose=None,
                            cam_poses_2d=[p.to_kps_array() for p in cam_poses_2d],
                            cam_projs=cam_projs,
                            obs_kps_format=cam_poses_2d[0].pose_type)
        pparam, pose = solver.solve()
        self.poses: List[Tuple[PoseShapeParam, Pose]] = [(pparam, pose)]

        self.bone_lengs: np.ndarray

        self.time_since_update = 0
        self.hits = 1
        self.state = TrackState.Tentative
        self.max_age = max_age
        self.n_inits = n_inits

    @property
    def last_pose_3d(self):
        return self.poses[-1][1]

    def __len__(self):
        return len(self.frame_idxs)

    def predict(self):
        self.time_since_update += 1

    def update(self, frm_idx: int, match: TrackletMatch, frames: List[FrameData]):
        cam_projs = [frames[v_idx].calib.P for v_idx in match.view_idxs]
        cam_poses = [frames[v_idx].poses[p_id] for v_idx, p_id in zip(match.view_idxs, match.pose_ids)]

        self.frame_idxs.append(frm_idx)
        self.cam_poses_2d.append(cam_poses)
        self.cam_projs.append(cam_projs)

        solver = PoseSolver(self.skel,
                            init_pose=self.poses[-1][0],
                            cam_poses_2d=[p.to_kps_array() for p in cam_poses],
                            cam_projs=cam_projs,
                            obs_kps_format=cam_poses[0].pose_type)
        pparam, pose = solver.solve()
        self.poses.append((pparam, pose))

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
    def __init__(self, skel: Skeleton):
        self.tracklets: List[MvTracklet] = []
        self.dead_tracklets: List[MvTracklet] = []
        self.skeleton = skel

    @classmethod
    def tracklet_to_pose_2d_cost(cls, tlet: MvTracklet, pose_2d: Pose, calib: Calib):
        kps_2ds, kps_3ds = map_to_common_keypoints(pose_2d, tlet.last_pose_3d)
        kps_rays = unproject_uv_to_rays(kps_2ds[:, :2], calib)
        cam_locs = np.repeat(calib.cam_loc.reshape((1, 3)), len(kps_rays), 0)
        kps_dsts = points_to_lines_distances(kps_3ds[:, :3], cam_locs, kps_rays)
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

    def update(self, frm_idx: int, d_frames: List[FrameData], debug_view_imgs: Dict[int, np.ndarray]):
        n_views = len(d_frames)
        for tlet in self.tracklets:
            tlet.predict()

        # only do association with alive tracks
        alive_tracklets = [tlet for tlet in self.tracklets if not tlet.is_dead()]

        # matching between each 3d tracklet and 2d poses in different views
        tlet_matches = [TrackletMatch() for _ in range(len(alive_tracklets))]
        for vi in range(n_views):
            matches = self.tracklet_to_poses_association(alive_tracklets, d_frames[vi])
            for t_idx, p_id in matches:
                tlet_matches[t_idx].view_idxs.append(vi)
                tlet_matches[t_idx].pose_ids.append(p_id)

        # update tracks with associated detections
        for tlet, tlet_match in zip(alive_tracklets, tlet_matches):
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

        # test mv matching
        # from mv_association import match_multiview_poses
        # cams_poses_ids = [[p_id for p_id in frm.poses.keys()]
        #                   for frm in new_d_frames]
        # cams_poses = [[new_d_frames[cam_idx].poses[p_id] for p_id in p_ids]
        #               for cam_idx, p_ids in enumerate(cams_poses_ids)]
        # cams_calibs = [frm.calib
        #                for frm in new_d_frames]
        # person_matches = match_multiview_poses(cams_poses, cams_calibs)
        # pose_grps = []
        # for grp_cam_pose_idxs in person_matches:
        #     grp = None
        #     for cam_idx, p_idx in grp_cam_pose_idxs:
        #         calib = cams_calibs[cam_idx]
        #         p_id = cams_poses_ids[cam_idx][p_idx]
        #         pose = cams_poses[cam_idx][p_idx]
        #         if grp is None:
        #             grp = PoseAssociation(cam=calib, frame_idx=frm_idx, view_id=cam_idx, id_obj=(p_id, pose),
        #                                   use_weighted_kps_score=True,
        #                                   match_threshold=12,
        #                                   min_triangulate_kps_score=0.1)
        #         else:
        #             grp.cams.append(calib)
        #             grp.view_ids.append(cam_idx)
        #             grp.id_poses.append((p_id, pose))
        #     pose_grps.append(grp)

        # debug
        pose_grps = match_objects_across_views(frame_idx=frm_idx, view_frames=new_d_frames,
                                               use_kps_weighted_score=True, match_threshold=12,
                                               min_triangulate_kps_score=0.1)
        for grp in pose_grps:
            if len(grp) >= 2:
                # p_3d_co = grp.triangulate()
                # p_3d = Pose(grp.poses[0].pose_type, p_3d_co[:, :3], p_3d_co[:, -1][:, np.newaxis], box=None)
                tlet = MvTracklet(frm_idx, cam_poses_2d=grp.poses, cam_projs=grp.cam_projs, skel=self.skeleton)
                self.tracklets.append(tlet)
                #
                # viz = grp.debug_get_association_viz(debug_view_imgs)
                # cv2.imshow('test', viz)
                # k = cv2.waitKeyEx(1)
                # while k != ord('n'):
                #     k = cv2.waitKeyEx(1)

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
    calib_paths = {cpath.stem: cpath for cpath in calib_dir.glob('*.*')}
    calibs = [load_calib(calib_paths[vpath.stem]) for vpath in cam_opn_dirs]

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

    skeleton = load_skeleton()
    tracker = MvTracker(skeleton)
    frm_idx = 0
    n_test = 1000
    n_test = min(len(frm_pose_paths), n_test)
    with tqdm(total=len(frm_pose_paths), desc='tracking') as bar:
        while True:
            bar.update()
            bar.set_description(
                f'tracking. n_dead = {len(tracker.dead_tracklets)}. n_tracks = {len(tracker.tracklets)}')
            oks_frames = [vreader.read() for vreader in vreaders]
            is_oks, org_frames = list(zip(*oks_frames))
            if not all(is_oks) or frm_idx >= len(frm_pose_paths):
                break
            org_frames = {view_idx: frm for view_idx, frm in enumerate(org_frames)}

            d_frames = load_pickle(frm_pose_paths[frm_idx], 'rb')
            frm_idx += 1

            # if frm_idx < 90:
            #     continue

            # for frm_img, frm_data in zip(org_frames, d_frames):
            #     draw_poses(frm_img, frm_data)
            #
            # for i in range(len(d_frames)):
            #     cv2.imshow(f'{i}', frm_img)
            # cv2.waitKeyEx(1)

            from mv_association import match_multiview_poses
            cams_poses_ids = [[p_id for p_id in frm.poses.keys()]
                              for frm in d_frames]
            cams_poses = [[d_frames[cam_idx].poses[p_id] for p_id in p_ids]
                          for cam_idx, p_ids in enumerate(cams_poses_ids)]
            cams_calibs = [frm.calib
                           for frm in d_frames]
            person_matches = match_multiview_poses(cams_poses, cams_calibs)
            pose_grps = []
            for grp_cam_pose_idxs in person_matches:
                grp = None
                for cam_idx, p_idx in grp_cam_pose_idxs:
                    calib = cams_calibs[cam_idx]
                    p_id = cams_poses_ids[cam_idx][p_idx]
                    pose = cams_poses[cam_idx][p_idx]
                    if grp is None:
                        grp = PoseAssociation(cam=calib, frame_idx=frm_idx, view_id=cam_idx, id_obj=(p_id, pose),
                                              use_weighted_kps_score=True,
                                              match_threshold=12,
                                              min_triangulate_kps_score=0.1)
                    else:
                        grp.cams.append(calib)
                        grp.view_ids.append(cam_idx)
                        grp.id_poses.append((p_id, pose))
                pose_grps.append(grp)

            debug_img_dir = '/media/F/thesis/data/debug/mv_association'
            os.makedirs(debug_img_dir, exist_ok=True)
            for p_id, grp in enumerate(pose_grps):
                viz = grp.debug_get_association_viz(org_frames)
                cv2.imwrite(f'{debug_img_dir}/{frm_idx}_{p_id}.jpg', viz)
                # cv2.imshow('test', viz)
                # k = cv2.waitKeyEx(1)
                # while k != ord('n'):
                #     k = cv2.waitKeyEx(1)

            # tracker.update(frm_idx, d_frames, debug_view_imgs=org_frames)

            if frm_idx >= n_test:
                break

    db_writer.close()

    all_tlets = tracker.tracklets + tracker.dead_tracklets
    all_tlets = sorted(all_tlets, key=lambda tlet: -len(tlet))

    with open(f'{out_dir}/traclets_shelf.pkl', 'wb') as file:
        pickle.dump(file=file, obj={"tracklets": all_tlets})

    for tlet in all_tlets:
        poses_3d = [p[-1] for p in tlet.poses]
        plot_poses_3d(poses_3d)


def viz_tracklets(in_tracklet_path):
    with open(in_tracklet_path, 'rb') as file:
        all_tlets = pickle.load(file)
        for tlet in all_tlets:
            poses_3d = [p[-1] for p in tlet.poses]
            plot_poses_3d(poses_3d)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['prepare', 'run', 'viz'],
                        help='run motion capture or prepare pre-generated data')

    parser.add_argument('--tlet_path', type=str, default='./tracklets.pkl',
                        help='tracklet pkl path that you want to visualize')

    parser.add_argument('--video_dir', type=str, default='', help='video directory')
    parser.add_argument('--data_dir', type=str, default='', help='pre-generated data directory')
    parser.add_argument('--output_dir', type=str, default='', help='output directory')

    parser.add_argument('--opn_kps_dir', type=str, default='',
                        help='openpose keypoints directory. each sub folder inside'
                             'this folder should contains keypoints corresponing videos')
    parser.add_argument('--calib_dir', type=str, default='', help='calibration directory')
    parser.add_argument('--out_data_dir', type=str, default='', help='output  data directory')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'run':
        run_main(video_dir=Path(args.video_dir),
                 pose_dir=Path(args.data_dir),
                 out_dir=Path(args.output_dir))
    elif args.mode == 'prepare':
        extract_frame_data_from_openpose(in_dir=Path(args.opn_kps_dir),
                                         calib_dir=Path(args.calib_dir),
                                         out_data_dir=Path(args.out_data_dir))
    elif args.mode == 'viz':
        viz_tracklets(args.tlet_path)
