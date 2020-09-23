import numpy as np
import cv2
import pickle
import subprocess
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import torch
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
                          points_to_lines_distances, calc_pairwise_f_mats, geometry_affinity, get_fundamental_matrix)
from pose_viz import plot_poses_3d
from mv_association import match_als, match_svt, match_eig, match_bip
from enum import Enum
from inverse_kinematics import PoseShapeParam, Skeleton, PoseSolver, load_skeleton
from pose_viz import draw_poses_concat, draw_pose_epiplar_lines, draw_pose
from collections import defaultdict

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
                                        min_valid_kps_score=0.1, invalid_default_error=np.nan)
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


def pose_to_bb(pose: Pose, min_valid_kps=0.1):
    valid_kps = pose.keypoints[pose.keypoints_score.flatten() > min_valid_kps, :]
    bmin, bmax = np.min(valid_kps, axis=0), np.max(valid_kps, axis=0)
    return np.concatenate([bmin, bmax])


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
class SpatialMatch:
    # view_idx, p_id
    view_idxs: List[int] = field(default_factory=list)
    pose_ids: List[int] = field(default_factory=list)

    # for debugging
    cost_matrix_idxs: List[int] = field(default_factory=list)

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
                 cam_poses_2d: List[Tuple[int, Pose]],
                 cam_projs: List[np.ndarray],
                 skel: Skeleton,
                 n_inits: int = 3,
                 max_age: int = 3):
        self.frame_idxs: List[int] = [frm_idx]
        self.cam_poses_2d: List[List[Tuple[int, Pose]]] = [cam_poses_2d]
        self.cam_projs: List[List[np.ndarray]] = [cam_projs]
        self.skel = skel
        solver = PoseSolver(skel, init_pose=None,
                            cam_poses_2d=[p[1].to_kps_array() for p in cam_poses_2d],
                            cam_projs=cam_projs,
                            obs_kps_format=cam_poses_2d[0][1].pose_type)
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

    def update(self, frm_idx: int, match: SpatialMatch, frames: List[FrameData]):
        cam_projs = [frames[v_idx].calib.P for v_idx in match.view_idxs]
        cam_poses = [(v_idx, frames[v_idx].poses[p_id]) for v_idx, p_id in zip(match.view_idxs, match.pose_ids)]

        self.frame_idxs.append(frm_idx)
        self.cam_poses_2d.append(cam_poses)
        self.cam_projs.append(cam_projs)

        solver = PoseSolver(self.skel,
                            init_pose=self.poses[-1][0],
                            cam_poses_2d=[p[1].to_kps_array() for p in cam_poses],
                            cam_projs=cam_projs,
                            obs_kps_format=cam_poses[0][1].pose_type)
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


def reprojection_error(p_3d: Pose, p_2d: Pose, calib: Calib, min_valid_kps_score=0.05, invalid_default_error=np.nan):
    p_3d_homo = np.concatenate([p_3d.keypoints, np.ones((len(p_3d.keypoints), 1))], axis=1)
    p_2d_reproj = calib.P @ p_3d_homo.T
    p_2d_reproj = (p_2d_reproj[:2] / (1e-5 + p_2d_reproj[2])).T

    score_mask = (p_2d.keypoints_score.flatten() * p_3d.keypoints_score.flatten()) > min_valid_kps_score
    if any(score_mask):
        e = np.linalg.norm(p_2d_reproj[score_mask, :2] - p_2d.keypoints[score_mask, :2], axis=-1)
        return np.mean(e)
    else:
        return invalid_default_error


def parse_match_result(match_mat_: np.ndarray, n, dims_group):
    match_mat = torch.tensor(match_mat_)
    bin_match = match_mat[:, torch.nonzero(torch.sum(match_mat, dim=0) > 1.9).squeeze()] > 0.9
    bin_match = bin_match.reshape(n, -1)
    matched_list = [[] for i in range(bin_match.shape[1])]
    for sub_img_id, row in enumerate(bin_match):
        if row.sum() != 0:
            pid = row.numpy().argmax()
            matched_list[pid].append(sub_img_id)

    dim_groups_matches = []
    for matches in matched_list:
        cur_matches = []
        for idx in matches:
            grp_offset = 0
            grp_idx = 0
            for cur_grp_idx, offset in enumerate(dims_group):
                if offset <= idx:
                    grp_offset = offset
                    grp_idx = cur_grp_idx
                else:
                    break

            local_idx = idx - grp_offset
            cur_matches.append((grp_idx, local_idx, idx))

        if cur_matches:
            dim_groups_matches.append(cur_matches)

    return dim_groups_matches


@dataclass
class SpatialTimeMatch:
    spatial_time_matches: Dict[int, SpatialMatch]  # Tracklet.track_id to view-poses
    spatial_matches: List[SpatialMatch]  # view poses with view poses

    # for debugging. mapping from tracklet_idx to cost matrix idx
    tlet_matrix_idxs: Dict[int, int] = field(default_factory=dict)
    view_pose_matrix_idxs: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    dst_mat: np.ndarray = None  # distance matrix
    sim_mat: np.ndarray = None  # similarity matrix
    match_mat: np.ndarray = None  # binarized matching matrix


g_prev_frame_images: List[np.ndarray] = []
g_cur_frame_images: List[np.ndarray] = []


def debug_draw_frame_poses(frame_idx, d_frames: List[FrameData]):
    all_vizs = []
    for view_idx, d_frame in enumerate(d_frames):
        # frame_idx -1 because tracklet belongs to the past
        img = g_cur_frame_images[view_idx].copy()
        for pose in d_frame.poses.values():
            img = draw_pose(img, pose, crop=False)
        all_vizs.append(img)
    return all_vizs


def debug_draw_dst_matrix_elemets(frame_idx, st_matches: SpatialTimeMatch, d_frames):
    st_vizs = []
    for view_idx, mat_idxs_pose_ids in st_matches.view_pose_matrix_idxs.items():
        poses = [d_frames[view_idx].poses[p_id] for p_id, mat_idx in mat_idxs_pose_ids]
        cam_imgs = [g_cur_frame_images[view_idx] for _ in range(len(poses))]
        debug_texts = [f'{mat_idx}' for p_id, mat_idx in mat_idxs_pose_ids]
        viz_1 = draw_poses_concat(poses, cam_imgs, debug_texts,
                                  top_text=f'{frame_idx}',
                                  crop_height=256)
        st_vizs.append(viz_1)

    return st_vizs


def debug_draw_spatial_time_matches(frame_idx,
                                    st_matches: SpatialTimeMatch,
                                    tracklets: List[MvTracklet],
                                    d_frames: List[FrameData]):
    st_vizs = []
    crop_height = 256

    for tlet_idx, s_match in st_matches.spatial_time_matches.items():
        tlet = tracklets[tlet_idx]
        tlet_poses = [x[1] for x in tlet.cam_poses_2d[-1]]
        tlet_cam_idxs = [x[0] for x in tlet.cam_poses_2d[-1]]
        tlet_cam_imgs = [g_prev_frame_images[c_idx] for c_idx in tlet_cam_idxs]

        poses = [d_frames[c_idx].poses[p_id] for c_idx, p_id in zip(s_match.view_idxs, s_match.pose_ids)]
        cam_idxs = s_match.view_idxs
        cam_imgs = [g_cur_frame_images[c_idx] for c_idx in cam_idxs]

        # frame_idx -1 because tracklet belongs to the past
        viz_0 = draw_poses_concat(tlet_poses, tlet_cam_imgs, None,
                                  top_text=f'{frame_idx - 1}_{st_matches.tlet_matrix_idxs[tlet_idx]}',
                                  crop_height=crop_height)

        if s_match.cost_matrix_idxs:
            debug_texts = [f'{c_idx}_{g_idx}' for c_idx, g_idx in zip(s_match.view_idxs, s_match.cost_matrix_idxs)]
        else:
            debug_texts = [f'{c_idx}' for c_idx in s_match.view_idxs]
        viz_1 = draw_poses_concat(poses, cam_imgs, debug_texts,
                                  top_text=f'{frame_idx}',
                                  crop_height=crop_height)

        if viz_0 is not None and viz_1 is not None:
            max_width = max(viz_0.shape[1], viz_1.shape[1])
            viz = np.zeros((crop_height * 2, max_width, 3), dtype=viz_0.dtype)
            viz[:crop_height, :viz_0.shape[1], :] = viz_0
            viz[crop_height:, :viz_1.shape[1], :] = viz_1
            st_vizs.append(viz)

    for s_match in st_matches.spatial_matches:
        poses = [d_frames[c_idx].poses[p_id] for c_idx, p_id in zip(s_match.view_idxs, s_match.pose_ids)]
        cam_idxs = s_match.view_idxs
        cam_imgs = [g_cur_frame_images[c_idx] for c_idx in cam_idxs]

        if s_match.cost_matrix_idxs:
            debug_texts = [f'{c_idx}_{g_idx}' for c_idx, g_idx in zip(s_match.view_idxs, s_match.cost_matrix_idxs)]
        else:
            debug_texts = [f'{c_idx}' for c_idx in s_match.view_idxs]
        viz_1 = draw_poses_concat(poses, cam_imgs, debug_texts,
                                  top_text=f'{frame_idx}',
                                  crop_height=crop_height)
        st_vizs.append(viz_1)

    return st_vizs


def match_spatial(frames: List[FrameData]) -> SpatialTimeMatch:
    cam_poses_id = [[pose for pose in frm.poses.keys()]
                    for frm in frames]
    cams_poses = [[frames[cam_idx].poses[p_id] for p_id in pose_ids]
                  for cam_idx, pose_ids in enumerate(cam_poses_id)]
    cams_calib = [frm.calib for frm in frames]
    pairwise_f_mats = calc_pairwise_f_mats(cams_calib)

    points_set = []
    dim_groups = [0]
    cnt = 0
    for poses in cams_poses:
        cnt += len(poses)
        dim_groups.append(cnt)
        for p in poses:
            points_set.append(p.keypoints)
    points_set = np.array(points_set)
    # build matrix
    dst_mat, s_mat = geometry_affinity(points_set, pairwise_f_mats, dim_groups)
    match_mat, _ = match_als(s_mat, dim_groups)
    dim_groups_matches = parse_match_result(match_mat, s_mat.shape[0], dim_groups)

    out_matches = SpatialTimeMatch({}, [])
    for grp_cam_pose_idxs in dim_groups_matches:
        s_match = SpatialMatch([], [])
        for cam_idx, p_idx, _ in grp_cam_pose_idxs:
            s_match.view_idxs.append(cam_idx)
            s_match.pose_ids.append(cam_poses_id[cam_idx][p_idx])
        out_matches.spatial_matches.append(s_match)

    # debug
    out_matches.dst_mat = dst_mat
    out_matches.sim_mat = s_mat

    return out_matches


def match_spatial_time(tlets: List[MvTracklet],
                       frames: List[FrameData],
                       pixel_error_threshold) -> SpatialTimeMatch:
    """
    :param tlets:
    :param frames:
    :param pixel_error_threshold:  match with error greater than this threshold will be discarded
    :return:
    """
    cam_poses_id = [[pose for pose in frm.poses.keys()]
                    for frm in frames]
    cams_poses = [[frames[cam_idx].poses[p_id] for p_id in pose_ids]
                  for cam_idx, pose_ids in enumerate(cam_poses_id)]
    cams_calib = [frm.calib for frm in frames]

    out_matches = SpatialTimeMatch({}, [])

    poses_3ds_2ds: List[Pose] = [tlet.last_pose_3d for tlet in tlets]
    pose_mask = ['3d'] * len(tlets)

    mat_idx_to_cam_idxs = [-1] * len(tlets)
    mat_idx_to_tracklet_ids = [tlet_idx for tlet_idx in range(len(tlets))]
    mat_idx_to_poses_2d_ids = [-1] * len(tlets)

    part_lens = [0, len(tlets)]
    for cam_idx, poses in enumerate(cams_poses):
        poses_3ds_2ds.extend(poses)
        pose_mask.extend(['2d'] * len(poses))
        part_lens.append(len(poses))
        mat_idx_to_cam_idxs.extend([cam_idx] * len(poses))
        mat_idx_to_poses_2d_ids.extend(cam_poses_id[cam_idx])
        mat_idx_to_tracklet_ids.extend([-1] * len(poses))

    dim_groups = np.cumsum(part_lens).tolist()

    INVALID_VALUE = np.nan

    n_poses = len(poses_3ds_2ds)
    dst_mat = np.zeros((n_poses, n_poses), dtype=np.float)
    for mat_idx in range(n_poses):
        for j in range(n_poses):
            # same pose
            if mat_idx == j:
                continue

            # two 3d poses

            # same view
            if mat_idx_to_cam_idxs[mat_idx] >= 0 and mat_idx_to_cam_idxs[mat_idx] == mat_idx_to_cam_idxs[j]:
                dst_mat[mat_idx, j] = INVALID_VALUE
                continue

            if pose_mask[mat_idx] == '2d' and pose_mask[j] == '2d':
                # epipolar error
                assert mat_idx_to_cam_idxs[mat_idx] >= 0 and mat_idx_to_cam_idxs[j] >= 0
                i_calib = cams_calib[mat_idx_to_cam_idxs[mat_idx]]
                j_calib = cams_calib[mat_idx_to_cam_idxs[j]]
                i_pose = poses_3ds_2ds[mat_idx]
                j_pose = poses_3ds_2ds[j]

                e_error = calc_epipolar_error(i_calib, i_pose.keypoints, i_pose.keypoints_score,
                                              j_calib, j_pose.keypoints, j_pose.keypoints_score,
                                              0.1, INVALID_VALUE)

                # img_w, img_h = 1032, 776
                # img1 = g_cur_frame_images[mat_idx_to_cam_idxs[i]].copy()
                # img2 = g_cur_frame_images[mat_idx_to_cam_idxs[j]].copy()
                # img1, img2 = draw_pose_epiplar_lines(img1, img2, i_pose, j_pose, get_fundamental_matrix(i_calib.P, j_calib.P))
                # plt.subplot(121)
                # plt.imshow(img1[:,:, ::-1])
                # plt.subplot(122)
                # plt.imshow(img2[:,:, ::-1])
                # plt.show()
                dst_mat[mat_idx, j] = e_error

            elif pose_mask[mat_idx] == '2d' and pose_mask[j] == '3d':
                # re-projection error
                p_2d = poses_3ds_2ds[mat_idx]
                calib = cams_calib[mat_idx_to_cam_idxs[mat_idx]]
                p_3d = poses_3ds_2ds[j]
                e_error = reprojection_error(p_3d, p_2d, calib, 0.1, INVALID_VALUE)
                dst_mat[mat_idx, j] = e_error

            elif pose_mask[mat_idx] == '3d' and pose_mask[j] == '2d':
                # re-projection error
                p_2d = poses_3ds_2ds[j]
                calib = cams_calib[mat_idx_to_cam_idxs[j]]
                p_3d = poses_3ds_2ds[mat_idx]
                e_error = reprojection_error(p_3d, p_2d, calib, 0.1, INVALID_VALUE)
                dst_mat[mat_idx, j] = e_error

            elif pose_mask[mat_idx] == '3d' and pose_mask[j] == '3d':
                dst_mat[mat_idx, j] = INVALID_VALUE

            else:
                # no match here
                pass

    # applying hard threshold
    # max_dst_value = np.nanmax(dst_mat)
    max_dst_value = np.nanmax(dst_mat)  # in pixel unit. need to adjust for different resolutions
    dst_mat[np.isnan(dst_mat)] = max_dst_value + 1.0

    # valid_values_mask = np.bitwise_and(dst_mat < 30.0, dst_mat > 0.01)
    # valid_values = dst_mat[valid_values_mask]
    # mean, std = np.mean(valid_values), np.std(valid_values)
    mean, std = 7, 5 # TODO: adjust it for different resolutions
    s_mat = - (dst_mat - mean) / std

    # TODO: add flexible factor
    s_mat = 1 / (1 + np.exp(-5 * s_mat))

    # match_mat, x_bin = match_bip(s_mat, 0.3)
    match_mat, x_bin = match_als(s_mat, dim_groups)
    # match_mat, x_bin = match_svt(s_mat, dim_groups)
    # match_mat, x_bin = match_eig(s_mat, dim_groups)

    dim_groups_matches = parse_match_result(match_mat, s_mat.shape[0], dim_groups)
    for cur_matches in dim_groups_matches:
        tracklet_idx = -1
        tracklet_global_idx = -1
        # first check if there is a tracklet [time-dim] in the match
        for grp_idx, local_idx, global_idx in cur_matches:
            if pose_mask[global_idx] == '3d':
                tracklet_idx = mat_idx_to_tracklet_ids[global_idx]
                tracklet_global_idx = global_idx
                break

        if tracklet_idx >= 0:
            # ok, we have a match between a tracklet and 2d poses
            s_match = SpatialMatch([], [])
            for grp_idx, local_idx, global_idx in cur_matches:
                if pose_mask[global_idx] == '2d':
                    view_idx = mat_idx_to_cam_idxs[global_idx]
                    pose_id = mat_idx_to_poses_2d_ids[global_idx]
                    # TODO: hack. make sure that only one pose per view is selected
                    if view_idx in s_match.view_idxs:
                        print('3d-2d matching: more than one pose per view is detected')
                        continue
                    s_match.view_idxs.append(view_idx)
                    s_match.pose_ids.append(pose_id)
                    # for debugging
                    s_match.cost_matrix_idxs.append(global_idx)

            if len(s_match) > 0:
                out_matches.spatial_time_matches[tracklet_idx] = s_match
                # for debugging
                out_matches.tlet_matrix_idxs[tracklet_idx] = tracklet_global_idx

        else:
            # the match consists of only 2d poses
            s_match = SpatialMatch([], [])
            for grp_idx, local_idx, global_idx in cur_matches:
                if pose_mask[global_idx] == '2d':
                    view_idx = mat_idx_to_cam_idxs[global_idx]
                    pose_id = mat_idx_to_poses_2d_ids[global_idx]
                    # @TODO: hack. make sure that only one pose per view is selected
                    if view_idx in s_match.view_idxs:
                        print('2d-2d matching: more than one pose per view is detected')
                        continue
                    s_match.view_idxs.append(view_idx)
                    s_match.pose_ids.append(pose_id)
                    # for debugging
                    s_match.cost_matrix_idxs.append(global_idx)
                else:
                    assert False, "expect that there are only 2d poses in this assocations"
            if len(s_match) > 0:
                out_matches.spatial_matches.append(s_match)

    # debug
    debug = True
    if debug:
        # save matrix index of (view_idx, p_id) pair for debugging
        for mat_idx in range(n_poses):
            cam_idx = mat_idx_to_cam_idxs[mat_idx]
            if cam_idx >= 0:
                p_id = mat_idx_to_poses_2d_ids[mat_idx]
                if cam_idx not in out_matches.view_pose_matrix_idxs:
                    out_matches.view_pose_matrix_idxs[cam_idx] = []
                out_matches.view_pose_matrix_idxs[cam_idx].append((p_id, mat_idx))

        out_matches.dst_mat = dst_mat
        out_matches.sim_mat = s_mat
        out_matches.match_mat = x_bin

    return out_matches


def associate_tracking(tlets: List[MvTracklet],
                       frames: List[FrameData],
                       min_pixel_error_hard_threshold) -> SpatialTimeMatch:
    if tlets:
        return match_spatial_time(tlets, frames, min_pixel_error_hard_threshold)
    else:
        return match_spatial(frames)


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

    def update_4d(self, frm_idx: int, d_frames: List[FrameData], debug_view_imgs: List[np.ndarray]):
        for tlet in self.tracklets:
            tlet.predict()

        # only do association with alive tracks
        alive_tracklets = [tlet for tlet in self.tracklets if not tlet.is_dead()]

        if frm_idx == 110:
            debug = True
        else:
            debug = False

        st_matches = associate_tracking(alive_tracklets, d_frames, min_pixel_error_hard_threshold=50)

        debug = True
        if debug:
            matches_vizs = debug_draw_spatial_time_matches(frm_idx, st_matches, alive_tracklets, d_frames)
            debug_dir = '/media/F/thesis/data/debug/st_match'
            os.makedirs(debug_dir, exist_ok=True)
            for idx, viz in enumerate(matches_vizs):
                if viz is not None:
                    cv2.imwrite(f'{debug_dir}/{frm_idx}_{idx}.jpg', viz)

            # debug_dir = '/media/F/thesis/data/debug/frame_poses'
            # os.makedirs(debug_dir, exist_ok=True)
            # frames_poses_vizs = debug_draw_frame_poses(frm_idx, d_frames)
            # for idx, viz in enumerate(frames_poses_vizs):
            #     cv2.imwrite(f'{debug_dir}/{frm_idx}_{idx}.jpg', viz)

            debug_dir = '/media/F/thesis/data/debug/frame_poses'
            os.makedirs(debug_dir, exist_ok=True)
            frames_poses_vizs = debug_draw_dst_matrix_elemets(frm_idx, st_matches, d_frames)
            for idx, viz in enumerate(frames_poses_vizs):
                cv2.imwrite(f'{debug_dir}/{frm_idx}_{idx}.jpg', viz)

            def _add_idx_to_mat(mat_: np.ndarray):
                """
                for visualizing correct index in excel
                :param mat_:
                :return:
                """
                return mat_

            debug_dir = '/media/F/thesis/data/debug/cost_matrix'
            os.makedirs(debug_dir, exist_ok=True)
            if st_matches.dst_mat is not None:
                filepath = f'{debug_dir}/{frm_idx}_dst.xlsx'
                pd.DataFrame(_add_idx_to_mat(st_matches.dst_mat)).to_excel(filepath, index=True)
            if st_matches.sim_mat is not None:
                filepath = f'{debug_dir}/{frm_idx}_sim.xlsx'
                pd.DataFrame(_add_idx_to_mat(st_matches.sim_mat)).to_excel(filepath, index=True)

            if st_matches.sim_mat is not None:
                filepath = f'{debug_dir}/{frm_idx}_match_binarized.xlsx'
                pd.DataFrame(_add_idx_to_mat(st_matches.match_mat)).to_excel(filepath, index=True)

        # spatial time matches
        for t_idx, tlet in enumerate(alive_tracklets):
            if t_idx in st_matches.spatial_time_matches:
                s_match = st_matches.spatial_time_matches[t_idx]
                if len(s_match) >= 2:
                    alive_tracklets[t_idx].update(frm_idx, s_match, d_frames)
                elif len(s_match) == 1:
                    # TODO: handle a single new pose
                    # alive_tracklets[t_idx].update(frm_idx, s_match, d_frames)
                    pass
            else:
                tlet.mark_missed()

        # spatial matches
        n_s_matches = len(st_matches.spatial_matches)
        for idx in range(n_s_matches):
            s_match = st_matches.spatial_matches[idx]
            if len(s_match) >= 2:
                pose_2ds = [(v_idx, d_frames[v_idx].poses[p_id]) for v_idx, p_id in
                            zip(s_match.view_idxs, s_match.pose_ids)]
                cam_projs = [d_frames[v_idx].calib.P for v_idx in s_match.view_idxs]
                tlet = MvTracklet(frm_idx, cam_poses_2d=pose_2ds, cam_projs=cam_projs, skel=self.skeleton)
                p_3d = tlet.last_pose_3d
                p_erros = []
                for v_idx, p_2d in pose_2ds:
                    e = reprojection_error(p_3d, p_2d, d_frames[v_idx].calib, min_valid_kps_score=0.05)
                    p_erros.append(e)
                mean_reproj_e = np.mean(p_erros)
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
    global g_cur_frame_images
    global g_prev_frame_images

    vpaths = sorted([vpath for vpath in video_dir.glob('*.*')], key=lambda path: path.stem)
    vreaders = [cv2.VideoCapture(str(vpath)) for vpath in vpaths]
    frm_pose_paths = sorted([ppath for ppath in pose_dir.glob('*.pkl')], key=lambda path: int(path.stem))
    vpath = '/media/F/thesis/data/debug/test.mp4'
    db_writer = imageio.get_writer(vpath)

    skeleton = load_skeleton()
    tracker = MvTracker(skeleton)
    frm_idx = 0
    n_test = 300
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

            g_cur_frame_images = org_frames

            d_frames: List[FrameData] = load_pickle(frm_pose_paths[frm_idx], 'rb')
            frm_idx += 1

            # if frm_idx < 90:
            #     continue

            test_spatial = False
            if test_spatial:
                s_matches = match_spatial(d_frames)
                for match_idx, s_match in enumerate(s_matches.spatial_matches):
                    poses_2ds = []
                    imgs = []
                    for v_idx, p_id in zip(s_match.view_idxs, s_match.pose_ids):
                        poses_2ds.append(d_frames[v_idx].poses[p_id])
                        imgs.append(org_frames[v_idx])
                    match_viz = draw_poses_concat(poses_2ds, imgs, text_per_poses=s_match.view_idxs, frm_idx=frm_idx)
                    debug_img_dir = '/media/F/thesis/data/debug/mv_association'
                    os.makedirs(debug_img_dir, exist_ok=True)
                    cv2.imwrite(f'{debug_img_dir}/{frm_idx}_{match_idx}.jpg', match_viz)
                    # cv2.imshow('test', match_viz)
                    # k = cv2.waitKeyEx(1)
                    # while k != ord('n'):
                    #     k = cv2.waitKeyEx(1)
            else:
                tracker.update_4d(frm_idx, d_frames, debug_view_imgs=org_frames)

            g_prev_frame_images = g_cur_frame_images

            if frm_idx >= n_test:
                break

    db_writer.close()

    all_tlets = tracker.tracklets + tracker.dead_tracklets
    all_tlets = sorted(all_tlets, key=lambda tlet: -len(tlet))

    # for tlet in all_tlets:
    #     poses_3d = [p[-1] for p in tlet.poses]
    #     plot_poses_3d(poses_3d)

    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/tracklets.pkl', 'wb') as file:
        pickle.dump(file=file, obj={"tracklets": all_tlets})


def video_to_images(vpath: Path, img_dir: Path, img_ext: str):
    out = subprocess.run(['ffmpeg',
                          '-y',
                          '-i', str(vpath),
                          '-hide_banner',
                          f'{img_dir}/%012d.{img_ext}'])
    print(f"video_to_images: {out}")
    return [ipath for ipath in img_dir.glob(f'*.{img_ext}')]


def draw_tracklet(tlet: MvTracklet, mv_img_paths: List[Dict[int, Path]], out_dir: Path):
    h = 256
    for f_idx, frm_poses in zip(tlet.frame_idxs, tlet.cam_poses_2d):
        all_crops = []
        for view_idx, pose in frm_poses:
            valid_kps = pose.keypoints[pose.keypoints_score.flatten() > 0.1, :]
            bmin, bmax = np.min(valid_kps, axis=0), np.max(valid_kps, axis=0)
            x1, y1, x2, y2 = np.concatenate([bmin, bmax]).astype(np.int)

            if f_idx not in mv_img_paths[view_idx]:
                print(f'view {view_idx} missing frames {f_idx}')
                continue
            img_path = mv_img_paths[view_idx][f_idx]
            img = cv2.imread(str(img_path))
            if y2 > y1 + 5 and x2 > x1 + 5:
                crop = img[y1:y2, x1:x2]
                c_h, c_w = crop.shape[:2]
                new_h = h
                new_w = int((c_w / c_h) * new_h)
                crop = cv2.resize(crop, dsize=(new_w, new_h))
                cv2.putText(crop, f'{view_idx}', (int(0.5 * new_w), 128),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
                all_crops.append(crop)

        # view_id increasing from left to right
        if all_crops:
            # sorted_view_ids = np.argsort(self.view_ids)
            # all_crops = [all_crops[idx] for idx in sorted_view_ids]

            viz = np.concatenate(all_crops, axis=1)
            cv2.putText(viz, f'{f_idx}', (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
            cv2.imwrite(f'{out_dir}/{f_idx}.jpg', viz)


def viz_tracklets(in_tracklet_path, video_dir: Path, out_dir: Path):
    import tempfile

    with open(in_tracklet_path, 'rb') as file:
        all_tlets: List[MvTracklet] = pickle.load(file)["tracklets"]

    with tempfile.TemporaryDirectory() as view_img_dir:
        view_img_dir = Path(view_img_dir)
        vpaths = sorted([vdir for vdir in video_dir.glob('*.*')], key=lambda x_: int(x_.stem))
        view_imgs_paths = []
        for v_idx, vpath in enumerate(vpaths):
            img_dir = view_img_dir / vpath.stem
            os.makedirs(img_dir, exist_ok=True)
            video_to_images(vpath, img_dir, 'jpg')
            img_paths = {int(ipath.stem): ipath for ipath in img_dir.glob('*.jpg')}
            view_imgs_paths.append(img_paths)

        for t_idx, tlet in enumerate(all_tlets):
            out_tlet_viz = out_dir / f'tlet_{t_idx}'
            os.makedirs(out_tlet_viz, exist_ok=True)
            draw_tracklet(tlet, view_imgs_paths, out_dir=out_tlet_viz)
            # poses_3d = [p[-1] for p in tlet.poses]
            # plot_poses_3d(poses_3d)


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
        out_tlet_viz_dir = '/media/F/datasets/shelf/debug/tracklets_viz'
        os.makedirs(out_tlet_viz_dir, exist_ok=True)
        viz_tracklets(args.tlet_path, Path(args.video_dir), Path(out_tlet_viz_dir))
