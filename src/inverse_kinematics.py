import yaml
import numpy as np
import os
from scipy.optimize import least_squares
from typing import Optional
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from Quaternions import Quaternions
from util import descendants_mask
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pose_def import KpsType, get_parent_index, KpsFormat, get_common_kps_idxs, get_kps_index, Pose
from mv_math_util import triangulate_point_groups_from_multiple_views_linear

solver_verbose = 0
matplotlib.use('Qt5Agg')


def offsets_to_bone_dirs_bone_lens(offsets):
    bone_lens = np.linalg.norm(offsets, axis=-1)
    bdirs = offsets.copy()
    bdirs[1:, :] = bdirs[1:, :] / bone_lens[1:][:, np.newaxis]
    return bdirs, bone_lens


def bone_dir_bone_lens_to_offsets(bone_dirs, bone_lens):
    return bone_dirs * bone_lens[:, np.newaxis]


def plot_poses_3d(poses_3d: np.array, bones_idxs, target_pose, interval=50):
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    if target_pose is not None:
        for bone in bones_idxs:
            p0, p1 = target_pose[bone[0], :3], target_pose[bone[1], :3]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], c='blue')

    bones_lines = [ax.plot([0, 0], [0, 0], [0, 0])[0] for _ in range(len(bones_idxs))]

    def _update_pose(frm_idx):
        for bone, line in zip(bones_idxs, bones_lines):
            p0, p1 = poses_3d[frm_idx, bone[0], :3], poses_3d[frm_idx, bone[1], :3]
            line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
            line.set_3d_properties([p0[2], p1[2]])

    # Setting the axes properties
    ax.set_xlim3d([-5.0, 5.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-5.0, 5.0])
    ax.set_zlabel('Z')

    # Creating the Animation object
    line_anim = animation.FuncAnimation(fig, _update_pose, len(poses_3d), interval=interval, blit=False)
    plt.show()


def plot_ik_result(init_pose, pred_pose, target_pose, bone_idxs, target_bone_idxs):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    n_joints = len(init_pose)
    # for j_idx in range(n_joints):
    #     ax.scatter(init_pose[j_idx, 0], init_pose[j_idx, 1], init_pose[j_idx, 2])
    #     ax.text(init_pose[j_idx, 0], init_pose[j_idx, 1], init_pose[j_idx, 2], f'{j_idx}', size=10, zorder=1, color='k')

    for bone in bone_idxs:
        p0, p1 = init_pose[bone[0], :3], init_pose[bone[1], :3]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='red')

    for bone in bone_idxs:
        p0, p1 = pred_pose[bone[0], :3], pred_pose[bone[1], :3]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='magenta')

    if target_bone_idxs is not None:
        for bone in bone_idxs:
            p0, p1 = target_pose[bone[0], :3], target_pose[bone[1], :3]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='blue')
    else:
        ax.plot(target_pose[:, 0], target_pose[:, 1], target_pose[:, 2], '+r')

    plt.show()


def swap_y_z(poses):
    y = poses[..., 1].copy()
    z = poses[..., 2].copy()
    poses[..., 2] = y
    poses[..., 1] = z
    return poses


@dataclass
class PoseShapeParam:
    root: np.ndarray
    euler_angles: np.ndarray
    bone_lens: np.ndarray


@dataclass
class Skeleton:
    ref_joint_euler_angles: np.ndarray
    ref_bone_lens: np.ndarray
    ref_bone_dirs: np.ndarray
    n_joints: int
    joint_parents: np.ndarray
    kps_format: KpsFormat

    @property
    def skel_kps_idx_map(self):
        return get_kps_index(self.kps_format)

    @property
    def bone_idxs(self):
        bone_idxs = []
        for i, i_p in enumerate(self.joint_parents[1:]):
            bone_idxs.append((i + 1, i_p))
        return bone_idxs


def load_skeleton():
    offsets = [
        [0, 0, 0],
        [0.15, 0, 0],
        [0, 0, -0.5],
        [0, 0, -0.5],
        [-0.15, 0, 0],
        [0, 0, -0.5],
        [0, 0, -0.5],
        [0, 0, 0.3],
        [0, 0, 0.3],
        [0.2, 0, 0],
        [0.3, 0, 0],
        [0.3, 0, 0],
        [-0.2, 0, 0],
        [-0.3, 0, 0],
        [-0.3, 0, 0],
        [0, 0, 0.3],
        [0.07, 0, 0.1],
        [-0.07, 0, 0.1]
    ]

    skl_offsets, skl_parents = np.array(offsets), get_parent_index(KpsFormat.BASIC_18)
    n_joints = len(skl_parents)
    bone_idxs = []
    for i, i_p in enumerate(skl_parents[1:]):
        bone_idxs.append((i + 1, i_p))

    skel_bdirs, skel_blens = offsets_to_bone_dirs_bone_lens(skl_offsets)
    skel = Skeleton(ref_joint_euler_angles=np.zeros((n_joints, 3)),
                    ref_bone_dirs=skel_bdirs,
                    ref_bone_lens=skel_blens,
                    joint_parents=skl_parents,
                    n_joints=len(skl_parents),
                    kps_format=KpsFormat.BASIC_18)
    # return np.array(offsets), get_parent_index(KpsFormat.BASIC_18)
    return skel


def foward_kinematics(skel: Skeleton, param: PoseShapeParam):
    root_loc = param.root
    rotations = Quaternions.from_euler(param.euler_angles)

    rot_mats = rotations.transforms()
    l_transforms = np.array([np.eye(4) for _ in range(skel.n_joints)])

    offsets = bone_dir_bone_lens_to_offsets(skel.ref_bone_dirs, param.bone_lens)

    for j_i in range(skel.n_joints):
        l_transforms[j_i, :3, :3] = rot_mats[j_i]
        if j_i != 0:
            l_transforms[j_i, :3, 3] = offsets[j_i]
        else:
            if root_loc is not None:
                l_transforms[j_i, :3, 3] = root_loc

    g_transforms = l_transforms.copy()
    for j_i in range(1, skel.n_joints):
        g_transforms[j_i, :, :] = g_transforms[skel.joint_parents[j_i], :, :] @ l_transforms[j_i, :, :]

    g_pos = g_transforms[:, :, 3]
    g_pos = g_pos[:, :3] / g_pos[:, 3, np.newaxis]
    return g_pos, g_transforms


def solve_pose(skel: Skeleton,
               obs_pose_3d: np.ndarray,
               obs_kps_idxs: List[int],
               skel_kps_idxs: List[int],
               init_param: PoseShapeParam) -> PoseShapeParam:
    init_locs, _ = foward_kinematics(skel, init_param)

    target_pose_3d_shared = obs_pose_3d[obs_kps_idxs, :]

    def _decompose(_x: PoseShapeParam):
        return _x[:3], _x[3:].reshape((-1, 3))

    def _compose(p: PoseShapeParam):
        return np.concatenate([p.root.flatten(), p.euler_angles.flatten()])

    def _residual_step_joints_3d(_x):
        _root, _angles = _decompose(_x)
        _joint_locs, _ = foward_kinematics(skel, PoseShapeParam(_root, _angles, init_param.bone_lens))
        _joint_locs = _joint_locs[skel_kps_idxs, :]
        _diffs = (_joint_locs - target_pose_3d_shared[:, :3])
        _diffs = _diffs * target_pose_3d_shared[:, -1:]
        return _diffs.flatten()

    results = least_squares(_residual_step_joints_3d, _compose(init_param), verbose=solver_verbose, max_nfev=15)
    root, angles = _decompose(results.x)
    return PoseShapeParam(root, angles, init_param.bone_lens)


def solve_pose_bone_lens(skel: Skeleton,
                         obs_pose_3d: np.ndarray,
                         obs_kps_idxs: List[int],
                         skel_kps_idxs: List[int],
                         init_param: PoseShapeParam):
    target_pose_3d_shared = obs_pose_3d[obs_kps_idxs, :]
    n_joints = skel.n_joints

    def _decompose(_x):
        return _x[:3], _x[3:3 + n_joints * 3].reshape((-1, 3)), _x[3 + n_joints * 3:]

    def _compose(p: PoseShapeParam):
        return np.concatenate([p.root.flatten(), p.euler_angles.flatten(), p.bone_lens.flatten()])

    def _residual_root_angles_bone_lens(_x):
        _root, _angles, _blens = _decompose(_x)
        _joint_locs, _ = foward_kinematics(skel,
                                           PoseShapeParam(_root, _angles, _blens))
        _joint_locs = _joint_locs[skel_kps_idxs, :]
        _diffs = (_joint_locs - target_pose_3d_shared[:, :3])
        _diffs = _diffs * target_pose_3d_shared[:, -1:]
        return _diffs.flatten()

    results = least_squares(_residual_root_angles_bone_lens, _compose(init_param), verbose=solver_verbose, max_nfev=15)
    root, angles, blens = _decompose(results.x)
    return PoseShapeParam(root, angles, blens)


class PoseSolver:
    def __init__(self,
                 skeleton: Skeleton,
                 init_pose: Optional[PoseShapeParam],
                 cam_poses_2d: List[np.ndarray],
                 cam_projs: List[np.ndarray],
                 obs_kps_format: KpsFormat):
        self.skel = skeleton
        self.n_joints = self.skel.n_joints
        self.init_pose = init_pose
        self.cam_poses_2d = cam_poses_2d
        self.cam_projs = cam_projs
        self.obs_kps_format = obs_kps_format
        self.skel_kps_format = self.skel.kps_format
        self.obs_kps_idx_map = get_kps_index(self.obs_kps_format)
        self.skel_kps_idxs, self.obs_kps_idxs = get_common_kps_idxs(self.skel_kps_format, obs_kps_format)

    def solve(self) -> Tuple[PoseShapeParam, Pose]:
        obs_pose_3d = triangulate_point_groups_from_multiple_views_linear(self.cam_projs,
                                                                          self.cam_poses_2d, 0.01, True)

        if self.init_pose is None:
            init_root = 0.5 * (obs_pose_3d[self.obs_kps_idx_map[KpsType.L_Hip], :3] +
                               obs_pose_3d[self.obs_kps_idx_map[KpsType.R_Hip], :3])
            init_blens = self.skel.ref_bone_lens.copy()
            init_angles = np.zeros((self.n_joints, 3), dtype=init_root.dtype)
            init_param = PoseShapeParam(init_root, init_angles, init_blens)
        else:
            init_param = self.init_pose

        param_1 = solve_pose(self.skel, obs_pose_3d, self.obs_kps_idxs, self.skel_kps_idxs, init_param)
        param_2 = solve_pose_bone_lens(self.skel, obs_pose_3d, self.obs_kps_idxs, self.skel_kps_idxs, param_1)

        pred_locs_1, _ = foward_kinematics(self.skel, param_1)
        pred_locs_2, _ = foward_kinematics(self.skel, param_2)
        # plot_ik_result(pred_locs_1, pred_locs_2,
        #                target_pose=obs_pose_3d,
        #                bone_idxs=self.skel.bone_idxs,
        #                target_bone_idxs=None)

        return param_2, Pose(keypoints=pred_locs_2,
                             keypoints_score=np.zeros((len(pred_locs_2), 1)),
                             box=None,
                             pose_type=KpsFormat.BASIC_18)


def run_test_ik(target_pose_3d: np.ndarray, cam_poses_2d: List[np.ndarray], cam_projs: List[np.ndarray]):
    obs_kps_format = KpsFormat.COCO
    skel_kps_format = KpsFormat.BASIC_18
    obs_kps_idx_map = get_kps_index(obs_kps_format)

    skel_kps_idxs, obs_kps_idxs = get_common_kps_idxs(skel_kps_format, obs_kps_format)
    cam_poses_2d_match = [pose[obs_kps_idxs, :] for pose in cam_poses_2d]
    target_pose_3d_match = target_pose_3d[obs_kps_idxs, :]

    skl_offsets, skl_parents = load_skeleton()
    skl_descendants = descendants_mask(skl_parents)
    n_joints = len(skl_parents)
    bone_idxs = []
    for i, i_p in enumerate(skl_parents[1:]):
        bone_idxs.append((i + 1, i_p))

    skel_bdirs, skel_blens = offsets_to_bone_dirs_bone_lens(skl_offsets)
    skel = Skeleton(ref_joint_euler_angles=np.zeros((n_joints, 3)),
                    ref_bone_dirs=skel_bdirs,
                    ref_bone_lens=skel_blens,
                    joint_parents=skl_parents,
                    n_joints=len(skl_parents),
                    kps_format=KpsFormat.BASIC_18)
    solver = PoseSolver(skel, init_pose=None, cam_poses_2d=cam_poses_2d, cam_projs=cam_projs,
                        obs_kps_format=KpsFormat.COCO)
    solver.solve()

    # def _residual_step_2(_x):
    #     _root_pos = _x[:3]
    #     _joint_quars = Quaternions.from_euler(_x[3:].reshape((-1, 3)))
    #     _joint_locs, _ = model.forward(_joint_quars, _root_pos)
    #     _joint_locs = _joint_locs[skel_kps_idxs, :]
    #
    #     _joint_homos = np.concatenate([_joint_locs, np.ones((_joint_locs.shape[0], 1))], axis=-1).T
    #     _diff_reprojs = []
    #     for _vi in range(n_cams):
    #         _proj = cam_projs[_vi] @ _joint_homos
    #         _proj = (_proj[:2] / _proj[2]).T
    #         _d = np.linalg.norm(_proj - cam_poses_2d_match[_vi][:, :2], axis=-1)
    #         _d = _d * cam_poses_2d_match[_vi][:, -1]
    #         _diff_reprojs.append(_d)
    #     _diff_reprojs = np.array(_diff_reprojs).flatten()
    #     return _diff_reprojs
