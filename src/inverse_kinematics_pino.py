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
import cv2
from util import descendants_mask
import subprocess
import time
from typing import List, Dict, Tuple
import copy
from pathlib import Path
from dataclasses import dataclass
from pose_def import (KpsType, get_parent_index, KpsFormat, get_common_kps_idxs, get_common_kps_idxs_1, get_kps_index,
                      Pose, get_sides_joint_idxs, get_sides_joints, get_kps_order, get_joint_side, get_flip_joint)
from mv_math_util import triangulate_point_groups_from_multiple_views_linear
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from os.path import join
from common import Calib
import pinocchio as pin
from qpsolvers import solve_qp
from pinocchio import SE3 as SE3
from pinocchio import Inertia as Inertia
from pinocchio import rpy as rpy

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


def plot_ik_result(pose_bones_colors):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    for (pose, bone_idxs, c) in pose_bones_colors:
        if bone_idxs is not None:
            for bone in bone_idxs:
                p0, p1 = pose[bone[0], :3], pose[bone[1], :3]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=c)
        else:
            ax.plot(pose[:, 0], pose[:, 1], pose[:, 2], '+r')
    plt.show()


def swap_y_z(poses):
    y = poses[..., 1].copy()
    z = poses[..., 2].copy()
    poses[..., 2] = y
    poses[..., 1] = z
    return poses


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
        [0, -0.02, 0.15],
        [+0.07, 0.02, 0.1],
        [-0.07, 0.02, 0.1]
    ]

    kps_format = KpsFormat.BASIC_18

    skl_offsets, skl_parents = np.array(offsets), get_parent_index(kps_format)
    n_joints = len(skl_parents)
    bone_idxs = []
    for i, i_p in enumerate(skl_parents[1:]):
        bone_idxs.append((i + 1, i_p))

    skel_bdirs, skel_blens = offsets_to_bone_dirs_bone_lens(skl_offsets)

    kps_idx_map = get_kps_index(kps_format)
    ljoints, rjoints, mjoints = get_sides_joints(kps_format)
    l_m_joints = ljoints + mjoints
    l_m_skel_blens = [skel_blens[kps_idx_map[j_type]] for j_type in l_m_joints]
    # mapping from index of l_m_skel_blens to full bone length list.
    l_m_to_full_map = []
    for jnt_type, jnt_idx in kps_idx_map.items():
        jnt_side = get_joint_side(jnt_type)
        if jnt_side in ['left', 'mid']:
            l_m_to_full_map.append(l_m_joints.index(jnt_type))
        else:
            jnt_type = get_flip_joint(jnt_type)
            l_m_to_full_map.append(l_m_joints.index(jnt_type))

    skel = Skeleton(ref_joint_euler_angles=np.zeros((n_joints, 3)),
                    ref_bone_dirs=skel_bdirs,
                    ref_side_bone_lens=np.array(l_m_skel_blens),
                    ref_side_to_full_bone_lens_map=l_m_to_full_map,
                    joint_parents=skl_parents,
                    n_joints=len(skl_parents),
                    kps_format=kps_format)
    return skel


@dataclass
class PoseShapeParam:
    joint_placements: np.ndarray
    pose: np.ndarray


class HumanModel:
    robot: RobotWrapper

    def __init__(self, package_dir: Path):
        filename = f"{package_dir}/romeo_description/urdf/romeo_khanh.urdf"
        self.robot = RobotWrapper.BuildFromURDF(filename, package_dirs=package_dir,
                                                root_joint=pin.JointModelFreeFlyer())
        self.data = self.robot.data
        self.model = self.robot.model

        # joints list for getting forward location for IK optimization
        joint_lists = {
            "LShoulderYaw": KpsType.L_Shoulder, "RShoulder"
                                                "Yaw": KpsType.R_Shoulder,
            "LElbowRoll": KpsType.L_Elbow, "RElbowRoll": KpsType.R_Elbow,
            "LWristPitch": KpsType.L_Wrist, 'RWristPitch': KpsType.R_Wrist,
            "RHipYaw": KpsType.R_Hip, "LHipYaw": KpsType.L_Hip,
            "LKneePitch": KpsType.L_Knee, "RKneePitch": KpsType.R_Knee,
            "LAnkleRoll": KpsType.L_Ankle, "RAnkleRoll": KpsType.R_Ankle,
        }

        # joints whose rest placements will be optimized.
        blens_joint_lists = {
            "LShoulderPitch": KpsType.L_Shoulder, "RShoulderPitch": KpsType.R_Shoulder,
            "LElbowYaw": KpsType.L_Elbow, "RElbowYaw": KpsType.R_Elbow,
            "LWristRoll": KpsType.L_Wrist, "RWristRoll": KpsType.R_Wrist,
            "LHipYaw": KpsType.L_Hip, "RHipYaw": KpsType.R_Hip,
            "LKneePitch": KpsType.L_Knee, "RKneePitch": KpsType.R_Knee,
            "LAnklePitch": KpsType.L_Ankle, "RAnklePitch": KpsType.R_Ankle,
        }

        self.target_joint_pino_ids = []
        self.target_joint_types = []

        self.opt_rest_joint_ids = []
        self.opt_rest_joint_types = []
        self.rest_bone_dirs = []

        for joint in self.model.joints:
            if joint.idx_q <= 0:
                continue
            j_id = joint.id
            j_name = self.model.names[joint.id]
            # print(j_name, self.model.names[self.model.parents[j_idx]], self.model.joints[j_idx].nq)
            if j_name in joint_lists:
                self.target_joint_pino_ids.append(j_id)
                self.target_joint_types.append(joint_lists[j_name])

            if j_name in blens_joint_lists:
                self.opt_rest_joint_ids.append(j_id)
                self.opt_rest_joint_types.append(blens_joint_lists[j_name])

                plc = self.model.jointPlacements[j_id]
                par_plc = self.model.jointPlacements[self.model.parents[j_id]]
                bdir = par_plc.translation - plc.translation
                bdir = bdir / np.linalg.norm(bdir) if np.linalg.norm(bdir) > 0.0 else bdir
                self.rest_bone_dirs.append(bdir)

        assert len(self.target_joint_pino_ids) == len(joint_lists)
        assert len(self.opt_rest_joint_types) == len(blens_joint_lists)

    def get_kps_idx_map(self) -> Dict[KpsType, int]:
        return {kps_type: idx for idx, kps_type in enumerate(self.target_joint_types)}

    def get_kps_types(self) -> List[KpsType]:
        return self.target_joint_types

    def forward_kinematics(self, x: np.ndarray):
        pin.forwardKinematics(self.model, self.data, x)
        locs = [self.data.oMi[j_id].translation for j_id in self.target_joint_pino_ids]
        return np.array(locs)


@dataclass
class Skeleton:
    ref_joint_euler_angles: np.ndarray
    ref_bone_dirs: np.ndarray
    ref_side_bone_lens: np.ndarray  # left side plus mid bone lengths
    ref_side_to_full_bone_lens_map: List[int]
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

    def to_full_bone_lens(self, side_blens):
        assert len(side_blens) == len(self.ref_side_bone_lens)
        return np.array([side_blens[idx] for idx in self.ref_side_to_full_bone_lens_map])


def image_jacobian(P: np.ndarray, proj_homo: np.ndarray):
    """
    :param P: 3x4 projection matrix
    :param proj_homo: result of Px[x,y,z,1]
    :return:
    """
    u, v, w = proj_homo
    w2 = w * w
    proj_jac = [[(P[0, 0] * w - P[2, 0] * u) / w2, (P[0, 1] * w - P[2, 1] * u) / w2,
                 (P[0, 2] * w - P[2, 2] * u) / w2],
                [(P[1, 0] * w - P[2, 0] * v) / w2, (P[1, 1] * w - P[2, 1] * v) / w2,
                 (P[1, 2] * w - P[2, 2] * v) / w2]]
    proj_jac = np.array(proj_jac)
    return proj_jac


def solve_bone_length_from_2d(skel: HumanModel,
                              obs_cam_pose_2d: List[np.ndarray],
                              cam_calibs: List[Calib],
                              obs_pose_3d: np.ndarray,
                              obs_kps_idxs: List[int],
                              skel_kps_idxs: List[int],
                              init_param: PoseShapeParam,
                              vis: bool,
                              n_max_iter=5) -> PoseShapeParam:
    opt_rest_joint_ids = skel.opt_rest_joint_ids
    rest_bone_dirs = skel.rest_bone_dirs

    if vis:
        skel.robot.display(init_param.pose)
        skel.robot.viewer.gui.refresh()

        coco_kps_types = get_kps_order(KpsFormat.COCO)
        for tar_idx in obs_kps_idxs:
            goal = pin.SE3.Identity()
            goal.translation = obs_pose_3d[tar_idx, :3]
            gname = f'world/goal_{coco_kps_types[tar_idx].name}'
            skel.robot.viewer.gui.addXYZaxis(gname, [1., 0., 0., 1.], .015, 0.1)
            skel.robot.viewer.gui.applyConfiguration(gname, se3ToXYZQUAT(goal))
            skel.robot.viewer.gui.refresh()

    n_views = len(cam_calibs)
    DT = 1e-1  # damping factor
    cur_iter = 0
    lm_damping = 1e-3  # damping factor
    impr_stop = 1e-6  # ending condition. relative error changes
    cost = 1000
    model, data = skel.model, skel.data
    x = init_param.pose.copy()
    pin_joint_ids = [skel.target_joint_pino_ids[idx] for idx in skel_kps_idxs]
    while cur_iter < n_max_iter:
        cur_iter += 1
        pin.forwardKinematics(model, data, x)

        nv = len(opt_rest_joint_ids) * 3
        ATA = np.zeros((nv, nv))
        ATb = np.zeros((nv,))
        all_errors = []

        for v_idx in range(n_views):
            calib = cam_calibs[v_idx]
            obs_pose = obs_cam_pose_2d[v_idx]
            obs_pose = obs_pose[obs_kps_idxs, :]
            p_3ds = np.array([data.oMi[j_id].translation for j_id in pin_joint_ids])
            p_3ds_homo = np.concatenate([p_3ds, np.ones((len(p_3ds), 1))], axis=-1)
            p_3ds_cam = (calib.Rt @ p_3ds_homo.T).T
            p_2ds_homo = (calib.K @ p_3ds_cam.T)
            p_2ds = (p_2ds_homo[:2] / p_2ds_homo[-1]).T
            p_2ds_homo = p_2ds_homo.T
            n_kps = len(pin_joint_ids)
            for i in range(n_kps):
                jac = pin.computeJointKinematicRegressor(model, data, pin_joint_ids[i],
                                                         pin.LOCAL_WORLD_ALIGNED)[:3, :]
                jac = np.concatenate([jac[:, (i - 1) * 6:(i - 1) * 6 + 3] for i in opt_rest_joint_ids], axis=1)

                kps_2d = p_2ds[i]

                proj_jac = image_jacobian(calib.P, p_2ds_homo[i])

                jac = proj_jac @ jac
                err = obs_pose[i, :2] - kps_2d
                err = -err
                mu = lm_damping * max(1e-3, err.dot(err))
                ATA += np.dot(jac.T, jac) + mu * np.eye(nv)
                ATb += np.dot(-err.T, jac)
                all_errors.append(err)

        prev_cost = cost
        cost = np.mean(np.concatenate(np.abs(all_errors)))
        impr = abs(cost - prev_cost) / prev_cost
        if impr < impr_stop:
            break

        dxs = -solve_qp(ATA, ATb)

        dxs = 0.1 * dxs
        for idx, j_id in enumerate(opt_rest_joint_ids):
            plc = model.jointPlacements[j_id]
            bdir = rest_bone_dirs[idx]
            dst = bdir.dot(dxs[idx * 3:(idx + 1) * 3])
            disp = bdir * dst
            plc.translation = plc.translation + disp

        if vis:
            skel.robot.display(x)
            skel.robot.viewer.gui.refresh()
            time.sleep(0.1)

    init_param.pose = x

    return init_param


def solve_pose_from_2d(skel: HumanModel,
                       obs_cam_pose_2d: List[np.ndarray],
                       cam_calibs: List[Calib],
                       obs_pose_3d: np.ndarray,
                       obs_kps_idxs: List[int],
                       skel_kps_idxs: List[int],
                       init_param: PoseShapeParam,
                       vis: bool,
                       n_max_iter=5) -> PoseShapeParam:
    out_debug_dir = "/media/F/thesis/real-time-motion-capture/yolo-tensorrt/samples/"
    file = cv2.FileStorage(f'{out_debug_dir}/test_poses.yaml', cv2.FileStorage_WRITE)
    file.write('pose_3d', obs_pose_3d)
    for idx, (pose, calib) in enumerate(zip(obs_cam_pose_2d, cam_calibs)):
        file.write(f'pose2d_{idx}', pose)
        file.write(f'cam_{idx}', calib.P)
    file.release()

    if vis:
        skel.robot.display(init_param.pose)
        skel.robot.viewer.gui.refresh()

        coco_kps_types = get_kps_order(KpsFormat.COCO)
        for tar_idx in obs_kps_idxs:
            goal = pin.SE3.Identity()
            goal.translation = obs_pose_3d[tar_idx, :3]
            gname = f'world/goal_{coco_kps_types[tar_idx].name}'
            skel.robot.viewer.gui.addXYZaxis(gname, [1., 0., 0., 1.], .015, 0.1)
            skel.robot.viewer.gui.applyConfiguration(gname, se3ToXYZQUAT(goal))
            skel.robot.viewer.gui.refresh()

        # time.sleep(5)

    n_views = len(cam_calibs)
    DT = 1e-1  # damping factor
    cur_iter = 0
    lm_damping = 1e-3  # damping factor
    impr_stop = 1e-6  # ending condition. relative error changes
    cost = 1000
    model, data = skel.model, skel.data
    x = init_param.pose.copy()
    pin_joint_ids = [skel.target_joint_pino_ids[idx] for idx in skel_kps_idxs]
    while cur_iter < n_max_iter:
        cur_iter += 1
        pin.computeJointJacobians(model, data, x)
        ATA = np.zeros((model.nv, model.nv))
        ATb = np.zeros((model.nv,))
        all_errors = []
        nv = skel.model.nv

        for v_idx in range(n_views):
            calib = cam_calibs[v_idx]
            obs_pose = obs_cam_pose_2d[v_idx]
            obs_pose = obs_pose[obs_kps_idxs, :]
            p_3ds = np.array([data.oMi[j_id].translation for j_id in pin_joint_ids])
            p_3ds_homo = np.concatenate([p_3ds, np.ones((len(p_3ds), 1))], axis=-1)
            p_3ds_cam = (calib.Rt @ p_3ds_homo.T).T
            p_2ds_homo = (calib.K @ p_3ds_cam.T)
            p_2ds = (p_2ds_homo[:2] / p_2ds_homo[-1]).T
            p_2ds_homo = p_2ds_homo.T
            n_kps = len(pin_joint_ids)
            for i in range(n_kps):
                print('\n\n\n kps_idx = {i}')
                org_jac = pin.getJointJacobian(model, data, pin_joint_ids[i], pin.LOCAL_WORLD_ALIGNED)[:3, :]
                kps_2d = p_2ds[i]

                proj_jac = image_jacobian(calib.P, p_2ds_homo[i])

                jac = proj_jac @ org_jac
                err = kps_2d - obs_pose[i, :2]
                mu = lm_damping * max(1e-3, err.dot(err))

                print('kps_3d: ', p_3ds[i])
                print('kps_3d_homo: ', p_2ds_homo[i])
                print('kps_2d: ', kps_2d)
                print('kps_2d_obs: ', obs_pose[i, :2])
                print('err: ', err)
                print('proj_jac: ', proj_jac)
                print('jac: ', jac)
                print('J: ', org_jac)

                ATA += np.dot(jac.T, jac) + mu * np.eye(nv)
                ATb += jac.T @ err
                all_errors.append(err)

        prev_cost = cost
        cost = np.mean(np.concatenate(np.abs(all_errors)))
        impr = abs(cost - prev_cost) / prev_cost
        if impr < impr_stop:
            break

        # dv = solve_qp(ATA, ATb)
        print("ATA: ", ATA)
        print("ATb: ", ATb)
        sol, resids, rank, s = np.linalg.lstsq(ATA, ATb)
        dv = -0.1 * sol
        print('dv: ', dv)
        x = pin.integrate(model, x, dv)

        if vis:
            skel.robot.display(x)
            skel.robot.viewer.gui.refresh()
            time.sleep(0.1)

    init_param.pose = x

    return init_param


def solve_pose_reprojection_auto(skel: HumanModel,
                                 obs_cam_pose_2d: List[np.ndarray],
                                 obs_cam_projs: List[np.ndarray],
                                 obs_pose_3d: np.ndarray,
                                 obs_kps_idxs: List[int],
                                 skel_kps_idxs: List[int],
                                 init_param: PoseShapeParam,
                                 n_max_iter=5) -> PoseShapeParam:
    import subprocess
    subprocess.Popen(['gepetto-gui'])

    n_views = len(obs_cam_projs)
    model, data = skel.model, skel.data
    x = init_param.pose.copy()
    pin_joint_ids = [skel.target_joint_pino_ids[idx] for idx in skel_kps_idxs]

    def _residual_step_2d(_x):
        _joint_locs = skel.forward_kinematics(_x)
        _diffs = []
        for v_idx in range(n_views):
            pose = obs_cam_pose_2d[v_idx]
            pose = pose[obs_kps_idxs, :]
            p_3ds = np.array([data.oMi[j_id].translation for j_id in pin_joint_ids])
            p_3ds = np.concatenate([p_3ds, np.ones((len(p_3ds), 1))], axis=-1)
            p_2ds_homo = (obs_cam_projs[v_idx] @ p_3ds.T)
            p_2ds = (p_2ds_homo[:2] / p_2ds_homo[-1]).T
            diffs = pose[:, :2] - p_2ds
            _diffs.append(diffs)
        _diffs = np.concatenate(_diffs, axis=0)
        return _diffs.flatten()

    results = least_squares(_residual_step_2d, x, verbose=2,
                            max_nfev=1000)
    x = results.x

    init_param.pose = x

    skel.robot.initDisplay(loadModel=True)
    skel.robot.display(init_param.pose)
    skel.robot.viewer.gui.refresh()

    coco_kps_types = get_kps_order(KpsFormat.COCO)
    for tar_idx in obs_kps_idxs:
        goal = pin.SE3.Identity()
        goal.translation = obs_pose_3d[tar_idx, :3]
        gname = f'world/goal_{coco_kps_types[tar_idx].name}'
        skel.robot.viewer.gui.addXYZaxis(gname, [1., 0., 0., 1.], .015, 0.1)
        skel.robot.viewer.gui.applyConfiguration(gname, se3ToXYZQUAT(goal))
        skel.robot.viewer.gui.refresh()

    input()
    return init_param


def solve_pose_from_3d(skel: HumanModel,
                       obs_pose_3d: np.ndarray,
                       obs_kps_idxs: List[int],
                       skel_kps_idxs: List[int],
                       init_param: PoseShapeParam,
                       n_max_iter=5) -> PoseShapeParam:
    import subprocess
    subprocess.Popen(['gepetto-gui'])

    target_pose_3d_shared = obs_pose_3d[obs_kps_idxs, :]

    # def _residual_step_joints_3d(_x):
    #     _joint_locs = skel.forward_kinematics(_x)
    #     _joint_locs = _joint_locs[skel_kps_idxs, :]
    #     _diffs = (_joint_locs - target_pose_3d_shared[:, :3])
    #     _diffs = _diffs * target_pose_3d_shared[:, -1:]
    #     return _diffs.flatten()
    #
    # results = least_squares(_residual_step_joints_3d, init_param.pose.copy(), verbose=2,
    #                         max_nfev=1000)

    DT = 1e-1  # damping factor
    cur_iter = 0
    lm_damping = 1e-3  # damping factor
    impr_stop = 1e-6  # ending condition. relative error changes
    cost = 1000
    model, data = skel.model, skel.data
    x = init_param.pose.copy()
    pin_joint_ids = [skel.target_joint_pino_ids[idx] for idx in skel_kps_idxs]
    while cur_iter < 10000:
        cur_iter += 1
        pin.computeJointJacobians(model, data, x)
        ATA = np.zeros((model.nv, model.nv))
        ATb = np.zeros((model.nv,))
        all_errors = []
        nv = skel.model.nv
        for j_id, tar_loc in zip(pin_joint_ids, target_pose_3d_shared):
            jac = pin.getJointJacobian(model, data, j_id, pin.LOCAL_WORLD_ALIGNED)
            err = tar_loc[:3] - data.oMi[j_id].translation
            err = -err
            jac = jac[:3, :]
            mu = lm_damping * max(1e-3, err.dot(err))
            ATA += np.dot(jac.T, jac) + mu * np.eye(nv)
            ATb += np.dot(-err.T, jac)
            all_errors.append(err)

        prev_cost = cost
        cost = np.mean(np.concatenate(np.abs(all_errors)))
        impr = abs(cost - prev_cost) / prev_cost
        if impr < impr_stop:
            break
        dv = solve_qp(ATA, ATb)
        x = pin.integrate(model, x, -dv * DT)

    init_param.pose = x

    skel.robot.initDisplay(loadModel=True)
    skel.robot.display(init_param.pose)
    skel.robot.viewer.gui.refresh()

    coco_kps_types = get_kps_order(KpsFormat.COCO)
    for tar_idx in obs_kps_idxs:
        goal = pin.SE3.Identity()
        goal.translation = obs_pose_3d[tar_idx, :3]
        gname = f'world/goal_{coco_kps_types[tar_idx].name}'
        skel.robot.viewer.gui.addXYZaxis(gname, [1., 0., 0., 1.], .015, 0.1)
        skel.robot.viewer.gui.applyConfiguration(gname, se3ToXYZQUAT(goal))
        skel.robot.viewer.gui.refresh()

    input()
    return init_param


class PoseSolver:
    def __init__(self,
                 skeleton: Skeleton,
                 init_pose: Optional[PoseShapeParam],
                 cam_poses_2d: List[np.ndarray],
                 cam_projs: List[np.ndarray],
                 cam_calibs: List[Calib],
                 obs_kps_format: KpsFormat):
        self.skel = skeleton
        self.n_joints = self.skel.n_joints
        self.init_pose = init_pose
        self.cam_poses_2d = cam_poses_2d
        self.cam_projs = cam_projs
        self.cam_calibs = cam_calibs
        self.obs_kps_format = obs_kps_format
        self.obs_kps_idx_map = get_kps_index(self.obs_kps_format)

        self.human = HumanModel("/media/F/thesis/libs/pinocchio/khanh_robots")
        self.skel_kps_idxs, self.obs_kps_idxs = get_common_kps_idxs_1(self.human.get_kps_idx_map(),
                                                                      self.obs_kps_idx_map)

    def solve(self):
        n_max_iter = 5

        # subprocess.Popen(['gepetto-gui'])
        # time.sleep(4)
        self.human.robot.initDisplay(loadModel=True)
        self.human.robot.display(self.human.robot.q0)
        self.human.robot.viewer.gui.refresh()

        obs_pose_3d = triangulate_point_groups_from_multiple_views_linear(self.cam_projs,
                                                                          self.cam_poses_2d, 0.01, True)
        init_param = PoseShapeParam(joint_placements=None, pose=self.human.robot.q0.copy())
        # time.sleep(5)
        param_1 = solve_pose_from_2d(self.human, self.cam_poses_2d, self.cam_calibs,
                                     obs_pose_3d, self.obs_kps_idxs, self.skel_kps_idxs, init_param,
                                     vis=True,
                                     n_max_iter=100)

        opt_blen = True
        if opt_blen:
            init_param = param_1
            solve_bone_length_from_2d(self.human, self.cam_poses_2d, self.cam_calibs,
                                      obs_pose_3d, self.obs_kps_idxs, self.skel_kps_idxs, init_param,
                                      vis=True,
                                      n_max_iter=10)
