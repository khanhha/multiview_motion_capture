from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class KpsType(Enum):
    """official name for different type of joints"""
    Nose = 0,
    L_Eye = 1,
    R_Eye = 2,
    L_Ear = 3,
    R_Ear = 4,
    Head_Top = 5,
    Head_Bottom = 6,  # upper_neck
    Head = 7,
    Neck = 8,
    L_Shoulder = 9,
    R_Shoulder = 10,
    L_Elbow = 11,
    R_Elbow = 12,
    L_Wrist = 13,
    R_Wrist = 14,
    L_Hip = 15,
    R_Hip = 16,
    Mid_Hip = 17,
    L_Knee = 18,
    R_Knee = 19,
    L_Ankle = 20,
    R_Ankle = 21,
    Pelvis = 22,
    Spine = 23,
    L_BaseBigToe = 24,
    R_BaseBigToe = 25,
    L_BigToe = 26
    R_BigToe = 27,
    L_SmallToe = 28
    R_SmallToe = 29,
    L_Hand = 30,
    R_Hand = 31
    L_Heel = 32,
    R_Heel = 33,
    Chest = 34,
    LowerNeck = 35,
    UpperNeck = 36,
    LowerBack = 37,
    UpperBack = 38,
    L_Clavicle = 39,
    R_Clavicle = 40,
    Root = 41


class KpsFormat(Enum):
    COCO = 0
    OPENPOSE_25 = 1
    SMPLX_22 = 2
    BASIC_18 = 3


@dataclass
class Pose:
    pose_type: KpsFormat
    keypoints: np.ndarray
    keypoints_score: Optional[np.ndarray]
    box: Optional[np.ndarray]

    def to_kps_array(self):
        return np.concatenate([self.keypoints, self.keypoints_score.reshape((-1, 1))], axis=1)


_COCO = [KpsType.Nose,
         KpsType.L_Eye,
         KpsType.R_Eye,

         KpsType.L_Ear,
         KpsType.R_Ear,

         KpsType.L_Shoulder,
         KpsType.R_Shoulder,

         KpsType.L_Elbow,
         KpsType.R_Elbow,

         KpsType.L_Wrist,
         KpsType.R_Wrist,

         KpsType.L_Hip,
         KpsType.R_Hip,

         KpsType.L_Knee,
         KpsType.R_Knee,

         KpsType.L_Ankle,
         KpsType.R_Ankle
         ]

_COCO_Index = {jtype: jidx for jidx, jtype in enumerate(_COCO)}

_COCO_Bone = [(KpsType.Nose, KpsType.L_Eye), (KpsType.L_Eye, KpsType.L_Ear),
              (KpsType.Nose, KpsType.R_Eye), (KpsType.R_Eye, KpsType.R_Ear),
              (KpsType.L_Shoulder, KpsType.R_Shoulder),
              (KpsType.L_Shoulder, KpsType.L_Elbow), (KpsType.L_Elbow, KpsType.L_Wrist),
              (KpsType.R_Shoulder, KpsType.R_Elbow), (KpsType.R_Elbow, KpsType.R_Wrist),
              (KpsType.L_Shoulder, KpsType.L_Hip), (KpsType.L_Hip, KpsType.L_Knee), (KpsType.L_Knee, KpsType.L_Ankle),
              (KpsType.R_Shoulder, KpsType.R_Hip), (KpsType.R_Hip, KpsType.R_Knee), (KpsType.R_Knee, KpsType.R_Ankle)]

_COCO_Bone_Index = [(_COCO_Index[j0], _COCO_Index[j1]) for (j0, j1) in _COCO_Bone]

########################################################################################################################
_OPENPOSE_25 = [
    KpsType.Nose,
    KpsType.Neck,
    KpsType.R_Shoulder,
    KpsType.R_Elbow,
    KpsType.R_Wrist,
    KpsType.L_Shoulder,
    KpsType.L_Elbow,
    KpsType.L_Wrist,
    KpsType.Mid_Hip,
    KpsType.R_Hip,
    KpsType.R_Knee,
    KpsType.R_Ankle,
    KpsType.L_Hip,
    KpsType.L_Knee,
    KpsType.L_Ankle,
    KpsType.R_Eye,
    KpsType.L_Eye,
    KpsType.R_Ear,
    KpsType.L_Ear,
    KpsType.L_BigToe,
    KpsType.L_SmallToe,
    KpsType.L_Heel,
    KpsType.R_BigToe,
    KpsType.R_SmallToe,
    KpsType.R_Heel
]
_OPENPOSE_25_INDEX = {jtype: jidx for jidx, jtype in enumerate(_OPENPOSE_25)}

_SMPLX_22 = [
    KpsType.Mid_Hip,
    KpsType.L_Hip,
    KpsType.R_Hip,
    KpsType.LowerBack,  # "lowerback",
    KpsType.L_Knee,  # "lknee",
    KpsType.R_Knee,  # "rknee",
    KpsType.UpperBack,  # "upperback",
    KpsType.L_Ankle,  # "lankle",
    KpsType.R_Ankle,  # "rankle",
    KpsType.Chest,  # "chest",
    KpsType.L_BigToe,  # "ltoe",
    KpsType.R_BigToe,  # "rtoe",
    KpsType.LowerNeck,  # "lowerneck",
    KpsType.L_Clavicle,  # "lclavicle",
    KpsType.R_Clavicle,  # "rclavicle",
    KpsType.UpperNeck,  # "upperneck",
    KpsType.L_Shoulder,  # "lshoulder",
    KpsType.R_Shoulder,  # "rshoulder",
    KpsType.L_Elbow,  # "lelbow",
    KpsType.R_Elbow,  # "relbow",
    KpsType.L_Wrist,  # "lwrist",
    KpsType.R_Wrist,  # "rwrist",
]

_SMPLX_22_Index = {jtype: jidx for jidx, jtype in enumerate(_SMPLX_22)}

_SMPLX_22_Bone = [(KpsType.Mid_Hip, KpsType.L_Hip), (KpsType.Mid_Hip, KpsType.R_Hip),
                  (KpsType.Mid_Hip, KpsType.LowerBack), (KpsType.LowerBack, KpsType.UpperBack),
                  (KpsType.L_Hip, KpsType.L_Knee), (KpsType.R_Hip, KpsType.R_Knee),
                  (KpsType.L_Knee, KpsType.L_Ankle), (KpsType.R_Knee, KpsType.R_Ankle),
                  (KpsType.UpperBack, KpsType.Chest),
                  (KpsType.L_Ankle, KpsType.L_BigToe), (KpsType.R_Ankle, KpsType.R_BigToe),
                  (KpsType.Chest, KpsType.LowerNeck), (KpsType.LowerNeck, KpsType.UpperNeck),
                  (KpsType.Chest, KpsType.R_Clavicle), (KpsType.R_Clavicle, KpsType.R_Shoulder),
                  (KpsType.R_Shoulder, KpsType.R_Elbow),
                  (KpsType.R_Elbow, KpsType.R_Wrist),
                  (KpsType.Chest, KpsType.L_Clavicle), (KpsType.L_Clavicle, KpsType.L_Shoulder),
                  (KpsType.L_Shoulder, KpsType.L_Elbow),
                  (KpsType.L_Elbow, KpsType.L_Wrist)]

_SMPLX_22_Bone_Index = [(_SMPLX_22_Index[j0], _SMPLX_22_Index[j1]) for (j0, j1) in _SMPLX_22_Bone]

_BASIC_18_PARENTS = {
    KpsType.Mid_Hip: KpsType.Mid_Hip,
    KpsType.L_Hip: KpsType.Mid_Hip,
    KpsType.L_Knee: KpsType.L_Hip,
    KpsType.L_Ankle: KpsType.L_Knee,
    KpsType.R_Hip: KpsType.Mid_Hip,
    KpsType.R_Knee: KpsType.R_Hip,
    KpsType.R_Ankle: KpsType.R_Knee,
    KpsType.Spine: KpsType.Mid_Hip,
    KpsType.Neck: KpsType.Spine,
    KpsType.L_Shoulder: KpsType.Neck,
    KpsType.L_Elbow: KpsType.L_Shoulder,
    KpsType.L_Wrist: KpsType.L_Elbow,
    KpsType.R_Shoulder: KpsType.Neck,
    KpsType.R_Elbow: KpsType.R_Shoulder,
    KpsType.R_Wrist: KpsType.R_Elbow,
    KpsType.Head_Bottom: KpsType.Neck,
    KpsType.L_Ear: KpsType.Head_Bottom,
    KpsType.R_Ear: KpsType.Head_Bottom
}

_BASIC_18 = [KpsType.Mid_Hip,
             KpsType.L_Hip,
             KpsType.L_Knee,
             KpsType.L_Ankle,
             KpsType.R_Hip,
             KpsType.R_Knee,
             KpsType.R_Ankle,
             KpsType.Spine,
             KpsType.Neck,
             KpsType.L_Shoulder,
             KpsType.L_Elbow,
             KpsType.L_Wrist,
             KpsType.R_Shoulder,
             KpsType.R_Elbow,
             KpsType.R_Wrist,
             KpsType.Head_Bottom,
             KpsType.L_Ear,
             KpsType.R_Ear]

_BASIC_18_Index = {jtype: jidx for jidx, jtype in enumerate(_BASIC_18)}
_BASIC_18_PARENTS_Index = [_BASIC_18_Index[_BASIC_18_PARENTS[jtype]] if _BASIC_18_PARENTS[jtype] != jtype else -1 for
                           jtype in
                           _BASIC_18]


def conversion_openpose_25_to_coco(poses_openpose):
    n_joint = len(_COCO)
    channel = poses_openpose.shape[-1]
    coco = np.zeros((n_joint, channel), dtype=poses_openpose.dtype)
    for j_type in _COCO:
        assert j_type in _OPENPOSE_25_INDEX, 'not supported keypoints'
        opn = poses_openpose[_OPENPOSE_25_INDEX[j_type], :]
        coco[_COCO_Index[j_type], :] = opn
    return coco


def map_to_common_keypoints(pose_0: Pose, pose_1: Pose):
    kps_idxs_0, kps_idxs_1 = get_common_kps_idxs(pose_0.pose_type, pose_1.pose_type)
    return pose_0.to_kps_array()[kps_idxs_0, :], pose_1.to_kps_array()[kps_idxs_1, :]


def get_common_kps_idxs(src_p_type, dst_p_type):
    src_order = get_kps_order(src_p_type)
    dst_idx_map = get_kps_index(dst_p_type)

    src_idxs = []
    dst_idxs = []
    for src_idx, src_jtype in enumerate(src_order):
        if src_jtype in dst_idx_map:
            src_idxs.append(src_idx)
            dst_idxs.append(dst_idx_map[src_jtype])
    return src_idxs, dst_idxs


def get_pose_bones_index(p_type):
    if p_type == KpsFormat.COCO:
        return _COCO_Bone_Index
    elif p_type == KpsFormat.SMPLX_22:
        return _SMPLX_22_Bone_Index
    else:
        raise ValueError('get_pose_bones_index')


def get_kps_order(p_type):
    if p_type == KpsFormat.COCO:
        return _COCO
    elif p_type == KpsFormat.OPENPOSE_25:
        return _OPENPOSE_25
    elif p_type == KpsFormat.SMPLX_22:
        return _SMPLX_22
    elif p_type == KpsFormat.BASIC_18:
        return _BASIC_18
    else:
        raise ValueError('get_kps_index')


def get_kps_index(p_type):
    if p_type == KpsFormat.COCO:
        return _COCO_Index
    elif p_type == KpsFormat.OPENPOSE_25:
        return _OPENPOSE_25_INDEX
    elif p_type == KpsFormat.SMPLX_22:
        return _SMPLX_22_Index
    elif p_type == KpsFormat.BASIC_18:
        return _BASIC_18_Index
    else:
        raise ValueError('get_kps_index')


def get_parent_index(p_type):
    if p_type == KpsFormat.BASIC_18:
        return _BASIC_18_PARENTS_Index
    else:
        raise ValueError(f'get_parent_index: {p_type}')
