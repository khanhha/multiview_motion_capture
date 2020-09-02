from pathlib import Path

import numpy as np
import torch
from fairmotion.data.amass import load as amass_load
from human_body_prior.body_model.body_model import BodyModel

from pose_def import Pose, KpsFormat
from pose_viz import plot_poses_3d

if __name__ == "__main__":
    model_path = '/media/F/datasets/amass/smplx/SMPLX_MALE.npz'
    pose_path = '/media/F/datasets/amass/motion_data/CMU/CMU/01/01_02_poses.npz'

    comp_device = torch.device("cpu")
    num_betas = 10  # number of body parameters
    bm = BodyModel(bm_path=model_path, num_betas=num_betas, model_type='smplx').to(comp_device)

    data = amass_load(pose_path, bm=bm, bm_path=model_path)
    motion_joints = data.positions(local=False)
    print(motion_joints.shape)
    n_poses = len(motion_joints)
    poses = [Pose(pose_type=KpsFormat.SMPLX_22,
                  keypoints=motion_joints[idx, :, :3],
                  keypoints_score=np.zeros((22, 1)),
                  box=None)
             for idx in range(0, n_poses, 4)]

    plot_poses_3d(poses_3d=poses, out_video=Path('./test.mp4'))
