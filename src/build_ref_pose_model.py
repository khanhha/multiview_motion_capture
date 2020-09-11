from pathlib import Path

import numpy as np
import torch
from fairmotion.data.amass import load as amass_load
from human_body_prior.body_model.body_model import BodyModel
import scipy.io as io
import scipy.stats as stats
from pose_def import Pose, KpsFormat
from pose_viz import plot_poses_3d
from tqdm import tqdm


def test_smpl():
    n_pca = 10
    ds_path = Path('/media/F/datasets/caesar')
    evectors = io.loadmat(ds_path / 'evectors.mat')["evectors"]
    evalues = io.loadmat(ds_path / 'evalues.mat')["evalues"]
    mean_shape = io.loadmat(ds_path / 'meanShape.mat')["points"]
    print(evalues.shape, evectors.shape)
    evectors = evectors[:n_pca, :]
    evalues = evalues[:, :n_pca]


def load_amass_base_poses(amass_dir: Path, models, num_joints, device, n_max=1000):
    ppaths = [ppath for ppath in amass_dir.rglob('*.npz') if ppath.stem.endswith('_poses')]
    if n_max < 0:
        n_max = len(ppaths)
    else:
        n_max = min(len(ppaths), n_max)
    ppaths = ppaths[:n_max]
    pose_body_zeros = torch.zeros((1, 3 * (num_joints - 1))).to(device)
    ref_poses = []
    with tqdm(total=n_max) as bar:
        for idx, ppath in enumerate(ppaths):
            bar.update()
            data = np.load(str(ppath), allow_pickle=True)
            gender = str(data["gender"].item())
            gender = 'female' if 'female' in gender else 'male'
            model = models[gender]
            betas = torch.Tensor(data["betas"][:10][np.newaxis]).to(device)
            body = model(pose_body=pose_body_zeros, betas=betas)
            base_position = body.Jtr.detach().cpu().numpy()[0, 0:num_joints]
            ref_poses.append(base_position)
    ref_poses = np.array(ref_poses)
    return ref_poses


def load_smplx_models(smplx_dir: Path, device='cuda'):
    comp_device = torch.device(device)
    num_betas = 10  # number of body parameters
    female = BodyModel(bm_path=str(smplx_dir / f'SMPLX_MALE.npz'), num_betas=num_betas, model_type='smplx').to(
        comp_device)
    male = BodyModel(bm_path=str(smplx_dir / f'SMPLX_FEMALE.npz'), num_betas=num_betas, model_type='smplx').to(
        comp_device)
    return {"male": male, 'female': female}


def build_shape_model():
    device = 'cuda'
    model_dir = Path('/media/F/datasets/amass/smplx')
    models = load_smplx_models(model_dir, device)
    amass_dir = Path('/media/F/datasets/amass/motion_data')
    n_joints = 22
    ref_poses = load_amass_base_poses(amass_dir, models, n_joints, device=device, n_max=10)
    n_poses = len(ref_poses)
    parents = models['male'].kintree_table[0].long()[:n_joints]

    ref_bvecs = n_joints * [np.zeros(0)]
    for i in range(n_joints):
        if i == 0:
            ref_bvecs[i] = np.zeros((n_poses, 3))
        else:
            ref_bvecs[i] = ref_poses[:, i, :3] - ref_poses[:, parents[i], :3]
    ref_blens = [np.zeros((n_poses, 3))] + [np.linalg.norm(bdirs, axis=-1) for bdirs in ref_bvecs[1:]]
    ref_bdirs = [np.zeros((n_poses, 3))] + [bdirs / blens[:, np.newaxis] for bdirs, blens in zip(ref_bvecs[1:], ref_blens[1:])]
    ref_mean_bdirs = [np.mean(bdirs, axis=0) for bdirs in ref_bdirs]

    print(stats.describe(ref_bdirs[1][:, 0], 0))
    print(stats.describe(ref_bdirs[1][:, 1], 0))
    print(stats.describe(ref_bdirs[1][:, 2], 0))


def run_test():
    model_path = '/media/F/datasets/amass/smplx/SMPLX_MALE.npz'
    # pose_path = '/media/F/datasets/amass/motion_data/CMU/CMU/01/01_09_poses.npz'
    # pose_path = '/media/F/datasets/amass/motion_data/CMU/CMU/20_21_rory1/20_12_poses.npz'
    # pose_path = '/media/F/datasets/amass/motion_data/DFaust67/DFaust_67/50007/50007_punching_poses.npz'
    pose_path = '/media/F/datasets/amass/motion_data/DFaust67/DFaust_67/50022/50022_one_leg_jump_poses.npz'

    comp_device = torch.device("cpu")
    num_betas = 10  # number of body parameters
    bm = BodyModel(bm_path=model_path, num_betas=num_betas, model_type='smplx').to(comp_device)

    num_joints = 22
    bdata = np.load(pose_path)
    betas = torch.Tensor(bdata["betas"][:10][np.newaxis]).to("cpu")
    betas_1 = betas + torch.rand(betas.size()) * betas
    print('betas diff: ', betas_1 - betas)
    pose_body_zeros = torch.zeros((1, 3 * (num_joints - 1)))
    body = bm(pose_body=pose_body_zeros, betas=betas_1)
    base_position = body.Jtr.detach().numpy()[0, 0:num_joints]
    poses = [Pose(pose_type=KpsFormat.SMPLX_22,
                  keypoints=base_position[:, :3],
                  keypoints_score=np.zeros((22, 1)),
                  box=None)
             for idx in range(1000)]
    plot_poses_3d(poses_3d=poses, out_video=Path('./test.mp4'))

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


if __name__ == "__main__":
    # test_amass_shapes()
    # test_smpl()
    # run_test()
    build_shape_model()
