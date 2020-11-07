import numpy as np
import fire
from common import FrameData, Calib
from pose_def import Pose, KpsType, KpsFormat
from pathlib import Path
import pickle
from tqdm import tqdm


def load_calib(cpath):
    calib = pickle.load(open(cpath, 'rb'))
    K = calib["K"]
    Rt = np.concatenate([calib["R"].reshape((3, 3)), calib["t"].reshape((3, 1))], axis=-1)
    KR = K @ Rt[:3, :3]
    c = Calib(K, Rt, K @ Rt, np.linalg.inv(KR), (1920, 1080))
    return c


def load_poses(np_path):
    kpts = np.load(str(np_path))
    poses = [Pose(pose_type=KpsFormat.COCO, keypoints=kp[:, :2], keypoints_score=kp[:, -1], box=np.zeros((4,)))
             for kp in kpts]
    return poses


def run_main(rdir, cdir, odir):
    rdir = Path(rdir)
    odir = Path(odir)
    cdir = Path(cdir)

    kps_paths = sorted([kps_path for kps_path in rdir.glob('*.npy')])
    calib_paths = sorted([cpath for cpath in cdir.glob('*.pkl')])
    calibs = [load_calib(cpath) for cpath in calib_paths]
    view_poses = [load_poses(ppath) for ppath in kps_paths]

    n_frms = min([len(poses) for poses in view_poses])
    for frm_idx in tqdm(range(n_frms)):
        v_poses = [poses[frm_idx] for poses in view_poses]
        frms = [FrameData(frm_idx, {0: pose}, calibs[v_idx], v_idx) for v_idx, pose in enumerate(v_poses)]
        with open(odir / f'{frm_idx}.pkl', 'wb') as file:
            pickle.dump(frms, file)


if __name__ == "__main__":
    fire.Fire(run_main)
