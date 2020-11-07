from pose_def import Pose
from typing import Dict, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class Calib:
    K: np.ndarray  # 3x3
    Rt: np.ndarray  # 3x4
    P: np.ndarray  # 3x4
    Kr_inv: np.ndarray  # 3x3
    img_wh_size: Tuple[int, int]

    @property
    def cam_loc(self):
        return -self.Rt[:3, :3].T @ self.Rt[:3, 3]


@dataclass
class FrameData:
    frame_idx: int
    poses: Dict[int, Pose]
    calib: Calib
    view_id: int
