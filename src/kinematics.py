import torch
import numpy as np
import math
import copy
import os
import sys
from scipy.spatial.transform import Rotation
from typing import List


class ForwardKinematics:
    def __init__(self, skel):
        self.offset = skel.offset
        self.parents = skel.topology
        self.chosen_joints = skel.chosen_joints
        self.n_joints = len(self.offset)

    def forward(self, rotations: List[Rotation]):
        l_trans = np.array([np.eye(4) for _ in range(self.n_joints)])
        for j_i in range(self.n_joints):
            l_trans[j_i, :3, :3] = rotations[j_i].as_matrix()
            if j_i != 0:
                l_trans[j_i, :3, 3] = self.offset[j_i]

        g_trans = l_trans.copy()
        for j_i in range(self.n_joints):
            g_trans[j_i, :, :] = g_trans[self.parents[j_i], :, :] @ l_trans[j_i, :, :]

        g_pos = g_trans[:, :3, 3]
        g_pos = g_pos[:, :3] / g_pos[:, 2:3]
        return g_pos
