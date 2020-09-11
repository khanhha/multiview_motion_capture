import yaml
import numpy as np
import os
import sys
from os.path import join as pjoin
from pathlib import Path


class Skeleton:
    def __init__(self):
        filename = Path(__file__).parent / 'skeleton_CMU.yml'
        with open(filename, "r") as file:
            skel = yaml.load(file, Loader=yaml.Loader)
        self.bvh_name = os.path.join(os.path.dirname(filename), skel['BVH'])
        self.offset = np.array(skel['offsets'])
        self.n_joints = len(self.offset)
        self.topology = np.array(skel['parents'])
        self.chosen_joints = np.array(skel['chosen_joints'])
        self.chosen_parents = np.array(skel['chosen_parents'])
        self.hips, self.sdrs = skel['hips'], skel['shoulders']
        self.head = skel['head']

