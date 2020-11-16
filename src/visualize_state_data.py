import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from os.path import join
from common import Calib
from qpsolvers import solve_qp
from pinocchio import SE3 as SE3
from pinocchio import Inertia as Inertia
from pinocchio import rpy as rpy
import time
from pathlib import Path

root_dir = '/media/F/thesis/data/shelf/robot_states/0'
stt_path = f'{root_dir}/frame_0.txt'
states = np.loadtxt(stt_path)

package_dir = "/media/F/thesis/libs/pinocchio/khanh_robots"
filename = f"{package_dir}/romeo_description/urdf/romeo_khanh.urdf"
robot = RobotWrapper.BuildFromURDF(filename, package_dirs=package_dir,
                                   root_joint=pin.JointModelFreeFlyer())
data = robot.data
model = robot.model

robot.initDisplay(loadModel=True)
robot.display(robot.q0)
robot.viewer.gui.refresh()

target_path = f'{root_dir}/frame_0_3d.txt'
if Path(target_path).exists():
    target_3ds = np.loadtxt(target_path)
    for p_idx, p in enumerate(target_3ds):
        goal = pin.SE3.Identity()
        goal.translation = p
        gname = f'world/goal_{p_idx}'
        robot.viewer.gui.addXYZaxis(gname, [1., 0., 0., 1.], .015, 0.1)
        robot.viewer.gui.applyConfiguration(gname, se3ToXYZQUAT(goal))
        robot.viewer.gui.refresh()

for idx in range(len(states)):
    x = states[idx, :]
    robot.display(x)
    time.sleep(0.5)
    robot.viewer.gui.refresh()
