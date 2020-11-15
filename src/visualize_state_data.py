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


stt_path = '/media/F/thesis/real-time-motion-capture/yolo-tensorrt/samples/robot_states.txt'
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

for idx in range(len(states)):
    x = states[idx, :]
    robot.display(x)
    time.sleep(0.5)
    robot.viewer.gui.refresh()

