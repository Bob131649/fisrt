import numpy as np
# from robot_arms.franka.franka_basic import Franka 
from robot_arms.arms import FrankaLeft as Franka

arm = Franka(robot_ip='192.168.31.11', relative_dynamics_factor=0.05)

# move the arm with curobo ik solver
# ee_trans = np.array([0.40343655, 0.0, 0.5467489 ])
# ee_quat = np.array([ 0.0, 0.0, 0.0, 1.0])
# arm.set_ee_pose(ee_trans, ee_quat, asynchronous=False)

# read ee position and orientation
trans, quat, rpy = arm.get_ee_pose()
print(trans, quat, rpy)
