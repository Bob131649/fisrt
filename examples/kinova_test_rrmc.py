from robot_arms.kinova.kinova_basic import Kinova
# from robot_arms.arms import KinovaFront
import numpy as np
import pdb
from scipy.spatial.transform import Rotation
# kinova = KinovaFront()

kinova = Kinova()
joint = kinova.get_joint_pose()
trans, quat, rpy = kinova.get_ee_pose(frame='local')
quat_target = quat

rotation_change = Rotation.from_euler('z', 10, degrees=True)  # 绕Z轴旋转10度
current_rotation = Rotation.from_quat(quat)
new_rotation = current_rotation * rotation_change
quat_target = new_rotation.as_quat()

# kinova.set_gripper_opening(0.05)
# print(joint)
trans_target = trans + np.array([0.0, 0.0, -0.10])
q = kinova.get_joint_pose()
# print(q)
# print(trans, quat, rpy)
# pdb.set_trace()
# kinova.ee_move_p_control(trans_target, quat_target)
kinova.ee_move_RRMC(trans_target, quat_target)