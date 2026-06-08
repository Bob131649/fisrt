from robot_arms.kinova.kinova_basic import Kinova
# from robot_arms.arms import KinovaFront

# kinova = KinovaFront()
kinova = Kinova()
joint = kinova.get_joint_pose()
trans, quat, rpy = kinova.get_ee_pose()
# kinova.set_gripper_opening(0.05)
# print(joint)
print(trans)

kinova.check_fk()
