# from franka_robot.tora import TORA
from scipy.spatial.transform import Rotation
from franky import Robot, Gripper, Measure
from franky import Affine, JointWaypointMotion, JointWaypoint, JointMotion, CartesianMotion, ReferenceType, JointState, CartesianWaypointMotion, CartesianWaypoint
import numpy as np
from solvers.curobo_solver import CuRoboIKSolver

# We use franky to control franka robot, please refer to https://github.com/TimSchneider42/franky for more details.

class Franka:
    def __init__(self, robot_ip, relative_dynamics_factor=0.05, control_mode='curobo') -> None:
        self.robot = Robot(robot_ip)
        self.gripper = Gripper(robot_ip)
        self.robot.recover_from_errors()

        self.ik_solver = CuRoboIKSolver('franka', use_cuda_graph=True)
        # self.tora = TORA()

        self.robot.relative_dynamics_factor = relative_dynamics_factor
        # self.robot.velocity_rel = 0.05
        self.robot.acceleration_rel = 1.0
        self.robot.jerk_rel = 1.0

        self.relative_df = relative_dynamics_factor
        self.control_mode = control_mode

        # franka base trans and quat in franka table frame (in the middle of the two franka arms)
        self.pose_translate = np.array([0, 0, 0])
        self.pose_rotate = np.array([0, 0, 0, 1])

        self.start_joint_pose = []
        self.sup_joint_pose = []

        imp_value = 1500
        torque_threshold = 50
        force_threshold = 60
        self.robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])
        self.robot.set_collision_behavior(
            [torque_threshold, torque_threshold, torque_threshold, torque_threshold, torque_threshold, torque_threshold, torque_threshold],  # Torque thresholds (Nm)
            [force_threshold, force_threshold, force_threshold, force_threshold, force_threshold, force_threshold]       # Force thresholds (N)
        )

        self.robot.recover_from_errors()

    def transform_to_local(self, trans_G, quat_G):
        rot_GL = Rotation.from_quat(self.pose_rotate)
        rot_G = Rotation.from_quat(quat_G)
        tran_GL = self.pose_translate

        trans_L = rot_GL.apply(trans_G - tran_GL, inverse=True)
        quat_L = (rot_GL.inv() * rot_G).as_quat()

        return trans_L, quat_L

    def transform_to_global(self, trans_L, quat_L):
        rot_GL = Rotation.from_quat(self.pose_rotate)
        rot_L = Rotation.from_quat(quat_L)
        tran_GL = self.pose_translate

        trans_G = rot_GL.apply(trans_L) + tran_GL
        quat_G = (rot_GL * rot_L).as_quat()

        return trans_G, quat_G

    def set_soft(self):
        imp_value = 150
        self.robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])

    def set_hard(self):
        imp_value = 700
        self.robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])

    def speed_down(self):
        self.robot.relative_dynamics_factor = self.relative_df * 0.5

    def speed_normal(self):
        self.robot.relative_dynamics_factor = self.relative_df

    def set_default_pose(self):
        # # self.robot.relative_dynamics_factor = 0.05
        # motion = CartesianMotion(Affine([-0.04, 0.0, 0.0]), ReferenceType.Relative)
        # self.robot.move(motion)    

        robot_pose = self.robot.current_pose
        ee_trans = robot_pose.end_effector_pose.translation
        ee_quat = robot_pose.end_effector_pose.quaternion
        # motion = CartesianMotion(Affine(ee_trans+np.array([-0.05, 0.0, 0.1]), ee_quat))
        # self.robot.move(motion)
        # if ee_trans[0] > 0.5:
        # self.set_joint_pose(self.sup_joint_pose)

        self.set_joint_pose(self.start_joint_pose)
        # self.robot.relative_dynamics_factor = self.relative_df

    def open_gripper(self, asynchronous=True):
        # if asynchronous:
        #     self.gripper.move_async(0.04, 0.02)
        # else:
        #     self.gripper.open(0.04)
        success = self.gripper.move(0.04, 0.02)

    def close_gripper(self, asynchronous=True):
        # if asynchronous:
        #     self.gripper.move_async(0.0, 0.03)
        # else:
        #     self.gripper.move(0.0, 0.03)
        success = self.gripper.move(0.0, 0.01)
        # self.gripper.grasp(0.0, 0.05, 20, epsilon_outer=1.0)

    def set_gripper_opening(self, width, asynchronous=True):
        current_width = self.gripper.width
        # if asynchronous:
        #     self.gripper.move_async(width, 0.02)
        # else:
        if width > 0.01:
            width = 0.04
        else:
            width = 0.0

        if abs(current_width - width) > 0.01:
            self.gripper.move(width, 0.03)
            # self.gripper.move_async(width, 0.03)
        # success = self.gripper.move(0.0, 0.02)

    def set_ee_pose(self, translation, quaternion, asynchronous=True, frame='global'):
        if self.control_mode == 'curobo':
            result = self.set_ee_pose_curobo(translation, quaternion, asynchronous, frame)
        else:
            result = self.set_ee_pose_armlib(translation, quaternion, asynchronous, frame)
        return result

    def set_ee_pose_armlib(self, translation, quaternion, asynchronous=True, frame='global'):
        if frame == 'global':
            trans_local, quat_local = self.transform_to_local(translation, quaternion)
        else:
            trans_local, quat_local = translation, quaternion
        motion = CartesianMotion(Affine(trans_local, quat_local))

        # can use try -- except to catch libfranka error, recover error and reexecute
        self.robot.move(motion, asynchronous=asynchronous)
        return True

    def set_ee_pose_curobo(self, translation, quaternion, asynchronous=True, frame='global'):
        if frame == 'global':
            trans_local, quat_local = self.transform_to_local(translation, quaternion)
        else:
            trans_local, quat_local = translation, quaternion

        # solving ik using cuRobo
        current_q = self.robot.state.q
        q_target = self.ik_solver.get_target_joint(current_q, trans_local, quat_local)
        self.set_joint_pose(q_target, asynchronous=asynchronous)
        return True

    def set_joint_pose(self, joint_pose, asynchronous=False):
        assert len(joint_pose) == 7
        m1 = JointMotion(joint_pose)
        self.robot.move(m1, asynchronous=asynchronous)

    def set_joint_trajectories(self, joint_trajectory, velocity_trajectory, asynchronous=False):
        waypoints = []
        if len(joint_trajectory) == 1:
            joint_trajectory = np.array([joint_trajectory])

        # print(joint_trajectory)
        step = 3
        for i in range(0, len(joint_trajectory)-1, step):
        # for js, jv in zip(joint_trajectory, velocity_trajectory):
            js = joint_trajectory[i]
            jv = joint_trajectory[i+step] - joint_trajectory[i]
            # jv = velocity_trajectory[i]
            # wpoint = JointWaypoint(js)
            print(js, jv)
            wpoint = JointWaypoint(JointState(position=js, velocity=jv*0.5))
            if i %step == 0:
                waypoints.append(wpoint)

        wpoint = JointWaypoint(JointState(position=joint_trajectory[-1]))
        waypoints.append(wpoint)

        motion = JointWaypointMotion(waypoints)
        # m1 = JointWaypointMotion([JointWaypoint(joint_pose)])
        self.robot.move(motion, asynchronous=False)

    def get_ee_pose(self, frame='global'):
        robot_pose = self.robot.current_pose
        trans_local = robot_pose.end_effector_pose.translation
        quat_local = robot_pose.end_effector_pose.quaternion
        # ee_rpy = Rotation.from_quat(ee_quat).as_euler('xyz')

        # shifted_ee_trans = ee_trans + self.pose_translate
        if frame == 'global':
            trans_global, quat_global = self.transform_to_global(trans_local, quat_local)
            ee_rpy = Rotation.from_quat(quat_global).as_euler('xyz')
            return trans_global, quat_global, ee_rpy
        else:
            ee_rpy = Rotation.from_quat(quat_local).as_euler('xyz')
            return trans_local, quat_local, ee_rpy

    def get_joint_pose(self):
        state = self.robot.state
        joint_pose = state.q
        return joint_pose

    def get_elbow_pose(self):
        state = self.robot.state
        elbow_pose = state.elbow
        return elbow_pose
    
    def get_joint_vel(self):
        return self.robot.current_joint_velocities
    
    def get_gripper_width(self):
        return self.gripper.width
    
    def get_ee_force(self):
        fx, fy, fz = self.robot.state.O_F_ext_hat_K[0:3]
        normal_force = (fx ** 2 + fy ** 2 + fz ** 2) ** 0.5
        # print(self.robot.state.O_F_ext_hat_K, normal_force)
        # normal_force = (fx ** 2 + fy ** 2 + fz ** 2) ** 0.5
        return fx, fy, fz, normal_force
    
    def check_fk(self):
        current_q = self.get_joint_pose()
        fk_ee_trans, fk_ee_quat = self.ik_solver.forward_kinematics(current_q)
        lib_ee_trans, lib_ee_quat, lib_ee_rpy = self.get_ee_pose(frame='local')

        print('FK:', fk_ee_trans, fk_ee_quat)
        print('lib:', lib_ee_trans, lib_ee_quat)

    def stop_motion(self):
        pass