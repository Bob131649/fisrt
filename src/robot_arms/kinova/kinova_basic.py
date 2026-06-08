#!/usr/bin/env python3

"""
kinova_basic.py
---------------
Kinova机械臂高层控制接口，封装了常用的运动、状态查询等功能。
参考franka_basic.py的结构设计，提供简洁易用的API。
"""

import sys
import os
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.ActuatorCyclicClientRpc import ActuatorCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, ActuatorConfig_pb2, ControlConfig_pb2
from kortex_api.RouterClient import RouterClientSendOptions
from solvers.curobo_solver import CuRoboIKSolver
from robot_arms.kinova.kinova_device_connection import DeviceConnection
from kortex_api.Exceptions.KServerException import KServerException
from controllers.simple_controller import resolved_rate_motion_control, so3_log, ee_p_control

# 尝试导入PyLibRM夹爪库
try:
    from pylibrm import RMAxis
    GRIPPER_AVAILABLE = True
except ImportError:
    print("[Kinova] 警告: PyLibRM库未安装，夹爪功能将不可用")
    GRIPPER_AVAILABLE = False

# 默认动作等待超时时间（秒）
TIMEOUT_DURATION_EE = 0.5
TIMEOUT_DURATION_JOINT = 20

def check_for_end_or_abort(e):
    """
    动作完成/中止事件回调。
    用于阻塞等待机械臂动作结束。
    
    参数：
        e (threading.Event): 事件对象，动作完成/中止时被设置。
    返回：
        回调函数，供OnNotificationActionTopic订阅使用。
    """
    def check(notification, e = e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

class Kinova:
    """
    Kinova类：Kinova机械臂高层控制接口，封装了常用的运动、状态查询等功能。
    """
    
    def __init__(self, robot_ip="192.168.31.13", port=10000, relative_dynamics_factor=0.05, 
                 gripper_port="/dev/ttyUSB0", gripper_baudrate=115200, gripper_slave_id=1, control_mode='lib'):
        """
        初始化Kinova机械臂对象。
        
        参数：
            robot_ip (str): 机器人IP地址，默认"192.168.1.10"
            port (int): 通信端口，默认10000
            relative_dynamics_factor (float): 运动学相对动态因子，影响速度/加速度等
            gripper_port (str): 夹爪串口路径，默认"/dev/ttyUSB0"
            gripper_baudrate (int): 夹爪波特率，默认115200
            gripper_slave_id (int): 夹爪从设备ID，默认1
        """
        # # 导入utilities模块
        # sys.path.insert(0, os.path.join(os.path.dirname(__file__), "./Kinova-kortex2_Gen3_G3L/api_python/examples"))
        # import utilities

        # 设置连接参数
        self.robot_ip = robot_ip
        self.port = port
        self.relative_dynamics_factor = relative_dynamics_factor
        self.control_mode = control_mode

        self.router = DeviceConnection.createTcpConnection(robot_ip)
        self.router = self.router.__enter__()  # 手动进入上下文管理器
        
        # 创建服务客户端
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        self.actuator_config = ActuatorConfigClient(self.router)
        self.control_config = ControlConfigClient(self.router)
        
        # 阻抗控制相关变量
        self.cyclic_running = False
        self.kill_the_thread = False
        self.cyclic_thread = None
        self.base_command = BaseCyclic_pb2.Command()
        self.base_feedback = BaseCyclic_pb2.Feedback()
        
        # 获取执行器数量
        self.actuator_count = self.base.GetActuatorCount().count
        
        # 初始化命令和反馈结构
        for i in range(self.actuator_count):
            self.base_command.actuators.add()
            self.base_feedback.actuators.add()
        
        # 设置发送选项
        self.sendOption = RouterClientSendOptions()
        # self.sendOption.andForget = False
        # self.sendOption.delay_ms = 0
        self.sendOption.timeout_ms = 1000000
        
        # 设置默认超时时间
        # self.timeout_duration = TIMEOUT_DURATION
        
        self.pose_translate = np.array([0, 0, 0])
        self.pose_rotate = np.array([0, 0, 0, 1])
        
        # 初始化状态
        self.start_joint_pose = [0, 0, 0, 0, 0, 0, 0]  # 预设起始关节角
        self.home_joint_pose = [0, 0, 0, 0, 0, 0, 0]   # 预设Home关节角
        self.last_torque = None
        self.last_pos = None
        
        # 初始化夹爪
        self.gripper = None
        self.gripper_port = gripper_port
        self.gripper_baudrate = gripper_baudrate
        self.gripper_slave_id = gripper_slave_id
        
        if GRIPPER_AVAILABLE:
            try:
                self.gripper = RMAxis.Axis_V6.create_modbus_rtu(
                    gripper_port, gripper_baudrate, gripper_slave_id
                )
                # 设置夹爪通信参数
                self.gripper.set_retries(5)
                self.gripper.set_timeout(100)  # 100ms超时
                # 开启夹爪伺服
                self.gripper.set_servo_on_off(True)
                # 设置默认运动参数
                self.gripper.config_motion(50.0, 100.0)  # 速度50mm/s, 加速度100mm/s²
                print(f"[Kinova] 夹爪连接成功: {gripper_port}")
            except Exception as e:
                print(f"[Kinova] 夹爪连接失败: {e}")
                self.gripper = None
        self.ik_solver = CuRoboIKSolver('kinova', use_cuda_graph=True)
        print("[Kinova] 机械臂连接成功")


    def __del__(self):
        """析构函数，确保连接正确关闭。"""
        # 停止阻抗控制线程
        if hasattr(self, 'cyclic_running') and self.cyclic_running:
            self.stop_impedance_control()
        
        # 关闭夹爪连接
        if hasattr(self, 'gripper') and self.gripper is not None:
            try:
                self.gripper.close()
            except:
                pass
    
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

    def close(self):
        """关闭连接。"""
        # 停止阻抗控制线程
        if hasattr(self, 'cyclic_running') and self.cyclic_running:
            self.stop_impedance_control()
        
        # 关闭夹爪连接
        if hasattr(self, 'gripper') and self.gripper is not None:
            try:
                self.gripper.close()
            except:
                pass
        
        # 关闭机械臂连接
        if hasattr(self, 'router') and hasattr(self.router, '__exit__'):
            try:
                self.router.__exit__(None, None, None)
            except:
                pass
        
        print("[Kinova] 连接已关闭")
    
   
    def speed_down(self):
        """降低运动速度。"""
        self.relative_dynamics_factor *= 0.5
        print(f"[Kinova] 速度降低到 {self.relative_dynamics_factor}")
    
    def speed_normal(self):
        """恢复默认运动速度。"""
        self.relative_dynamics_factor = 0.05
        print(f"[Kinova] 速度恢复到 {self.relative_dynamics_factor}")
    
    def set_speed_factor(self, factor):
        """
        设置速度因子。
        
        参数：
            factor (float): 速度因子，0.1-1.0之间
        """
        if 0.1 <= factor <= 1.0:
            self.relative_dynamics_factor = factor
            print(f"[Kinova] 速度因子设置为 {factor}")
            return True
        else:
            print(f"[Kinova] 速度因子必须在0.1-1.0之间，当前值: {factor}")
            return False
    
    def set_default_pose(self):
        """
        让机器人回到预设的默认姿态。
        先到Home位，再到起始位。
        """
        print("[Kinova] 移动到默认姿态")
        # self.set_joint_pose(self.home_joint_pose)
        self.set_joint_pose(self.start_joint_pose, asynchronous=False)
    
    def open_gripper(self, asynchronous=True):
        """
        张开夹爪。
        
        参数：
            asynchronous (bool): 是否异步执行
        返回：
            bool: True-动作完成，False-超时/中止
        """
        if self.gripper is None:
            print("[Kinova] 夹爪未连接")
            return False
        
        try:
            # 使用push方法张开夹爪，推压距离20mm，力限制15%
            self.gripper.move_absolute(0.04, 50.0, 100.0, 100.0, 0.1)
            
            if not asynchronous:
                # 等待动作完成
                import time
                while not self.gripper.is_moving():
                    time.sleep(0.1)
                # print("[Kinova] 夹爪张开完成")
                return True
            else:
                # print("[Kinova] 夹爪张开指令已发送")
                return True
                
        except Exception as e:
            print(f"[Kinova] 夹爪张开失败: {e}")
            return False
    
    def close_gripper(self, asynchronous=True):
        """
        闭合夹爪。
        
        参数：
            asynchronous (bool): 是否异步执行
        返回：
            bool: True-动作完成，False-超时/中止
        """
        if self.gripper is None:
            print("[Kinova] 夹爪未连接")
            return False
        
        try:
            # 使用push方法闭合夹爪，推压距离20mm，力限制15%
            self.gripper.move_absolute(40.0, 50.0, 100.0, 100.0, 0.1)
            
            if not asynchronous:
                # 等待动作完成
                import time
                while not self.gripper.is_moving():
                    time.sleep(0.1)
                # print("[Kinova] 夹爪闭合完成")
                return True
            else:
                # print("[Kinova] 夹爪闭合指令已发送")
                return True
                
        except Exception as e:
            print(f"[Kinova] 夹爪闭合失败: {e}")
            return False
    
    def set_gripper_opening(self, width, asynchronous=True):
        """
        设置夹爪开口宽度。
        
        参数：
            width (float): 目标开口宽度（米）
            asynchronous (bool): 是否异步执行
        返回：
            bool: True-动作完成，False-超时/中止
        """
        current_width = (40-self.gripper.position())/1000
        print('set gripper openning:', width, "current openning:", current_width)
        if self.gripper is None:
            print("[Kinova] 夹爪未连接")
            return False
        
        try:
            if width > 0.01:
                width = 0.04
            else:
                width = 0.002

            if abs(current_width - width) > 0.01:
                # 使用绝对运动到指定位置
                self.gripper.move_absolute(40-width*1000, 50.0, 100.0, 100.0, 0.1)
            
            if not asynchronous:
                # 等待动作完成
                import time
                while not self.gripper.is_moving():
                    time.sleep(0.1)
                return True
            else:
                return True
                
        except Exception as e:
            print(f"[Kinova] 夹爪宽度设置失败: {e}")
            return False
    
    def set_ee_pose(self, translation, quaternion, asynchronous=True, frame='global'):
        # shifted_translation = translation
        if frame == 'global':
            trans_local, quat_local = self.transform_to_local(translation, quaternion)
        else:
            trans_local, quat_local = translation, quaternion

        print('[kinova] target set', trans_local, quat_local)
        if self.control_mode == 'curobo':
            result = self.set_ee_pose_curobo(trans_local, quat_local, asynchronous)
        else:
            # self.ee_move(trans_local, quat_local, asynchronous)
            # if not asynchronous:
            #     self.ee_move_ik(trans_local, quat_local)
            # else:
            result = self.ee_move_p_control(trans_local, quat_local, asynchronous)
                # self.ee_move_RRMC(trans_local, quat_local)

        trans_lib, quat_lib, _ = self.get_ee_pose(frame='local')
        print('[kinova] target reached', trans_lib, quat_lib)
        return result

    def corrected_target(self, trans_target, quat_target):
        # Current FK and measured EE
        lib_joint = self.get_joint_pose()
        fk_ee_trans, fk_ee_quat = self.ik_solver.forward_kinematics(lib_joint)
        lib_ee_trans, lib_ee_quat, _ = self.get_ee_pose(frame='local')
        print("[Kinova] Lib:", lib_ee_trans, lib_ee_quat)
        print("[Kinova] FK:", fk_ee_trans, fk_ee_quat)

        # Compute translation error (measured - FK)
        trans_diff = lib_ee_trans - fk_ee_trans.cpu().numpy()
        print("[Kinova] translation shift:", trans_diff)

        # Apply translation correction to the target
        trans_cmd = trans_target + trans_diff[0]

        # orrect rotation by comparing quaternions:
        rot_fk = Rotation.from_quat(fk_ee_quat.cpu().numpy())
        rot_ms = Rotation.from_quat(lib_ee_quat)
        rot_err = rot_ms * rot_fk.inv()
        quat_cmd = (rot_err.inv() * Rotation.from_quat(quat_target)).as_quat()
        # quat_cmd = quat_target 
        return trans_cmd, quat_cmd

    def set_ee_pose_curobo(self, trans_local, quat_local, asynchronous=True):
        """
        控制机器人运动到指定末端位姿。
        
        参数：
            translation (list/np.ndarray): 3维位置
            quaternion (list/np.ndarray): 4维四元数
            asynchronous (bool): 是否异步执行
            frame (str): 'global'或'local'，指定输入位姿的参考系
        """
        # print('[kinova] target set', trans_local, quat_local)
        # solving ik using cuRobo
        current_q = self.get_joint_pose()
        # trans_corrected, quat_corrected = self.corrected_target(trans_local, quat_local)
        # q_target = self.ik_solver.get_target_joint(current_q, trans_corrected, quat_corrected)
        q_target = self.ik_solver.get_target_joint(current_q, trans_local, quat_local)

        self.set_joint_pose(q_target, asynchronous=asynchronous)
        # trans_lib, quat_lib, _ = self.get_ee_pose(frame='local')
        # print('[kinova] target reached', trans_lib, quat_lib)
        return True

    def set_joint_pose(self, joint_pose, asynchronous=False):
        """
        控制机器人运动到指定关节角。
        
        参数：
            joint_pose (list/np.ndarray): 7维关节角 -π到π
            asynchronous (bool): 是否异步执行
        """
        
        if len(joint_pose) != 7:
            raise ValueError("joint_pose 必须为长度为7的列表或数组")
        
        # 将关节角从 -π到π 转换为 0-360 度
        joint_pose_deg = []
        for angle in joint_pose:
            # 将弧度转换为度数
            angle_deg = np.rad2deg(angle)
            # 将度数范围从 -180到180 转换为 0-360
            angle_deg_360 = (angle_deg + 360) % 360
            joint_pose_deg.append(angle_deg_360)
        
        action = Base_pb2.Action()
        action.name = "Set Joint Pose"
        action.application_data = ""
        
        for joint_id, angle in enumerate(joint_pose_deg):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = angle
        
        if not asynchronous:
            e = threading.Event()
            notification_handle = self.base.OnNotificationActionTopic(
                check_for_end_or_abort(e),
                Base_pb2.NotificationOptions()
            )
            
            # print(f"[Kinova] 移动到关节角: {joint_pose}")
            self.base.ExecuteAction(action)
            
            finished = e.wait(TIMEOUT_DURATION_JOINT)
            self.base.Unsubscribe(notification_handle)            
            current_q = self.get_joint_pose()
            print('[Kinova] q target:', joint_pose)
            print('[Kinova] q reached:', current_q)

            return finished
        else:
            # print(f"[Kinova] 异步移动到关节角: {joint_pose}")
            self.base.ExecuteAction(action)
            return True
    

    def set_joint_velocity(self, joint_velocities):
        """
        发送关节速度控制命令。

        参数：
            joint_velocities (list/tuple/np.ndarray): 7个关节的速度值（单位: rad/s 或 deg/s, 
                                                     取决于 Kinova API 的配置）。
        """
        import numbers

        # --- 输入检查 ---
        if not isinstance(joint_velocities, (list, tuple, np.ndarray)):
            raise TypeError("joint_velocities must be a list, tuple, or numpy array")
        if len(joint_velocities) != 7:
            raise ValueError("joint_velocities must contain exactly 7 values")
        if not all(isinstance(v, numbers.Number) for v in joint_velocities):
            raise ValueError("all elements in joint_velocities must be numeric")

        # --- 构造 JointSpeeds 消息 ---
        joint_velocities = np.rad2deg(joint_velocities)
        print('joint_velocities', joint_velocities)
        joint_speeds = Base_pb2.JointSpeeds()
        for i, v in enumerate(joint_velocities):
            joint_speed = joint_speeds.joint_speeds.add()
            joint_speed.joint_identifier = i
            joint_speed.value = float(v)  # 转 float
            # joint_speed.duration = 0      # 0 表示持续直到下一条命令

        # --- 发送命令 ---
        self.base.SendJointSpeedsCommand(joint_speeds)


    def set_joint_trajectories(self, joint_trajectory, velocity_trajectory=None, asynchronous=False):
        """
        执行关节空间轨迹。
        
        参数：
            joint_trajectory (np.ndarray): 关节角轨迹序列
            velocity_trajectory (np.ndarray): 关节速度轨迹序列（可选）
            asynchronous (bool): 是否异步执行
        """
        from kortex_api.autogen.messages import Base_pb2
        
        waypoints = Base_pb2.WaypointList()
        waypoints.duration = 0.0
        waypoints.use_optimal_blending = False
        
        for idx, joint_pose in enumerate(joint_trajectory):
            waypoint = waypoints.waypoints.add()
            waypoint.name = f"waypoint_{idx}"
            
            angular_wp = waypoint.angular_waypoint
            angular_wp.angles.extend(joint_pose)
            angular_wp.duration = 5.0  # 每个路点的持续时间
        
        # 验证轨迹
        result = self.base.ValidateWaypointList(waypoints)
        if len(result.trajectory_error_report.trajectory_error_elements) != 0:
            print("[Kinova] 关节轨迹验证失败")
            return False
        
        if not asynchronous:
            e = threading.Event()
            notification_handle = self.base.OnNotificationActionTopic(
                check_for_end_or_abort(e),
                Base_pb2.NotificationOptions()
            )
            
            print(f"[Kinova] 执行关节轨迹，路点数: {len(joint_trajectory)}")
            self.base.ExecuteWaypointTrajectory(waypoints)
            
            finished = e.wait(TIMEOUT_DURATION_JOINT * len(joint_trajectory))
            self.base.Unsubscribe(notification_handle)
            
            if finished:
                print("[Kinova] 关节轨迹执行完成")
            else:
                print("[Kinova] 关节轨迹执行超时或中止")
            
            return finished
        else:
            print(f"[Kinova] 异步执行关节轨迹，路点数: {len(joint_trajectory)}")
            self.base.ExecuteWaypointTrajectory(waypoints)
            return True
    
    
    def get_ee_pose(self, frame='global'):
        """
        获取当前末端位姿（位置、四元数、欧拉角）。
        frame (str): 'global'或'local'，指定输入位姿的参考系
        
        返回：
            shifted_ee_trans (np.ndarray): 末端位置
            ee_quat (np.ndarray): 末端四元数
            ee_rpy (np.ndarray): 末端欧拉角
        """
        feedback = self.base_cyclic.RefreshFeedback()
        
        # 获取末端位置
        ee_trans = np.array([
            feedback.base.tool_pose_x,
            feedback.base.tool_pose_y,
            feedback.base.tool_pose_z
        ])
        
        # 获取末端欧拉角并转换为四元数
        ee_euler_deg = np.array([
            feedback.base.tool_pose_theta_x,
            feedback.base.tool_pose_theta_y,
            feedback.base.tool_pose_theta_z
        ])
        ee_euler_rad = np.radians(ee_euler_deg)
        
        rotation = Rotation.from_euler('xyz', ee_euler_rad, degrees=False)
        ee_quat = rotation.as_quat()  # [x, y, z, w] 格式
        
        # shifted_ee_trans = ee_trans + self.pose_translate
        if frame == 'local':
            ee_rpy = Rotation.from_quat(ee_quat).as_euler('xyz')
            return ee_trans, ee_quat, ee_rpy

        elif frame == 'global':
            trans_global, quat_global = self.transform_to_global(ee_trans, ee_quat)
            ee_rpy = Rotation.from_quat(quat_global).as_euler('xyz')

            return trans_global, quat_global, ee_rpy
    
    def get_joint_pose(self):
        """
        获取当前关节角度。
        
        返回：
            joint_pose (np.ndarray): 7维关节角 -π到π
        """
        feedback = self.base_cyclic.RefreshFeedback()
        joint_pose = np.array([actuator.position for actuator in feedback.actuators])
        
        # 确保返回7维
        while len(joint_pose) < 7:
            joint_pose = np.append(joint_pose, 0.0)
        joint_pose = joint_pose[:7]
        
        # 将 0-360 度转换为 -π 到 π 弧度
        joint_pose_rad = []
        for angle in joint_pose:
            # 将度数转换为弧度
            angle_rad = np.deg2rad(angle)
            # 将弧度范围从 0-2π 转换为 -π 到 π
            angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi
            joint_pose_rad.append(angle_rad)
            
        return np.array(joint_pose_rad)
    

    
    def get_joint_vel(self):
        """
        获取当前关节速度。
        
        返回：
            np.ndarray: 7维关节速度
        """
        feedback = self.base_cyclic.RefreshFeedback()
        joint_vel = np.array([actuator.velocity for actuator in feedback.actuators])
        
        # 确保返回7维
        while len(joint_vel) < 7:
            joint_vel = np.append(joint_vel, 0.0)
        
        return joint_vel[:7]
    
    def get_gripper_width(self):
        """
        获取当前夹爪开口宽度（毫米）。
        
        返回：
            float: 夹爪宽度（毫米）
        """
        if self.gripper is None:
            print("[Kinova] 夹爪未连接")
            return 0.0
        
        try:
            width = (40-self.gripper.position())/1000
            return width
        except Exception as e:
            print(f"[Kinova] 夹爪宽度获取失败: {e}")
            return 0.0
    
    def get_gripper_force(self):
        """
        获取夹爪力传感器数值（牛顿）。
        
        返回：
            float: 夹爪力传感器读数（N）
        """
        if self.gripper is None:
            print("[Kinova] 夹爪未连接")
            return 0.0
        
        try:
            force = self.gripper.force_sensor()
            return force
        except Exception as e:
            print(f"[Kinova] 夹爪力传感器读取失败: {e}")
            return 0.0
    
    
    def get_joint_torque(self):
        """
        获取当前关节力矩。
        
        返回：
            np.ndarray: 7维关节力矩
        """
        feedback = self.base_cyclic.RefreshFeedback()
        joint_torque = np.array([actuator.torque for actuator in feedback.actuators])
        
        # 确保返回7维
        while len(joint_torque) < 7:
            joint_torque = np.append(joint_torque, 0.0)
        
        return joint_torque[:7]
    
    def ee_move_p_control(self, trans_target, quat_target, asynchronous):
        trans_current, quat_current, _ = self.get_ee_pose(frame='local')
        p_des = np.array([trans_target[0], trans_target[1], trans_target[2]])
        R_des = Rotation.from_quat(quat_target).as_matrix()
        trans_threshold = 1e-2# 1 mm
        rot_threshold = 1.0   # degree
        torque_threshold = 2.0  # 力矩变化阈值

        result = True
        if asynchronous:
            v_cmd, w_cmd = ee_p_control(trans_current, quat_current, trans_target, quat_target)
            w_cmd_degree = np.rad2deg(w_cmd)
            command = Base_pb2.TwistCommand()

            command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            # command.duration = 0

            twist = command.twist
            twist.linear_x = v_cmd[0]
            twist.linear_y = v_cmd[1]
            twist.linear_z = v_cmd[2]
            twist.angular_x = w_cmd_degree[0]
            twist.angular_y = w_cmd_degree[1]
            twist.angular_z = w_cmd_degree[2]

            # print ("Sending the twist command for 5 seconds...")
            self.base.SendTwistCommand(command)
            return True
        else:
            while True:
                trans_current, quat_current, _ = self.get_ee_pose(frame='local')
                curr_torque = self.get_joint_torque()
                if not self.check_torque_change(torque_threshold):
                    print('力矩变化超过阈值，放弃控制并退回到上一个位置')
                    result = False
                    break
                p_cur = np.array(trans_current)
                R_cur = Rotation.from_quat(quat_current).as_matrix()

                # 误差计算
                trans_error = np.linalg.norm(p_des - p_cur)  # 欧氏距离
                rot_error = so3_log(R_des @ R_cur.T)
                rot_error = np.linalg.norm(rot_error)
                rot_error_deg = np.degrees(rot_error)

                # print('trans_error', trans_error)
                # print('rot_error', rot_error_deg)

                # 判断是否收敛
                if trans_error < trans_threshold and rot_error_deg < rot_threshold:
                    self.last_torque = curr_torque
                    self.last_pos = (trans_current, quat_current, _)
                    print('ok')
                    self.stop_motion()
                    break
                
                v_cmd, w_cmd = ee_p_control(trans_current, quat_current, trans_target, quat_target)
                w_cmd = np.rad2deg(w_cmd)
                command = Base_pb2.TwistCommand()

                command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
                # command.duration = 0
                self.last_pos = (trans_current, quat_current, _)
                self.last_torque = curr_torque

                twist = command.twist
                twist.linear_x = v_cmd[0]
                twist.linear_y = v_cmd[1]
                twist.linear_z = v_cmd[2]
                twist.angular_x = w_cmd[0]
                twist.angular_y = w_cmd[1]
                twist.angular_z = w_cmd[2]

                # print ("Sending the twist command for 5 seconds...")
                self.base.SendTwistCommand(command)
                time.sleep(0.05)
        return result

    def ee_move_ik(self, trans, quat):
        """
        Compute IK using kinova lib and move to the target pose.
        """        
        computed_joint_angles = self.inverse_kinematics(trans, quat)

        action = Base_pb2.Action()
        action.name = "Set Joint Pose"
        action.application_data = ""
        
        for joint_id, angle in enumerate(computed_joint_angles.joint_angles):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = angle.value

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
        # print(f"[Kinova] 移动到关节角: {joint_pose}")
        self.base.ExecuteAction(action)
        
        finished = e.wait(20)
        self.base.Unsubscribe(notification_handle)    

    def check_torque_change(self, torque_threshold=8.0):
            """
            Check if the torque change is within the threshold.
            """
            current_torque = self.get_joint_torque()
            if self.last_torque is not None:
                torque_diff = np.abs(np.array(current_torque) - np.array(self.last_torque))
                max_torque_change = np.max(torque_diff)
                if max_torque_change > torque_threshold:
                    self.stop_motion()
                    if self.last_pos is not None:
                        last_trans, last_quat, _ = self.last_pos
                        # print(f'退回到位置: {last_trans}, {last_quat}')
                        # 使用简单的ee_move方法退回到上一个位置
                        self.ee_move(last_trans, last_quat, asynchronous=False)
                    self.last_torque = None
                    self.last_pos = None
                    return False
                else:
                    return True

            return True


    def ee_move_RRMC(self, trans_target, quat_target):
        """
        Simple resolved-rate motion control (RRMC) that moves the end-effector to the target pose linearly.
        """
        trans_current, quat_current, _ = self.get_ee_pose(frame='local')
        # p_cur = np.array([trans_current[0], trans_current[1], trans_current[2]])
        # R_cur = Rotation.from_quat(quat_current).as_matrix()

        p_des = np.array([trans_target[0], trans_target[1], trans_target[2]])
        R_des = Rotation.from_quat(quat_target).as_matrix()

        trans_threshold = 1e-2 # meter
        rot_threshold = 0.2   # degree

        while True:
        # 当前末端执行器的位姿
            trans_current, quat_current, _ = self.get_ee_pose(frame='local')
            print('quat_current', quat_current)
            p_cur = np.array(trans_current)
            R_cur = Rotation.from_quat(quat_current).as_matrix()

            # 误差计算
            trans_error = np.linalg.norm(p_des - p_cur)  # 欧氏距离
            rot_error = so3_log(R_des @ R_cur.T)
            rot_error = np.linalg.norm(rot_error)
            rot_error_deg = np.degrees(rot_error)
            print('trans_error', trans_error)
            print('rot_error', rot_error_deg)

            # 判断是否收敛
            if trans_error < trans_threshold and rot_error_deg < rot_threshold:
                print('ok')
                self.stop_motion()
                break

            # 获取关节状态和雅可比矩阵
            q = self.get_joint_pose()
            J = self.compute_jacobian_numeric(q)

            # 计算关节速度（RRMC控制器）
            q_v, _ = resolved_rate_motion_control(J, p_cur, R_cur, p_des, R_des)
            print('q_v', q_v)
            self.set_joint_velocity(q_v)

            # break
            # 控制频率 10Hz
            time.sleep(0.1)

    def ee_move(self, trans, quat, asynchronous=False):
        R = Rotation.from_quat(quat)
        euler = R.as_euler('xyz', degrees=True)

        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        action = Base_pb2.Action()
        action.name = "Example Cartesian action movement"
        action.application_data = ""

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = trans[0]        # (meters)
        cartesian_pose.y = trans[1]    # (meters)
        cartesian_pose.z = trans[2]    # (meters)
        cartesian_pose.theta_x = euler[0] # (degrees)
        cartesian_pose.theta_y = euler[1] # (degrees)
        cartesian_pose.theta_z = euler[2] # (degrees)

        e = threading.Event()
        def concise_check(notification, e=e):
            event_name = Base_pb2.ActionEvent.Name(notification.action_event)
            if notification.action_event == Base_pb2.ACTION_ABORT:
                print(f"EVENT : {event_name}  (奇异位置/无法到达/超限), 切换curobo IK")
                # self.set_ee_pose(trans, quat, asynchronous=True)
                # e.set()
                # return
            else:
                print(f"EVENT : {event_name}")
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()

        if asynchronous:
            # notification_handle = self.base.OnNotificationActionTopic(
            #     concise_check,
            #     Base_pb2.NotificationOptions()
            # )
            self.base.ExecuteAction(action)
            return True
        else:
            finished = False
            while not finished:
                notification_handle = self.base.OnNotificationActionTopic(
                    concise_check,
                    Base_pb2.NotificationOptions()
                )                
                self.base.ExecuteAction(action)

                finished = e.wait(TIMEOUT_DURATION_EE)
                self.base.Unsubscribe(notification_handle)

            print('[Kinova ee move result]', finished, TIMEOUT_DURATION_EE)
            print('[Kinova ee move]', cartesian_pose)

            return finished

    # def forward_kinemetics(self, joint_q):
        # pose = self.base.ComputeForwardKinematics(joint_q)
        # print('trans', pose.x, pose.y, pose.z)
        # print('rotate', pose.theta_x, pose.theta_y, pose.theta_z)
        # R_matrix = Rotation.from_euler('xyz', [pose.theta_x, pose.theta_y, pose.theta_z]).as_matrix()
        # translation = np.array([pose.x, pose.y, pose.z])
        # # rotation = np.array([pose.theta_x, pose.theta_y, pose.theta_z])
        # rotation = R_matrix
        # return translation, rotation
    def forward_kinematics(self, joint_angles):
        """
        正运动学：根据关节角计算末端位姿。
        参数：
            joint_angles (list[float]): 长度为7的关节角度列表（度）。
        返回：
            tuple: (x, y, z, theta_x, theta_y, theta_z)
        """
        # pdb.set_trace()
        if not isinstance(joint_angles, (list, tuple, np.ndarray)) or len(joint_angles) != 7:
            raise ValueError("joint_angles 必须为长度为7的列表或元组")
        # from kortex_api.autogen.messages import Base_pb2
        # print('joint_angles', joint_angles)
        joint_angles = np.rad2deg(joint_angles)
        joint_angles = [(angle + 360) % 360 for angle in joint_angles]
        # print('joint_angles', joint_angles)

        joint_angles_msg = Base_pb2.JointAngles()
        for idx, angle in enumerate(joint_angles):
            joint_angle = joint_angles_msg.joint_angles.add()
            joint_angle.joint_identifier = idx
            joint_angle.value = angle

        # Computing Foward Kinematics (Angle -> cartesian convert) from arm's current joint angles
        pose = None
        while pose is None:
            try:
                # print("Computing Foward Kinematics using joint angles...")
                pose = self.base.ComputeForwardKinematics(joint_angles_msg)
            except KServerException as ex:
                print("Unable to compute forward kinematics")
                print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
                print("Caught expected error: {}".format(ex))
                pose = None
                time.sleep(0.1)

        # print('trans', pose.x, pose.y, pose.z)
        # print('rotate', pose.theta_x, pose.theta_y, pose.theta_z)

        rotation_degree = np.array([pose.theta_x, pose.theta_y, pose.theta_z])
        rotation_rad = np.deg2rad(rotation_degree)
        R_matrix = Rotation.from_euler('xyz', rotation_rad).as_matrix()
        translation = np.array([pose.x, pose.y, pose.z])
        # rotation = np.array([pose.theta_x, pose.theta_y, pose.theta_z])

        rotation = R_matrix

        return translation, rotation


    def inverse_kinematics(self, trans, quat):
        # Object containing cartesian coordinates and Angle Guess
        input_IkData = Base_pb2.IKData()
        
        # Fill the IKData Object with the cartesian coordinates that need to be converted
        R = Rotation.from_quat(quat)
        euler = R.as_euler('xyz', degrees=True)
        input_IkData.cartesian_pose.x = trans[0]
        input_IkData.cartesian_pose.y = trans[1]
        input_IkData.cartesian_pose.z = trans[2]
        input_IkData.cartesian_pose.theta_x = euler[0]
        input_IkData.cartesian_pose.theta_y = euler[1]
        input_IkData.cartesian_pose.theta_z = euler[2]

        # Fill the IKData Object with the guessed joint angles
        input_joint_angles = self.base.GetMeasuredJointAngles()
        for joint_angle in input_joint_angles.joint_angles :
            jAngle = input_IkData.guess.joint_angles.add()
            # '- 1' to generate an actual "guess" for current joint angles
            jAngle.value = joint_angle.value - 1
        
        try:
            print("Computing Inverse Kinematics using joint angles and pose...")
            computed_joint_angles = self.base.ComputeInverseKinematics(input_IkData)
        except KServerException as ex:
            print("Unable to compute inverse kinematics")
            print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
            print("Caught expected error: {}".format(ex))
            return False

        print("Joint ID : Joint Angle")
        joint_identifier = 0
        for joint_angle in computed_joint_angles.joint_angles :
            print(joint_identifier, " : ", joint_angle.value)
            joint_identifier += 1

        return computed_joint_angles

    def compute_jacobian_numeric(self, q, h=1e-5):
        """
        Builds a 6 x n spatial Jacobian by central-differencing FK.
        fk(q) must return (p (3,), R (3x3)) in the world frame.
        """
        n = len(q)
        J = np.zeros((6, n))
        p0, R0 = self.forward_kinematics(q)
        for i in range(n):
            dq = np.zeros_like(q); dq[i] = h
            p_plus,  R_plus  = self.forward_kinematics(q + dq)
            p_minus, R_minus = self.forward_kinematics(q - dq)
            # linear part
            J[0:3, i] = (p_plus - p_minus) / (2*h)
            # angular part: local logs around current pose to reduce bias
            w_plus  = so3_log(R_plus  @ R0.T)
            w_minus = so3_log(R_minus @ R0.T)
            J[3:6, i] = (w_plus - w_minus) / (2*h)
        return J

    def check_fk(self):
        current_q = self.get_joint_pose()
        fk_ee_trans, fk_ee_quat = self.ik_solver.forward_kinematics(current_q)
        lib_ee_trans, lib_ee_quat, lib_ee_rpy = self.get_ee_pose(frame='local')

        input_joint_angles = self.base.GetMeasuredJointAngles()
        self.forward_kinematics(input_joint_angles)

        print('FK:', fk_ee_trans, fk_ee_quat)
        print('lib:', lib_ee_trans, lib_ee_quat)

    def stop_motion(self):
        """
        停止机械臂运动。
        """
        try:
            self.base.Stop()
            print("[Kinova] 机械臂运动已停止")
            return True
        except Exception as e:
            print(f"[Kinova] 停止机械臂运动失败: {e}")
            return False

