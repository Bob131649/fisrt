## 官方python库
https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L
- 官方python库，独立于ROS
- 比较底层，为了方便使用，需要进一步封装
- 自带 末端位置控制、关节角控制、阻抗控制、IK、关节角末端位置读取、关节力矩读取
- 自带IK并不好用，自带IK基于初始猜测值优化采样，因此有时候会解不出来（初始猜测值不好的话）
- 自带末端控制同样不好用，原因就是IK不好用
> 可能ROS不存在这些问题，但本人没有尝试

## 机械臂本体
分为：
- Gen3 7自由度
- Gen3 lite 6自由度
### urdf
https://github.com/Kinovarobotics/ros2_kortex/tree/main/kortex_description/arms/gen3/7dof/urdf

## 自写python封装
根据官方python example，上层进一步封装了简洁的接口，参考程序，其中ik解算使用curo

## API文档

### 初始化
```python
kinova = Kinova(
    robot_ip="192.168.31.13",           # 机器人IP地址
    port=10000,                         # 通信端口
    relative_dynamics_factor=0.05,      # 运动学相对动态因子
    gripper_port="/dev/ttyUSB0",        # 夹爪串口路径
    gripper_baudrate=115200,            # 夹爪波特率
    gripper_slave_id=1,                 # 夹爪从设备ID
    use_curo_ik=False                   # 是否使用curo IK求解器
)
```

### 基础运动控制

#### 关节空间控制
```python
# 设置关节角度 (度)
kinova.set_joint_pose([0, 0, 0, 0, 0, 0, 0], asynchronous=False)

# 执行关节轨迹
kinova.set_joint_trajectories(joint_trajectory, velocity_trajectory=None, asynchronous=False)

# 发送关节速度命令 (度/秒)
kinova.send_joint_speeds([10.0, 0.0, -10.0, 0.0, 10.0, 0.0, -10.0], duration=0)
```

#### 笛卡尔空间控制
```python
# 末端位置控制 (位置+四元数)
kinova.ee_move([0.5, 0.0, 0.3], [0, 0, 0, 1], asynchronous=False)

# 使用curo IK的末端控制
kinova.ee_move_curo([0.5, 0.0, 0.3], [0, 0, 0, 1], asynchronous=False)

# 笛卡尔相对运动
kinova.cartesian_move(dx=0.1, dy=0, dz=0, dtheta_x=0, dtheta_y=0, dtheta_z=0)

# 执行笛卡尔轨迹
waypoints = [
    (0.7, 0.0, 0.5, 0.0, 90.0, 0.0, 90.0),    # (x, y, z, blending_radius, theta_x, theta_y, theta_z)
    (0.7, 0.0, 0.33, 0.1, 90.0, 0.0, 90.0)
]
kinova.set_ee_trajectory(waypoints, optimize=False, asynchronous=False)
kinova.set_ee_trajectory_with_optimization(waypoints, asynchronous=False)
```

### 状态查询
```python
# 获取关节角度 (度)
joint_pose = kinova.get_joint_pose()

# 获取关节速度 (度/秒)
joint_vel = kinova.get_joint_vel()

# 获取关节力矩
joint_torque = kinova.get_joint_torque()

# 获取末端位姿
ee_pos, ee_quat, ee_euler = kinova.get_ee_pose()

# 获取末端受力
fx, fy, fz, total_force = kinova.get_ee_force()
```

### 运动学计算
```python
# 正运动学 (关节角 -> 末端位姿)
pose = kinova.compute_fk([0, 0, 0, 0, 0, 0, 0])  # 返回 (x, y, z, theta_x, theta_y, theta_z)

# 逆运动学 (末端位姿 -> 关节角)
joint_angles = kinova.compute_ik((0.5, 0.0, 0.3, 0, 0, 0), guess=None)

# 使用curo IK求解器
joint_angles = kinova.curo_ik([0.5, 0.0, 0.3], [0, 0, 0, 1], guess=None)
ee_pos, ee_quat = kinova.curo_fk([0, 0, 0, 0, 0, 0, 0])
```

### 夹爪控制
```python
# 夹爪开合
kinova.open_gripper(asynchronous=True)
kinova.close_gripper(asynchronous=True)

# 设置夹爪开口宽度 (毫米)
kinova.set_gripper_opening(20.0, asynchronous=True)

# 精密推压
kinova.gripper_precise_push(distance=10.0, force=5.0, velocity_factor=1.0)

# 夹爪状态查询
width = kinova.get_gripper_width()      # 开口宽度 (毫米)
force = kinova.get_gripper_force()      # 力传感器读数 (牛顿)
is_moving = kinova.is_gripper_moving()  # 是否在运动
is_captured = kinova.is_gripper_captured()  # 是否夹持成功

# 夹爪控制
kinova.gripper_go_home()    # 回原点
kinova.gripper_stop()       # 停止运动
kinova.gripper_reset_error()  # 重置错误
```

### 阻抗控制
```python
# 启动阻抗控制
kinova.start_impedance_control(stiffness=0.5, damping=0.1, duration=30)

# 停止阻抗控制
kinova.stop_impedance_control()
```

### 其他功能
```python
# 停止运动
kinova.stop_motion()

# 移动到默认姿态
kinova.set_default_pose()

# 恢复错误状态
kinova.recover_from_errors()

# 关闭连接
kinova.close()
```

### 注意事项
- 所有角度单位为度，位置单位为米
- 四元数格式为 [x, y, z, w]
- 欧拉角使用Kinova的xyz外旋定义
- 异步模式下函数立即返回，不等待动作完成
- 夹爪功能需要安装PyLibRM库
- curo IK需要安装curobo库
- ！！如果要自写上层封装，注意kinova底层关节角协议0-360度，末端欧拉角是-pi~pi
