# PyLibRM API 文档

## 项目概述

PyLibRM 是一个用于控制 RobustMotion 夹爪/机械轴的 Python 库。该库基于 Modbus 通信协议，支持 RTU 和 TCP 两种连接方式，提供了完整的运动控制、力控制和状态监测功能。

**版本**: 0.0.8
**作者**: Kang Yao (yaokang@rmaxis.com)
**许可证**: MIT License


## 特性

- 支持 Modbus RTU 和 TCP 通信
- 多种运动模式：绝对运动、相对运动、推压运动、精密推压等
- 力控制和力传感器读取
- 位置监控和状态反馈
- 错误检测和处理
- 命令队列和批处理执行
- 数据采集和监控功能

## 安装

### 依赖要求

- Python >= 3.6
- pymodbus == 2.5.3

### 安装方法

```bash
pip install pylibrm
```

或从源码安装：

```bash
git clone <repository-url>
cd pylibrm
python setup.py install
```

## 快速开始

### 基本连接

```python
import pylibrm.RMAxis as RMAxis

# Modbus RTU 连接
axis_rtu = RMAxis.Axis_V6.create_modbus_rtu('\\\\.\\COM8', 115200, 1)

# Modbus TCP 连接
axis_tcp = RMAxis.Axis_V6.create_modbus_tcp('192.168.1.100', 502, 1)
```

### 基本操作

```python
# 设置运动参数
axis_rtu.config_motion(50.0, 100.0)  # 速度50mm/s, 加速度100mm/s²

## 绝对运动到指定位置
#----------------------------------------
axis_rtu.move_absolute(10.0, 50.0, 100.0, 100.0, 0.1)
#-------------非阻塞的！ 0-40--------------
#-----------------------------------------
# 读取当前位置
position = axis_rtu.position()
print(f"当前位置: {position} mm")

# 关闭连接
axis_rtu.close()
```

## 核心类

### Axis_V6

核心控制类，提供对夹爪/机械轴的完整控制功能。

#### 初始化方法

##### `create_modbus_rtu(port, baudrate, slave_id)`

创建 Modbus RTU 连接

**参数:**
- `port` (str): 串口名称，如 'COM8' 或 '/dev/ttyUSB0'
- `baudrate` (int): 波特率，默认 115200
- `slave_id` (int): 从设备ID

**返回:** `Axis_V6` 实例

**示例:**
```python
axis = Axis_V6.create_modbus_rtu('\\\\.\\COM8', 115200, 1)
```

##### `create_modbus_tcp(host, port, slave_id)`

创建 Modbus TCP 连接

**参数:**
- `host` (str): 主机IP地址
- `port` (int): 端口号，通常为 502
- `slave_id` (int): 从设备ID

**返回:** `Axis_V6` 实例

**示例:**
```python
axis = Axis_V6.create_modbus_tcp('192.168.1.100', 502, 1)
```

## API 参考

### 连接管理

#### `set_retries(max_retry)`
设置最大重试次数

**参数:**
- `max_retry` (int): 最大重试次数

#### `set_timeout(timeout_ms)`
设置通信超时时间

**参数:**
- `timeout_ms` (int): 超时时间（毫秒）

#### `close()`
关闭连接

### 运动控制

#### `config_motion(velocity, acceleration)`
配置直接运动模式的速度和加速度

**参数:**
- `velocity` (float): 速度 (mm/s)
- `acceleration` (float): 加速度 (mm/s²)

#### `read_config_motion()`
读取当前运动配置

**返回:** `[velocity, acceleration]` 列表

#### `move_to(position)`
直接运动模式移动到指定位置

**参数:**
- `position` (float): 目标位置 (mm)

#### `move_absolute(position, velocity, acceleration, deceleration, band)`
绝对位置运动

**参数:**
- `position` (float): 目标位置 (mm)
- `velocity` (float): 速度 (mm/s)
- `acceleration` (float): 加速度 (mm/s²)
- `deceleration` (float): 减速度 (mm/s²)
- `band` (float): 定位精度范围 (mm)

**示例:**
```python
# 移动到位置10mm，速度50mm/s，加减速度100mm/s²，精度0.1mm
axis.move_absolute(10.0, 50.0, 100.0, 100.0, 0.1)
```

#### `push(distance, velocity, acceleration, force_limit, pos_band_mm, time_band_ms)`
推压运动
## ！！！！！
# 停下来条件：达到位置或达到指定力   时间是力控检测时间
# 第四个参数是力控的力的百分比

**参数:**
- `distance` (float): 推压距离 (mm)
- `velocity` (float): 速度 (mm/s)
- `acceleration` (float): 加速度 (mm/s²)
- `force_limit` (float): 力限制（0-1之间的比例值）
- `pos_band_mm` (float): 位置范围 (mm)
- `time_band_ms` (float): 时间范围 (ms)

**示例:**
```python
# 推压10mm，速度20mm/s，加速度500mm/s²，力限制15%
axis.push(10.0, 20.0, 500.0, 0.15, 0.1, 500.0)
```

#### `precise_push(distance, force, velocity_factor, impact_factor, band_n, band_ms)`
精密推压----------------精确力值

**参数:**
- `distance` (float): 推压距离 (mm)
- `force` (float): 目标力 (N)
- `velocity_factor` (float): 速度系数
- `impact_factor` (float): 冲击系数
- `band_n` (float): 力定位范围 (N)
- `band_ms` (float): 稳压时间 (ms)

**示例:**
```python
# 精密推压10mm，目标力10N，速度系数1，冲击系数0
axis.precise_push(10.0, 10.0, 1.0, 0.0, 0.1, 500.0)
```

#### `precise_touch(distance, velocity, acceleration, force_threshold, band_mm)`
精密触碰

**参数:**
- `distance` (float): 移动距离 (mm)
- `velocity` (float): 速度 (mm/s)
- `acceleration` (float): 加速度 (mm/s²)
- `force_threshold` (float): 力阈值 (N)
- `band_mm` (float): 定位范围 (mm)

#### `go_home()`
回原点

#### `go_home_z(distance, velocity, acceleration, force_limit, band_mm)`
Z轴回原点

**参数:**
- `distance` (float): 移动距离 (mm)
- `velocity` (float): 速度 (mm/s)
- `acceleration` (float): 加速度 (mm/s²)
- `force_limit` (float): 力限制（0-1之间的比例值）
- `band_mm` (float): 定位范围 (mm)

### 状态读取

#### `position()`
读取当前位置

**返回:** `float` - 当前位置 (mm)

#### `velocity()`
读取当前速度

**返回:** `float` - 当前速度 (mm/s)

#### `torque()`
读取当前扭矩

**返回:** `float` - 当前扭矩

#### `force_sensor()`
读取力传感器数值

**返回:** `float` - 力传感器读数 (N)

#### `read_external_force()`
读取外部力传感器数值

**返回:** `float` - 外部力传感器读数 (N)

### 状态判断

#### `is_finished(index)`
判断指定序号的命令是否完成

**参数:**
- `index` (int): 命令序号 (0-15)

**返回:** `bool` - 是否完成

#### `is_moving()`
判断是否正在运动

**返回:** `bool` - 是否在运动

#### `is_captured()`
判断是否夹持成功（用于推压运动）

**返回:** `bool` - 是否夹持成功

#### `is_reached()`
判断是否到达目标（用于绝对运动、精密推压、相对运动）

**返回:** `bool` - 是否到达目标

#### `is_push_empty()`
判断是否空推

**返回:** `bool` - 是否空推

#### `is_ready()`
判断设备是否就绪

**返回:** `bool` - 是否就绪

#### `get_command_status()`
获取命令状态

**返回:** `float` - 状态码
- 0: pending（等待中）
- 1: 完成
- 2: 停止
- 3: 位置到达
- 4: 力到达

### 命令管理

#### `set_command(index, command)`
设置指定序号的命令

**参数:**
- `index` (int): 命令序号 (0-15)
- `command` (dict): 命令字典

**命令类型和参数:**

##### 1. 无命令 (COMMAND_NONE)
```python
command = {
    "type": 0,
    "next_command_index": -1
}
```

##### 2. 回原点 (COMMAND_GO_HOME)
```python
command = {
    "type": 1,
    "next_command_index": 0,
    "origin_offset": 0.2  # 原点偏移
}
```

##### 3. 延时 (COMMAND_DELAY)
```python
command = {
    "type": 2,
    "next_command_index": 1,
    "ms": 1000  # 延时时间(ms)
}
```

##### 4. 绝对运动 (COMMAND_MOVE_ABSOLUTE)
```python
command = {
    "type": 3,
    "next_command_index": 2,
    "position": 10.0,      # 目标位置(mm)
    "velocity": 50.0,      # 速度(mm/s)
    "acceleration": 100.0, # 加速度(mm/s²)
    "deceleration": 100.0, # 减速度(mm/s²)
    "band": 0.1           # 定位精度(mm)
}
```

##### 5. 推压运动 (COMMAND_PUSH)
```python
command = {
    "type": 4,
    "next_command_index": 3,
    "distance": 10.0,       # 推压距离(mm)
    "velocity": 20.0,       # 速度(mm/s)
    "acceleration": 100.0,  # 加速度(mm/s²)
    "force_limit": 0.15,    # 力限制(0-1)
    "pos_band_mm": 0.1,     # 位置范围(mm)
    "time_band_ms": 500.0   # 时间范围(ms)
}
```

##### 6. 相对运动 (COMMAND_MOVE_RELATIVE)
```python
command = {
    "type": 5,
    "next_command_index": 4,
    "distance": 5.0,        # 移动距离(mm)
    "velocity": 30.0,       # 速度(mm/s)
    "acceleration": 100.0,  # 加速度(mm/s²)
    "deceleration": 100.0,  # 减速度(mm/s²)
    "band": 0.1            # 定位精度(mm)
}
```

##### 7. 精密推压 (COMMAND_PRECISE_PUSH)
```python
command = {
    "type": 6,
    "next_command_index": 5,
    "distance": 10.0,         # 推压距离(mm)
    "force": 5.0,            # 目标力(N)
    "velocity_factor": 1.0,   # 速度系数
    "impact_factor": 0.0,     # 冲击系数
    "band_n": 0.1,           # 力定位范围(N)
    "band_ms": 500.0         # 稳压时间(ms)
}
```

##### 8. 重置力 (COMMAND_RESET_FORCE)
```python
command = {
    "type": 7,
    "next_command_index": 6
}
```

##### 9. 停止 (COMMAND_STOP)
```python
command = {
    "type": 8,
    "next_command_index": 7
}
```

##### 10. 开始采集 (COMMAND_START_MONITOR)
```python
command = {
    "type": 9,
    "next_command_index": 8,
    "freq_hz": 1000,    # 采集频率(Hz)
    "length": 2,        # 采集通道数(最大3)
    "count": 1000,      # 采集数量
    "var_0": 0,         # 变量0(0:位置,1:速度,2:出力,3:目标位置,4:受力)
    "var_1": 4,         # 变量1
    "var_2": 0          # 变量2
}
```

##### 11. Z回原点 (COMMAND_GO_HOME_Z)
```python
command = {
    "type": 10,
    "next_command_index": 9,
    "distance": 10.0,       # 移动距离(mm)
    "velocity": 20.0,       # 速度(mm/s)
    "acceleration": 100.0,  # 加速度(mm/s²)
    "force_limit": 0.5,     # 力限制(0-1)
    "band_mm": 0.1         # 定位范围(mm)
}
```

##### 12. 精密触碰 (COMMAND_PRECISE_TOUCH)
```python
command = {
    "type": 11,
    "next_command_index": 10,
    "distance": 10.0,         # 移动距离(mm)
    "velocity": 20.0,         # 速度(mm/s)
    "acceleration": 100.0,    # 加速度(mm/s²)
    "force_threshold": 1.0,   # 力阈值(N)
    "band_mm": 0.1           # 定位范围(mm)
}
```

#### `get_command(index)`
获取指定序号的命令

**参数:**
- `index` (int): 命令序号 (0-15)

**返回:** `dict` - 命令字典

#### `trig_command(index)`
触发执行指定序号的命令

**参数:**
- `index` (int): 命令序号 (0-15)

#### `exec_command(command)`
直接执行命令

**参数:**
- `command` (dict): 命令字典

#### `save_command()`
保存命令到设备

#### `load_commands()`
从设备加载命令

### 控制操作

#### `stop()`
停止当前运动

#### `reset_error()`
重置错误状态

#### `reset_force()`
重置力传感器

#### `set_servo_on_off(on_off)`
设置伺服开关

**参数:**
- `on_off` (bool): 开关状态

### 错误处理

#### `error_code()`
获取错误代码列表

**返回:** `list` - 错误代码列表
- "Position_deviation_overflow": 位置偏差溢出
- "Velocity_deviation_overflow": 速度偏差溢出  
- "Motor_stuck": 电机卡死
- "Force_overload": 力过载

#### `enable_error_position_deviation_overflow(on_off)`
启用/禁用位置偏差溢出检测

#### `enable_error_velocity_deviation_overflow(on_off)`
启用/禁用速度偏差溢出检测

#### `enable_motor_stuck(on_off)`
启用/禁用电机卡死检测

### 数据采集

#### `start_monitor(next, freq_hz, length, count, var_0, var_1, var_2)`
开始数据监控

**参数:**
- `next` (int): 下一个命令索引
- `freq_hz` (int): 采集频率 (Hz)
- `length` (int): 采集通道数 (最大3)
- `count` (int): 采集数量
- `var_0, var_1, var_2` (int): 采集变量 (0:位置,1:速度,2:出力,3:目标位置,4:受力)

#### `sample_result()`
获取采集结果

**返回:** `list` - 采集数据列表

### 参数设置

#### `set_command_force(index, force)`
设置精密推压命令的力参数

**参数:**
- `index` (int): 命令序号
- `force` (float): 力值 (N)

#### `set_command_pos(index, pos)`
设置绝对运动命令的位置参数

**参数:**
- `index` (int): 命令序号
- `pos` (float): 位置 (mm)

### IO控制

#### `set_b0_state(on_off)`, `set_b1_state(on_off)`, `set_b3_state(on_off)`
设置布尔输出状态

#### `set_f1(pos)`, `set_f2(pos)`, `set_f3(force_limit)`
设置浮点参数

#### `set_f4(coefficient)`, `set_f5(coefficient)`, `set_f6(coefficient)`
设置系数参数

### 工具方法

#### `wait(time_in_ms)`
等待指定时间

**参数:**
- `time_in_ms` (int): 等待时间 (毫秒)

#### `wait_for_reached(timeout)`
等待到达目标位置

**参数:**
- `timeout` (int): 超时时间 (毫秒)

#### `get_version()`
获取设备版本信息

**返回:** `dict` - 版本信息字典
```python
{
    "major": 1,
    "minor": 0,
    "build": 5,
    "type": 1
}
```

## 常量定义

### 命令类型
```python
COMMAND_NONE = 0              # 无命令
COMMAND_GO_HOME = 1           # 回原点
COMMAND_DELAY = 2             # 延时
COMMAND_MOVE_ABSOLUTE = 3     # 绝对运动
COMMAND_PUSH = 4              # 推压运动
COMMAND_MOVE_RELATIVE = 5     # 相对运动
COMMAND_PRECISE_PUSH = 6      # 精密推压
COMMAND_RESET_FORCE = 7       # 重置力
COMMAND_STOP = 8              # 停止
COMMAND_START_MONITOR = 9     # 开始采集
COMMAND_GO_HOME_Z = 10        # Z回原点
COMMAND_PRECISE_TOUCH = 11    # 精密触碰
```

### 状态观察器
```python
STATE_OBSERVER_POSITION = 0   # 位置
STATE_OBSERVER_VELOCITY = 2   # 速度
STATE_OBSERVER_TORQUE = 4     # 扭矩
```

## 完整示例

### 基本运动控制示例

```python
import pylibrm.RMAxis as RMAxis
from time import sleep

# 创建连接
axis = RMAxis.Axis_V6.create_modbus_rtu('\\\\.\\COM8', 115200, 1)

try:
    # 设置运动参数
    axis.config_motion(50.0, 100.0)
    
    # 绝对运动到位置10mm
    axis.move_absolute(10.0, 50.0, 100.0, 100.0, 0.1)
    
    # 等待运动完成
    while not axis.is_finished(15):
        sleep(0.1)
    
    # 推压运动
    axis.push(5.0, 20.0, 100.0, 0.2, 0.1, 500.0)
    
    # 等待推压完成
    while not axis.is_finished(15):
        sleep(0.1)
    
    # 检查是否夹持成功
    if axis.is_captured():
        print("夹持成功")
    else:
        print("夹持失败")
    
    # 读取当前状态
    pos = axis.position()
    force = axis.force_sensor()
    print(f"当前位置: {pos:.2f} mm")
    print(f"当前力: {force:.2f} N")
    
finally:
    # 关闭连接
    axis.close()
```

### 命令队列示例

```python
import pylibrm.RMAxis as RMAxis

axis = RMAxis.Axis_V6.create_modbus_rtu('\\\\.\\COM8', 115200, 1)

try:
    # 设置命令序列
    
    # 命令0: 回原点
    command0 = {
        "type": 1,
        "next_command_index": 1,
        "origin_offset": 0.0
    }
    axis.set_command(0, command0)
    
    # 命令1: 绝对运动
    command1 = {
        "type": 3,
        "next_command_index": 2,
        "position": 10.0,
        "velocity": 50.0,
        "acceleration": 100.0,
        "deceleration": 100.0,
        "band": 0.1
    }
    axis.set_command(1, command1)
    
    # 命令2: 精密推压
    command2 = {
        "type": 6,
        "next_command_index": -1,
        "distance": 5.0,
        "force": 10.0,
        "velocity_factor": 1.0,
        "impact_factor": 0.0,
        "band_n": 0.5,
        "band_ms": 1000.0
    }
    axis.set_command(2, command2)
    
    # 保存命令
    axis.save_command()
    
    # 执行命令序列
    axis.trig_command(0)
    
    # 监控执行状态
    while not axis.is_finished(15):
        status = axis.get_command_status()
        pos = axis.position()
        print(f"状态: {status}, 位置: {pos:.2f} mm")
        sleep(0.2)
    
finally:
    axis.close()
```

### 数据采集示例

```python
import pylibrm.RMAxis as RMAxis

axis = RMAxis.Axis_V6.create_modbus_rtu('\\\\.\\COM8', 115200, 1)

try:
    # 开始数据采集
    # 采集频率1000Hz，2个通道，采集1000个数据点
    # var_0=0(位置), var_1=4(受力)
    axis.start_monitor(-1, 1000, 2, 1000, 0, 4, 0)
    
    # 同时执行运动
    axis.move_absolute(20.0, 30.0, 100.0, 100.0, 0.1)
    
    # 获取采集结果
    data = axis.sample_result()
    print(f"采集到 {len(data)} 个数据点")
    
    # 处理数据（这里只是简单打印前10个点）
    for i in range(min(10, len(data))):
        print(f"数据点 {i}: {data[i]}")
    
finally:
    axis.close()
```

## 错误处理

```python
import pylibrm.RMAxis as RMAxis

axis = RMAxis.Axis_V6.create_modbus_rtu('\\\\.\\COM8', 115200, 1)

try:
    # 检查设备状态
    if not axis.is_ready():
        # 检查错误
        errors = axis.error_code()
        if errors:
            print(f"检测到错误: {errors}")
            # 重置错误
            axis.reset_error()
        
        # 启用伺服
        axis.set_servo_on_off(True)
    
    # 执行运动...
    
except Exception as e:
    print(f"执行过程中发生错误: {e}")
    # 停止运动
    axis.stop()
    # 重置错误
    axis.reset_error()

finally:
    axis.close()
```

## 注意事项

1. **连接管理**: 使用完毕后务必调用 `close()` 方法关闭连接
2. **错误处理**: 定期检查 `error_code()` 并在必要时调用 `reset_error()`
3. **状态监控**: 在执行运动命令后使用 `is_finished()` 等方法监控完成状态
4. **参数范围**: 
   - 力限制参数通常为 0-1 之间的比例值
   - 位置和速度单位为毫米和毫米/秒
   - 力的单位为牛顿(N)
5. **通信稳定性**: 可通过 `set_retries()` 和 `set_timeout()` 调整通信参数
6. **命令序列**: 使用命令队列时注意设置正确的 `next_command_index`

## 许可证

MIT License - 详见 LICENSE.txt 文件 