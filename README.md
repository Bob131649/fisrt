# robot-arms

Unified Python helpers for controlling Franka and Kinova robot arms, plus cuRobo-based IK utilities. Ships URDFs internally and exposes a small, clear API you can use from other projects.

## Install

- Editable (recommended for development):
  - `pip install -e .`
- Regular install:
  - `pip install .`

Notes
- Python 3.8+ is recommended.
- Franka control depends on `franky` (install separately if needed).
- Kinova control requires the Kinova SDK (`kortex_api`) and optionally `pylibrm` for the gripper; these are included in the `[kinova]` extra if available.

### External SDKs (Franka, Kinova, Gripper)

This package provides a thin convenience layer. You still need the vendor SDKs/drivers installed for your hardware. Follow the upstream instructions linked below, then verify imports before using the helpers here.

#### Franka (Franky)

- Repo: https://github.com/TimSchneider42/franky


#### Kinova Kortex2 SDK (Gen3 / Gen3 Lite)

- Repo/Docs: https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L
- A simple documentation is inside folder ./external/kinova/

#### Gripper SDK/API (RMAXIS v6)

- Docs: https://doc.rmaxis.com/docs/sdk/api_v6/
- The python sdk with a simple documentation are inside folder ./external/gripper_SDK/

### cuRobo Install

- Docs: https://curobo.org/get_started/1_install_instructions.html


## Quickstart

Franka
```python
from robot_arms.franka.franka_basic import Franka

robot = Franka(robot_ip="192.168.31.11")
robot.set_ee_pose([0.4, 0.0, 0.3], [0, 0, 0, 1], asynchronous=False)
print("joints:", robot.get_joint_pose())
```

Kinova
```python
from robot_arms.kinova.kinova_basic import Kinova

arm = Kinova(robot_ip="192.168.31.13")
arm.open_gripper()
arm.set_ee_pose([0.6, 0.0, 0.3], [0, 0, 0, 1], asynchronous=False)
```

cuRobo IK
```python
import numpy as np
from robot_arms.solvers.curobo_solver import CuRoboIKSolver

ik = CuRoboIKSolver("franka")
current_q = np.zeros(7)
pos = np.array([0.4, 0.0, 0.3])
quat_xyzw = np.array([0, 0, 0, 1])
q_target = ik.get_target_joint(current_q, pos, quat_xyzw)
pos_fk, quat_fk = ik.forward_kinematics(q_target)
```

## Accessing URDFs
URDFs are packaged and can be resolved at runtime:
```python
from importlib.resources import files
franka_urdf = files("robot_arms.descriptions.urdf.franka") / "fr3_franka_hand.urdf"
kinova_urdf = files("robot_arms.descriptions.urdf.kinova") / "kinova_arm.urdf"
```

Make sure the URDF TCP link matches your arm settings in the configuration webpage!

## Environment

- IPs: Set the correct robot IPs when constructing controllers (see Quickstart examples).
- Python: Use a virtual environment. CUDA is required for cuRobo features.
- Verification: Before using these helpers, verify that vendor SDK imports work as shown above.
