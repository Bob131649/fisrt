from dataclasses import dataclass


@dataclass(frozen=True)
class RobotConfig:
    ip: str = "192.168.31.12"
    arm: str = "basic"
    control_mode: str = "armlib"
    relative_dynamics_factor: float = 0.02
    async_motion: bool = True


@dataclass(frozen=True)
class CameraConfig:
    serial: str = "230422272989"
    width: int = 640
    height: int = 480
    fps: int = 30


ROBOT = RobotConfig()
CAMERA = CameraConfig()

DEFAULT_STEPS = 200
DEFAULT_RATE_HZ = 5.0
ACTION_SMOOTHING = 0.6
OPEN_GRIPPER_WIDTH = 0.04
CLOSED_GRIPPER_WIDTH = 0.0
GRIPPER_DEADBAND = 0.001

TRAIN_RGB_CROP = (0, 90, 360, 480)  # top, left, height, width
FALLBACK_IMAGE_SIZE = (224, 224)
