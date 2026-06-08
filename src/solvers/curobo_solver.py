import torch
import numpy as np
import importlib.resources as ir
# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel, CudaRobotModelConfig,
)

class CuRoboIKSolver:
    """
    IK-only solver using cuRobo that:
      - loads robot from name ("franka" or "kinova"),
      - extracts joint limits from URDF via RobotConfig,
      - computes an IK target joint pose for a desired EE pose,
      - selects the target by balancing motion smoothness vs staying away from limits.
    """

    def __init__(
        self,
        robot_name: str,
        num_seeds: int = 100,
        position_threshold: float = 0.001,
        rotation_threshold: float = 0.01,
        use_cuda_graph: bool = True,
    ):
        """
        Args:
            robot_name: "franka" or "kinova"
            num_seeds:   number of parallel IK seeds
            position_threshold: IK positional tolerance (meters)
            rotation_threshold: IK angular tolerance (radians)
            use_cuda_graph: enable CUDA graphs for speed (fixed shapes recommended)
        """
        self.tensor_args = TensorDeviceType()  # picks CUDA if available
        self.device = self.tensor_args.device
        self.dtype = self.tensor_args.dtype
        self.num_seeds = num_seeds

        name = robot_name.strip().lower()
        if name == "franka":
            urdf_file = str(ir.files("robot_arms.descriptions.urdf.franka") / "fr3_franka_hand.urdf")
            base_link = "base"
            ee_link   = "fr3_hand_tcp"
        elif name == "kinova":
            urdf_file = str(ir.files("robot_arms.descriptions.urdf.kinova") / "kinova_arm.urdf")
            base_link = "base_link"
            ee_link   = "tcp_link"
        else:
            raise ValueError(f"Unsupported robot '{robot_name}'. Use 'franka' or 'kinova'.")

        # --- build RobotConfig from URDF (this pulls limits from the URDF)
        robot_cfg = RobotConfig.from_basic(
            urdf_file,
            base_link,
            ee_link,
            self.tensor_args
        )
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

        #--- get joint limits
        cuda_cfg = CudaRobotModelConfig.from_basic_urdf(
            urdf_file, base_link, ee_link, tensor_args=self.tensor_args
        )
        self.robot_model = CudaRobotModel(cuda_cfg)
        jl = self.robot_model.get_joint_limits()  # JointLimits object
        self.q_lower = jl.position[0, :].to(self.device, self.dtype)
        self.q_upper = jl.position[1, :].to(self.device, self.dtype)

        # --- IK solver config
        self.ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=None,                 # add a world if you want collision checks here
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            num_seeds=num_seeds,
            self_collision_check=True,       # set True if you need it
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=use_cuda_graph,
        )
        self.ik = IKSolver(self.ik_cfg)

    @staticmethod
    def _xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (x,y,z,w) -> (w,x,y,z)."""
        # Supports [B,4] or [4]
        if quat_xyzw.dim() == 1:
            x, y, z, w = quat_xyzw
            return torch.stack((w, x, y, z), dim=0)
        elif quat_xyzw.dim() == 2:
            x, y, z, w = quat_xyzw.unbind(dim=1)
            return torch.stack((w, x, y, z), dim=1)
        else:
            raise ValueError("Quaternion tensor must be shape [4] or [B,4].")

    @staticmethod
    def _wxyz_to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (x,y,z,w) -> (w,x,y,z)."""
        # Supports [B,4] or [4]
        if quat_wxyz.dim() == 1:
            w, x, y, z = quat_wxyz
            return torch.stack((x, y, z, w), dim=0)
        elif quat_wxyz.dim() == 2:
            w, x, y, z = quat_wxyz.unbind(dim=1)
            return torch.stack((x, y, z, w), dim=1)
        else:
            raise ValueError("Quaternion tensor must be shape [4] or [B,4].")

    def get_target_joint(
        self,
        current_q: np.ndarray,
        ee_trans: np.ndarray,
        ee_quat_xyzw: np.ndarray,
        ratio: float = 0.0,
    ) -> np.ndarray:
        """
        Args:
            current_q:       (dof,) numpy array of current joint angles (radians)
            ee_trans:        (3,) numpy array of target position (meters)
            ee_quat_xyzw:    (4,) numpy array of target quaternion (x,y,z,w)
            ratio:           float in [0,1]. 0 -> prioritize smoothness,
                                              1 -> prioritize staying away from limits.

        Returns:
            q_best:          (dof,) numpy array with the selected joint target.
        """        
        # --- convert inputs to torch
        current_q = torch.as_tensor(np.array(current_q), device=self.device, dtype=self.dtype).view(-1)
        # out = self.kin_model.get_state(current_q.view(1, -1))

        pos = torch.as_tensor(ee_trans, device=self.device, dtype=self.dtype).view(1, 3)
        quat_xyzw = torch.as_tensor(ee_quat_xyzw, device=self.device, dtype=self.dtype).view(1, 4)
        quat_wxyz = self._xyzw_to_wxyz(quat_xyzw)

        # Build cuRobo Pose (expects batches)
        goal = Pose(pos, quat_wxyz)
        res = self.ik.solve_single(goal,
                                   seed_config=current_q.view(1, 1, -1),
                                   retract_config=current_q.view(1, -1),
                                   return_seeds=self.num_seeds,
                                   )

        ok = res.success.view(-1)
        if not ok.any():
            raise RuntimeError("IK failed: no valid solutions for the given EE pose.")
        # q_best = res.solution[ok][0][0]  # [N, dof]

        candidates = res.solution.squeeze()[ok]  # [N, dof]

        # balancing between smoothness and staying away from limits
        q_lower = self.q_lower
        q_upper = self.q_upper
        ranges = (q_upper - q_lower).clamp(min=1e-6)

        # print(candidates)
        # print(current_q)
        # print(q_lower, q_upper, ranges)

        margins = torch.minimum(candidates - q_lower, q_upper - candidates) / ranges  # [N, dof], in [0..0.5]
        margins = torch.clamp(margins, min=0.0, max=0.2)
        mean_margin = torch.mean(margins, dim=1)  # [N]

        angle_diff = (candidates - current_q + np.pi) % (2*np.pi) - np.pi
        motion = torch.linalg.norm(angle_diff, dim=1)                     # [N]
        # motion = torch.mean(torch.abs(candidates - current_q), dim=1)  # [N], mean absolute joint motion

        r = float(max(0.0, min(1.0, ratio)))
        cost = (1.0 - r) * motion - r * mean_margin

        best_idx = torch.argmin(cost)
        q_best = candidates[best_idx]

        # out = self.kin_model.get_state(q_best.view(1, -1))
        # print('[curobo] target received:', ee_trans, ee_quat_xyzw)
        # print('[curobo] target FK:', out.ee_position, out.ee_quaternion)
        # print('[curobo] target q value:', q_best.shape, q_best)
        return q_best.detach().cpu().numpy().tolist()

    def forward_kinematics(self, joint_q):
        joint_q = torch.as_tensor(joint_q, device=self.device, dtype=self.dtype).view(-1)
        out = self.kin_model.get_state(joint_q.view(1, -1))
        quat_xyzw = self._wxyz_to_xyzw(out.ee_quaternion)
        return out.ee_position, quat_xyzw
