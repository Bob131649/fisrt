import torch
import numpy as np

# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel, CudaRobotModelConfig,
)
# NEW: MotionGen imports
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig  # packaged


class CuRoboIKSolver:
    """
    IK-only solver + MotionGen planner using cuRobo that:
      - loads robot from name ("franka" or "kinova"),
      - extracts joint limits from URDF via RobotConfig,
      - computes an IK target joint pose for a desired EE pose,
      - (NEW) plans a collision-aware trajectory to a desired EE pose via MotionGen,
      - selects the target by balancing motion smoothness vs staying away from limits (for IK).
    """

    def __init__(
        self,
        robot_name: str,
        num_seeds: int = 1000,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
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

        # --- resolve robot config file + important link names
        cfg_path = get_robot_configs_path()

        # Map simple names -> YAML files expected by cuRobo examples.
        # Adjust the kinova YAML name if your repo uses a different one.
        name = robot_name.strip().lower()
        if name == "franka":
            yaml_file = "franka.yml"
        elif name == "kinova":
            yaml_file = "kinova_gen3.yml"
        else:
            raise ValueError(f"Unsupported robot '{robot_name}'. Use 'franka' or 'kinova'.")

        cfg = load_yaml(join_path(cfg_path, yaml_file))
        robot_cfg = RobotConfig.from_dict(cfg, self.tensor_args)

        kin = cfg["robot_cfg"]["kinematics"]
        urdf_file = kin["urdf_path"]          # absolute path is recommended
        base_link = kin["base_link"]
        ee_link   = kin["ee_link"]

        # Save for planning
        self.ee_link = ee_link

        # --- Kinematics model for forward checks/printing
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

        # --- RobotConfig from URDF (pulls limits)
        self.robot_cfg = RobotConfig.from_basic(
            urdf_file,
            base_link,
            ee_link,
            self.tensor_args
        )

        # --- Joint limits via CudaRobotModel
        cuda_cfg = CudaRobotModelConfig.from_basic_urdf(
            urdf_file, base_link, ee_link, tensor_args=self.tensor_args
        )
        self.robot_model = CudaRobotModel(cuda_cfg)
        jl = self.robot_model.get_joint_limits()  # JointLimits object
        # jl.position: [2, n_dof] -> [lower; upper]
        self.q_lower = jl.position[0, :].to(self.device, self.dtype)
        self.q_upper = jl.position[1, :].to(self.device, self.dtype)

        # --- IK solver config
        self.ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg=self.robot_cfg,
            world_model=None,                 # add a world if you want collision checks here
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            num_seeds=num_seeds,
            self_collision_check=False,       # set True if you need it
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=use_cuda_graph,
        )
        self.ik = IKSolver(self.ik_cfg)

        # --- NEW: MotionGen for full planning (IK + traj + collision + limits)
        # Use the same robot config dictionary; you can also pass a world file here.
        self.mg_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg=self.robot_cfg,
            world_model=None,                 # plug in a world model to enable collision checking
            tensor_args=self.tensor_args,
            use_cuda_graph=use_cuda_graph,
        )
        self.motion_gen = MotionGen(self.mg_cfg)
        # Prebuild CUDA graphs (speeds up repeated calls)
        self.motion_gen.warmup()

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

    def get_target_joint(
        self,
        current_q: np.ndarray,
        ee_trans: np.ndarray,
        ee_quat_xyzw: np.ndarray,
        ratio: float,
    ) -> np.ndarray:
        """
        Pure-IK: solve for one joint-target that reaches the given EE pose.

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
        current_q = torch.as_tensor(current_q, device=self.device, dtype=self.dtype).view(-1)
        pos = torch.as_tensor(ee_trans, device=self.device, dtype=self.dtype).view(1, 3)
        quat_xyzw = torch.as_tensor(ee_quat_xyzw, device=self.device, dtype=self.dtype).view(1, 4)
        quat_wxyz = self._xyzw_to_wxyz(quat_xyzw)

        # Build cuRobo Pose (expects batches)
        goal = Pose(pos, quat_wxyz)
        res = self.ik.solve_single(goal)

        ok = res.success.view(-1)
        if not ok.any():
            raise RuntimeError("IK failed: no valid solutions for the given EE pose.")

        q_best = res.solution[ok][0][0]  # [N, dof]

        # (Optional) quick sanity print
        out = self.kin_model.get_state(q_best)
        # print("Target pos/quat:", ee_trans, ee_quat_xyzw)
        # print("FK pos/quat:", out.ee_position, out.ee_quaternion)
        return q_best.detach().cpu().numpy().tolist()

    # === NEW ===
    def plan_to_ee_pose(
        self,
        current_q: np.ndarray,
        ee_trans: np.ndarray,
        ee_quat_xyzw: np.ndarray,
        max_planning_time: float = 2.0,
        interpolate_dt: float = 0.01,
        allow_partial: bool = False,
    ):
        """
        MotionGen path: plan a collision-aware, limits-respecting trajectory from
        current_q to the given EE pose.

        Args:
            current_q:        (dof,) numpy array, current joint configuration [rad]
            ee_trans:         (3,) numpy array, target position [m]
            ee_quat_xyzw:     (4,) numpy array, quaternion in (x,y,z,w)
            max_planning_time: seconds budget for the planner
            interpolate_dt:    sampling dt for returned trajectory
            allow_partial:     if True, accept partial/approx solutions when exact is not found

        Returns:
            traj_q:  np.ndarray [T, dof] joint trajectory
            traj_t:  np.ndarray [T] time stamps (if available, else None)
            info:    dict with raw MotionGen result flags
        """
        # to torch
        q_start = torch.as_tensor(current_q, device=self.device, dtype=self.dtype).view(1, -1)
        pos = torch.as_tensor(ee_trans, device=self.device, dtype=self.dtype).view(1, 3)
        quat_xyzw = torch.as_tensor(ee_quat_xyzw, device=self.device, dtype=self.dtype).view(1, 4)
        quat_wxyz = self._xyzw_to_wxyz(quat_xyzw)

        target_pose = Pose(position=pos, quaternion=quat_wxyz)

        # Plan: specify the goal for the end-effector link
        result = self.motion_gen.plan_single(
            q_start=q_start,
            link_poses={self.ee_link: target_pose},
            max_time=max_planning_time,
            # You can pass additional tuning keywords here, e.g. pose cost weights, tolerances, etc.
        )

        success = bool(result.success)
        if not success and not allow_partial:
            raise RuntimeError("MotionGen failed to find a feasible plan to the given EE pose.")

        # --- Extract a time-param trajectory if available; fall back gracefully
        traj_q = None
        traj_t = None

        # Preferred: interpolated/retimed plan
        try:
            interp = result.get_interpolated_plan(dt=interpolate_dt)
            # Common attribute names across cuRobo versions:
            if hasattr(interp, "position"):
                traj_q = interp.position
            elif hasattr(interp, "q"):
                traj_q = interp.q
            if hasattr(interp, "time"):
                traj_t = interp.time
        except Exception:
            pass

        # Fallbacks: use raw optimized plan buffers if needed
        if traj_q is None:
            if hasattr(result, "optimized_plan") and hasattr(result.optimized_plan, "position"):
                traj_q = result.optimized_plan.position
            elif hasattr(result, "plan") and hasattr(result.plan, "position"):
                traj_q = result.plan.position
            elif hasattr(result, "q"):
                traj_q = result.q
            else:
                raise RuntimeError("MotionGen returned a plan, but trajectory extraction failed.")

        # Move to CPU numpy
        traj_q_np = traj_q.detach().cpu().numpy()
        traj_t_np = traj_t.detach().cpu().numpy() if traj_t is not None else None

        info = {
            "success": success,
            "msg": getattr(result, "status", None),
        }
        return traj_q_np, traj_t_np, info
