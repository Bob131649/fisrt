import numpy as np
import torch
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm

import gym 

from fixed_reset_wrapper import FixedResetWrapper


SUCCESS_RADIUS = 0.5


def build_wrapped_env(env_name, start_mode="cycle", start_noise_scale=0.1, goal_noise_scale=0):
    base_env = gym.make(env_name)
    return FixedResetWrapper(
        base_env,
        env_name=env_name,
        start_mode=start_mode,
        start_noise_scale=start_noise_scale,
        goal_noise_scale=goal_noise_scale,
    )


def get_env_goal(env):
    if "maze2d" in env.spec.id:
        if hasattr(env, "get_target"):
            return np.array(env.get_target()[:2], dtype=np.float32)
        return np.array(getattr(env, "_target", np.zeros(2))[:2], dtype=np.float32)

    base_env = env.unwrapped
    if hasattr(base_env, "target_goal") and base_env.target_goal is not None:
        return np.array(base_env.target_goal[:2], dtype=np.float32)
    if hasattr(base_env, "_goal") and base_env._goal is not None:
        return np.array(base_env._goal[:2], dtype=np.float32)
    if hasattr(base_env, "get_target"):
        return np.array(base_env.get_target()[:2], dtype=np.float32)
    return np.array(getattr(base_env, "_target", np.zeros(2))[:2], dtype=np.float32)


def get_env_start_xy(env):
    base_env = env.unwrapped
    if hasattr(base_env, "get_xy"):
        return np.array(base_env.get_xy()[:2], dtype=np.float32)
    if hasattr(base_env, "sim") and hasattr(base_env.sim, "data"):
        return np.array(base_env.sim.data.qpos[:2], dtype=np.float32)
    if hasattr(base_env, "physics") and hasattr(base_env.physics, "data"):
        return np.array(base_env.physics.data.qpos[:2], dtype=np.float32)
    return np.array([np.nan, np.nan], dtype=np.float32)


def is_success(env_name, reward, obs, goal):
    if "antmaze" in env_name:
        return reward >= 1.0
    goal_distance = np.linalg.norm(obs[:2] - goal[:2])
    if "maze2d" in env_name:
        return goal_distance <= SUCCESS_RADIUS or reward >= 0.95
    return goal_distance <= SUCCESS_RADIUS


def estimate_ope(policy, env, replay_buffer, eval_episodes=50):
    states = []
    for _ in range(eval_episodes):
        state = env.reset()
        states.append(replay_buffer.normalize_state(np.array(state)))
    states = torch.FloatTensor(np.array(states)).to(policy.device)
    return policy.estimate_value(states).mean().item()


def mc_eval_value_alignment(
    policy,
    replay_buffer,
    env_name,
    discount,
    eval_episodes_per_start=5,
    max_episode_steps=None,
    plot=False,
    figure_path=None,
):
    start_states = []
    env = build_wrapped_env(env_name)
    if not env.fixed_starts:
        raise ValueError("Wrapper did not provide any fixed starts for this env.")

    total_episodes = len(env.fixed_starts) * eval_episodes_per_start
    color_list = cm.rainbow(np.linspace(0, 1, total_episodes + 2))
    rollout_limit = max_episode_steps or getattr(env, "_max_episode_steps", 1000)

    start_returns = []
    start_v_preds = []
    start_q_preds = []
    traj_returns = []
    traj_v_preds = []
    traj_q_preds = []
    success_flags = []

    if plot:
        plt.clf()

    for i in range(total_episodes):
        state, done = env.reset(), False
        start_xy = get_env_start_xy(env)
        goal_xy = get_env_goal(env)
        print(
            f"mc eval episode {i}: start={start_xy.tolist()}, "
            f"goal={goal_xy.tolist()}, start_id={getattr(env, 'last_reset_start_idx', None)}"
        )
        states_list = []
        start_states.append(state)
        step_count = 0
        rewards = []
        norm_states = []
        success = False

        while not done and step_count < rollout_limit:
            norm_state = replay_buffer.normalize_state(np.array(state))
            norm_states.append(norm_state)
            action, q1, q2 = policy.select_action(norm_state)
            env_action = replay_buffer.unnormalize_action(action)
            state, reward, done, _ = env.step(env_action)
            rewards.append(reward)
            step_count += 1
            states_list.append(state)
            if is_success(env_name, reward, state, goal_xy):
                success = True

        if rewards:
            returns_to_go = []
            running_return = 0.0
            for reward in reversed(rewards):
                running_return = reward + discount * running_return
                returns_to_go.append(running_return)
            returns_to_go.reverse()

            states_tensor = torch.FloatTensor(np.array(norm_states)).to(policy.device)
            with torch.no_grad():
                v_pred_values = policy.estimate_value(states_tensor).detach().cpu().numpy().reshape(-1)
                action_tensor = policy.target_policy.select_action_tensor_ope(states_tensor)
                q1_pred, q2_pred = policy.critic(states_tensor, action_tensor)
                q_pred_values = policy.get_min_q(q1_pred, q2_pred).detach().cpu().numpy().reshape(-1)

            start_returns.append(float(returns_to_go[0]))
            start_v_preds.append(float(v_pred_values[0]))
            start_q_preds.append(float(q_pred_values[0]))

            if success:
                traj_returns.extend([float(x) for x in returns_to_go])
                traj_v_preds.extend([float(x) for x in v_pred_values])
                traj_q_preds.extend([float(x) for x in q_pred_values])

        success_flags.append(float(success))
        print(
            f"   --- start_mc_return={start_returns[-1] if start_returns else 0.0:.6f}, "
            f"start_v_pred={start_v_preds[-1] if start_v_preds else 0.0:.6f}, "
            f"start_q_pred={start_q_preds[-1] if start_q_preds else 0.0:.6f}, "
            f"success={success}, steps={step_count}"
        )

        if plot and states_list:
            states_list = np.array(states_list)
            plt.scatter(states_list[:, 0], states_list[:, 1], color=color_list[i], alpha=0.1)

    if plot and start_states:
        start_states = np.array(start_states)
        plt.scatter(start_states[:, 0], start_states[:, 1], color="red")
        if figure_path is None:
            figure_path = "./ope_eval_fig"
        plt.savefig(figure_path)

    env.close()

    start_returns = np.array(start_returns, dtype=np.float32)
    start_v_preds = np.array(start_v_preds, dtype=np.float32)
    start_q_preds = np.array(start_q_preds, dtype=np.float32)
    traj_returns = np.array(traj_returns, dtype=np.float32)
    traj_v_preds = np.array(traj_v_preds, dtype=np.float32)
    traj_q_preds = np.array(traj_q_preds, dtype=np.float32)
    success_flags = np.array(success_flags, dtype=np.float32)

    def mse(pred, target):
        if pred.size == 0:
            return float("nan")
        return float(np.mean((pred - target) ** 2))

    def mae(pred, target):
        if pred.size == 0:
            return float("nan")
        return float(np.mean(np.abs(pred - target)))

    info = {
        "MC_Success_Rate": float(np.mean(success_flags)) if success_flags.size > 0 else float("nan"),
        "Start_MC_Return_Mean": float(np.mean(start_returns)) if start_returns.size > 0 else float("nan"),
        "Start_MC_MSE_V": mse(start_v_preds, start_returns),
        "Start_MC_MAE_V": mae(start_v_preds, start_returns),
        "Start_MC_MSE_Q": mse(start_q_preds, start_returns),
        "Start_MC_MAE_Q": mae(start_q_preds, start_returns),
        "Traj_MC_Num_States": float(traj_returns.size),
        "Traj_MC_MSE_V": mse(traj_v_preds, traj_returns),
        "Traj_MC_MAE_V": mae(traj_v_preds, traj_returns),
        "Traj_MC_MSE_Q": mse(traj_q_preds, traj_returns),
        "Traj_MC_MAE_Q": mae(traj_q_preds, traj_returns),
    }
    print("---------------------------------------")
    print(
        "MC evaluation: "
        f"success_rate={info['MC_Success_Rate']:.6f}, "
        f"start_mae_v={info['Start_MC_MAE_V']:.6f}, "
        f"start_mae_q={info['Start_MC_MAE_Q']:.6f}, "
        f"traj_states={info['Traj_MC_Num_States']:.0f}, "
        f"traj_mae_v={info['Traj_MC_MAE_V']:.6f}, "
        f"traj_mae_q={info['Traj_MC_MAE_Q']:.6f}"
    )
    print("---------------------------------------")
    return info
