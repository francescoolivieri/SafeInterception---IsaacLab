"""Interception rewards and DG-PPO costs.

M4 penalizes each safety violation. The optional closing-speed factor scales
the capture bonus."""
from __future__ import annotations

import torch


def capture_closing_speed_factor(
    pursuer_pos: torch.Tensor,
    pursuer_vel: torch.Tensor,
    evader_pos: torch.Tensor,
    evader_vel: torch.Tensor,
    *,
    safe_speed: float,
    alpha: float,
) -> torch.Tensor:
    """Paper Eq. (10): discount a capture above the line-of-sight closing speed."""
    delta_pos = evader_pos - pursuer_pos
    line_of_sight = delta_pos / torch.linalg.vector_norm(delta_pos, dim=-1, keepdim=True).clamp_min(1e-6)
    closing_speed = -torch.sum(line_of_sight * (evader_vel - pursuer_vel), dim=-1)
    return torch.exp(-alpha * torch.clamp_min(closing_speed - safe_speed, 0.0))


def compute_rewards(env) -> torch.Tensor:
    env_origins = env._terrain.env_origins
    pos_local = env._robot.data.root_pos_w - env_origins

    # Progress toward the evader. Entering the capture radius is rewarded only
    # by the one-shot success event below.
    pos_error_now = torch.norm(env._reference_pos - pos_local, dim=-1)
    pos_error_old = env._last_position_error
    progress_reward = env.cfg.reward_pos * (pos_error_old - pos_error_now)
    # Clone: the += below would otherwise also change the logged "pos" component.
    rewards = progress_reward.clone()
    components: dict[str, torch.Tensor] = {"pos": progress_reward}

    # Body-rate shaping.
    ang_vel_b = env._robot.data.root_ang_vel_b
    roll_pitch = torch.sum(ang_vel_b[:, :2] ** 2, dim=-1)
    yaw = ang_vel_b[:, 2] ** 2
    body_roll_pitch_penalty = -env.cfg.reward_body_rates_roll_pitch * roll_pitch
    body_yaw_penalty = -env.cfg.reward_body_rates_yaw * yaw
    rewards += body_roll_pitch_penalty + body_yaw_penalty
    components["body_rates"] = body_roll_pitch_penalty + body_yaw_penalty

    smoothness_penalty = float(env.cfg.reward_action_smoothness) * torch.sum(env._action_diff**2, dim=-1)
    rewards -= smoothness_penalty
    components["action_smoothness"] = -smoothness_penalty

    # Safety penalties for penalized PPO. DG-PPO uses the separate signed
    # costs below instead of mixing constraint violations into its reward.
    if env.cfg.enable_collision_penalty:
        altitude_limit, xy_limit = env._arena_limit_masks(pos_local)
        collision = env._collision_mask(pos_local)
        w_safe = float(env.cfg.penalty_safety)
        altitude_pen = w_safe * altitude_limit.to(rewards.dtype)
        xy_boundary_pen = w_safe * xy_limit.to(rewards.dtype)
        collision_pen = w_safe * collision.to(rewards.dtype)
        rewards -= altitude_pen + xy_boundary_pen + collision_pen
        components["altitude_limit"] = -altitude_pen
        components["xy_boundary"] = -xy_boundary_pen
        components["collision"] = -collision_pen
    else:
        zeros = torch.zeros_like(rewards)
        components["altitude_limit"] = zeros
        components["xy_boundary"] = zeros
        components["collision"] = zeros

    success_reward = float(env.cfg.reward_success) * env._last_success_event.to(rewards.dtype)
    if env.cfg.enable_closing_speed_capture_penalty:
        factor = env._dgppo_transition["capture_closing_speed_factor"]
        success_reward *= torch.where(env._last_success_event, factor, torch.ones_like(factor))
    timeout_penalty = float(env.cfg.penalty_timeout) * env.reset_time_outs.to(rewards.dtype)
    rewards += success_reward - timeout_penalty
    components["success"] = success_reward
    components["timeout"] = -timeout_penalty

    env._last_reward_components = components
    env._last_position_error = pos_error_now.detach().clone()
    return rewards


def compute_dgppo_reward(env, *, actions: torch.Tensor) -> torch.Tensor:
    """Compute the DG-PPO objective from stored rollout data.

    Progress uses ground-truth errors at t and t+1, as in the PPO reward. The
    measured graph lags the simulator, so mixing a measured pre-step error with a
    true post-step error would add a non-telescoping sensor-lag term.
    """
    n_envs = env.num_envs
    dtype = torch.float32
    actions = torch.as_tensor(actions, device=env.device, dtype=dtype)
    if actions.ndim == 0 or actions.shape[0] != n_envs:
        raise ValueError(f"DG-PPO reward got action shape {tuple(actions.shape)} for {n_envs} environments.")

    transition = env._dgppo_transition
    body_rates = transition["body_rates"].to(device=env.device, dtype=dtype).reshape(n_envs, 3)
    action_diff = transition["action_diff"].to(device=env.device, dtype=dtype).reshape(n_envs, -1)
    success_event = transition["success_event"].to(device=env.device, dtype=torch.bool).reshape(n_envs)
    timeout = transition["timeout"].to(device=env.device, dtype=torch.bool).reshape(n_envs)
    pos_error_old = transition["position_error_prev"].to(device=env.device, dtype=dtype).reshape(n_envs)
    pos_error_now = transition["position_error"].to(device=env.device, dtype=dtype).reshape(n_envs)

    progress_reward = float(env.cfg.reward_pos) * (pos_error_old - pos_error_now)
    reward = progress_reward

    roll_pitch = torch.sum(body_rates[:, :2] ** 2, dim=-1)
    yaw = body_rates[:, 2] ** 2
    reward -= float(env.cfg.reward_body_rates_roll_pitch) * roll_pitch
    reward -= float(env.cfg.reward_body_rates_yaw) * yaw

    reward -= float(env.cfg.reward_action_smoothness) * torch.sum(action_diff**2, dim=-1)

    success_reward = float(env.cfg.reward_success) * success_event.to(reward.dtype)
    if env.cfg.enable_closing_speed_capture_penalty:
        factor = transition["capture_closing_speed_factor"].to(device=env.device, dtype=reward.dtype).reshape(n_envs)
        success_reward *= torch.where(success_event, factor, torch.ones_like(factor))
    reward += success_reward
    reward -= float(env.cfg.penalty_timeout) * timeout.to(reward.dtype)
    return reward


def shape_dgppo_costs(constraints: torch.Tensor, eps: float = 0.5) -> torch.Tensor:
    """DG-PPO cost shaping of raw signed constraints (<= 0 safe).

    Samples are pushed eps away from zero on their own side, then clipped to [-1, 1],
    as in the DG-PPO reference; the margin separates safe and unsafe samples for the
    constraint critic.
    """
    if not 0.0 <= eps < 1.0:
        raise ValueError(f"DG-PPO cost margin must lie in [0, 1), got {eps}")
    shifted = torch.where(constraints <= 0.0, constraints - eps, constraints + eps)
    return torch.clamp(shifted, min=-1.0, max=1.0)
