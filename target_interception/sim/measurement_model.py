"""Measurement model: what the policy sees instead of the true vehicle state.

Thesis, Table "Measurement uncertainty". Every channel is sampled on one 100 Hz
sensor clock whose phase on the 500 Hz physics grid is drawn per episode.

    position    p~ = p(t - age) + white noise                      sigma_p
    velocity    v~ = v(t - age - delay_v) + bias_v + residual_v     residual_v is AR(1), time constant tau_v
    body rate   w~ = w(t - age) + bias_w + white noise              sigma_w
    attitude    R~ = R(t - age)

age (sensor phase), delay_v and both biases are drawn at reset and held for the
episode; the white noise and the AR(1) residual evolve at every policy step.
"""

from __future__ import annotations

import math

import torch


class MeasurementModel:
    """500 Hz state history read back with the sensor phase, delays, biases and noise."""

    def __init__(self, num_envs: int, device: torch.device | str, sim_dt: float, policy_dt: float, cfg) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.sim_dt = float(sim_dt)
        self.cfg = cfg

        self.sensor_period_steps = max(1, int(round(1.0 / float(cfg.sensor_rate_hz) / self.sim_dt)))
        max_velocity_delay_steps = int(math.ceil(float(cfg.velocity_delay_max_s) / self.sim_dt))
        self.history_length = (self.sensor_period_steps - 1) + max_velocity_delay_steps + 3
        self.position_history_w = torch.zeros(self.num_envs, self.history_length, 3, device=self.device)
        self.velocity_history_w = torch.zeros_like(self.position_history_w)
        self.body_rate_history = torch.zeros_like(self.position_history_w)
        self.quaternion_history = torch.zeros(self.num_envs, self.history_length, 4, device=self.device)
        self.quaternion_history[..., 0] = 1.0
        self.write_index = 0
        self._env_ids = torch.arange(self.num_envs, device=self.device)

        # Per-episode draws.
        self.sensor_age_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.velocity_delay_steps = torch.zeros_like(self.sensor_age_steps)
        self.velocity_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self.gyro_bias = torch.zeros_like(self.velocity_bias)
        # Per-step state.
        self.velocity_residual = torch.zeros_like(self.velocity_bias)

        self.position_std = self._vec(cfg.position_white_std_m)
        self.velocity_bias_std = self._vec(cfg.velocity_episode_bias_std_m_s)
        self.velocity_bias_clip = self._vec(cfg.velocity_episode_bias_clip_m_s)
        self.velocity_residual_std = self._vec(cfg.velocity_residual_std_m_s)
        self.gyro_white_std = self._vec(cfg.gyro_white_std_rad_s)
        self.gyro_bias_std = self._vec(cfg.gyro_episode_bias_std_rad_s)
        # AR(1): r_k = rho r_{k-1} + sqrt(1 - rho^2) sigma n_k keeps the stationary std at sigma.
        tau = self._vec(cfg.velocity_residual_time_constant_s).clamp_min(1.0e-6)
        self.velocity_residual_rho = torch.exp(-float(policy_dt) / tau)
        self.velocity_residual_innovation_scale = torch.sqrt((1.0 - self.velocity_residual_rho.square()).clamp_min(0.0))

    def _vec(self, value) -> torch.Tensor:
        return torch.tensor(value, device=self.device, dtype=torch.float32).view(1, 3)

    def reset(
        self,
        env_ids: torch.Tensor,
        position_w: torch.Tensor,
        quaternion_w: torch.Tensor,
        velocity_w: torch.Tensor,
        body_rate_b: torch.Tensor,
    ) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        n = env_ids.numel()
        if n == 0:
            return
        self.position_history_w[env_ids] = position_w.unsqueeze(1)
        self.quaternion_history[env_ids] = quaternion_w.unsqueeze(1)
        self.velocity_history_w[env_ids] = velocity_w.unsqueeze(1)
        self.body_rate_history[env_ids] = body_rate_b.unsqueeze(1)

        if bool(self.cfg.randomize_sensor_phase) and self.sensor_period_steps > 1:
            self.sensor_age_steps[env_ids] = torch.randint(0, self.sensor_period_steps, (n,), device=self.device)
        else:
            self.sensor_age_steps[env_ids] = 0
        delay_s = torch.empty(n, device=self.device).uniform_(
            float(self.cfg.velocity_delay_min_s), float(self.cfg.velocity_delay_max_s)
        )
        self.velocity_delay_steps[env_ids] = torch.round(delay_s / self.sim_dt).to(torch.long)

        bias = torch.randn(n, 3, device=self.device) * self.velocity_bias_std
        self.velocity_bias[env_ids] = torch.maximum(torch.minimum(bias, self.velocity_bias_clip), -self.velocity_bias_clip)
        self.velocity_residual[env_ids] = torch.randn(n, 3, device=self.device) * self.velocity_residual_std
        self.gyro_bias[env_ids] = torch.randn(n, 3, device=self.device) * self.gyro_bias_std

    def capture(
        self,
        position_w: torch.Tensor,
        quaternion_w: torch.Tensor,
        velocity_w: torch.Tensor,
        body_rate_b: torch.Tensor,
    ) -> None:
        """Record the true state of one physics step."""
        self.position_history_w[:, self.write_index] = position_w
        self.quaternion_history[:, self.write_index] = quaternion_w
        self.velocity_history_w[:, self.write_index] = velocity_w
        self.body_rate_history[:, self.write_index] = body_rate_b
        self.write_index = (self.write_index + 1) % self.history_length

    def sample(self, env_origins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the measured (quaternion_w, position_local, velocity_w, body_rate_b)."""
        latest_index = self.write_index - 1
        sensor_indices = (latest_index - self.sensor_age_steps) % self.history_length
        velocity_indices = (sensor_indices - self.velocity_delay_steps) % self.history_length

        position_w = self.position_history_w[self._env_ids, sensor_indices]
        quaternion_w = self.quaternion_history[self._env_ids, sensor_indices]
        body_rate_b = self.body_rate_history[self._env_ids, sensor_indices]
        velocity_w = self.velocity_history_w[self._env_ids, velocity_indices]

        self.velocity_residual.mul_(self.velocity_residual_rho).add_(
            torch.randn_like(self.velocity_residual) * self.velocity_residual_std * self.velocity_residual_innovation_scale
        )
        velocity_w = velocity_w + self.velocity_bias + self.velocity_residual
        position_local = position_w - env_origins + torch.randn_like(position_w) * self.position_std
        body_rate_b = body_rate_b + self.gyro_bias + torch.randn_like(body_rate_b) * self.gyro_white_std
        return quaternion_w, position_local, velocity_w, body_rate_b
