"""Observation model and filter dynamics of the PNCBF certificate."""
from __future__ import annotations

import torch


class PNCBFObservationModel:
    """Full task observation, differentiated under a single-integrator model.

    The command moves the pursuer, target and obstacles keep their measured velocities.
    """

    def __init__(self, obs_layout) -> None:
        obstacle_slice = obs_layout.extras.get("obstacles")
        velocity_slice = obs_layout.extras.get("obstacle_vel")
        if obstacle_slice is None or velocity_slice is None:
            raise ValueError("PNCBF requires obstacle positions and velocities")
        if obs_layout.extras.get("previous_action") is None:
            raise ValueError("PNCBF requires the previous-action observation")
        self.layout = obs_layout
        self.num_obstacles = (obstacle_slice.stop - obstacle_slice.start) // 3
        if velocity_slice.stop - velocity_slice.start != self.num_obstacles * 2:
            raise ValueError("obstacle position and velocity slot counts do not match")
        self.state_dim = max(obs_layout.ego.stop, obs_layout.target.stop, *(s.stop for s in obs_layout.extras.values()))

    def decode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.state_dim:
            raise ValueError(f"PNCBF expects {self.state_dim} observations, got {obs.shape[-1]}")
        # Clone: the vector env may reuse the observation buffer.
        return obs.clone()

    def drift_and_control(self, state: torch.Tensor, command_scale) -> tuple[torch.Tensor, torch.Tensor]:
        """d(obs)/dt = f + G a for the normalized velocity action a."""
        scale = torch.as_tensor(command_scale, device=state.device, dtype=state.dtype).reshape(-1)
        if scale.numel() < 3:
            raise ValueError("command_scale must contain vx, vy, and vz scales")
        drift = torch.zeros_like(state)
        control = state.new_zeros(len(state), self.state_dim, 3)
        ego = state[:, self.layout.ego]
        target = state[:, self.layout.target]
        target_position = slice(self.layout.target.start, self.layout.target.start + 3)
        drift[:, target_position] = target[:, 3:6] + ego[:, 3:6]
        axes = torch.diag(scale[:3]).expand(len(state), -1, -1)
        control[:, self.layout.ego.start:self.layout.ego.start + 3] = axes
        control[:, target_position] = -axes
        obstacles = state[:, self.layout.extras["obstacles"]].reshape(len(state), self.num_obstacles, 3)
        # Padding slots sit at (-10, -10, -10); they must not move.
        valid = ((obstacles + 10.).abs().amax(dim=-1) > 1.).to(state.dtype)
        obstacle_velocity = state.new_zeros(len(state), self.num_obstacles, 3)
        obstacle_velocity[..., :2] = (
            state[:, self.layout.extras["obstacle_vel"]].reshape(len(state), self.num_obstacles, 2)
            + ego[:, None, 3:5]
        )
        drift[:, self.layout.extras["obstacles"]] = (obstacle_velocity * valid[..., None]).reshape(len(state), -1)
        control[:, self.layout.extras["obstacles"]] = (
            -axes[:, None] * valid[..., None, None]
        ).reshape(len(state), -1, 3)
        return drift, control

    def metadata(self) -> dict:
        layout = self.layout
        return {
            "representation": "full_observation_graph_v1",
            "state_dim": self.state_dim,
            "num_obstacles": self.num_obstacles,
            "arena_min": None if layout.arena_min is None else [float(v) for v in layout.arena_min],
            "arena_max": None if layout.arena_max is None else [float(v) for v in layout.arena_max],
            "previous_action_dim": layout.extras["previous_action"].stop - layout.extras["previous_action"].start,
            "ego": [layout.ego.start, layout.ego.stop],
            "target": [layout.target.start, layout.target.stop],
            "extras": {k: [v.start, v.stop] for k, v in layout.extras.items()},
        }
