"""PNCBF certificate: one value per obstacle slot and per arena plane."""
from __future__ import annotations

import torch
from torch import nn

from target_interception.nn.mlp import MLP

ARENA_PLANES = 6
INACTIVE_VALUE = -1.0e3
# Input scales in m and m/s; saved checkpoints depend on them.
POSITION_SCALE = 2.0
VELOCITY_SCALE = 2.0


def project_observation(observations, layout):
    """Observation as the single-integrator filter sees it.

    World-frame target and obstacle velocities; ego velocity, rates and previous action
    zeroed, attitude level.
    """
    projected = observations.clone()
    ego = layout.ego.start
    ego_velocity = observations[:, ego + 3:ego + 6]
    target_velocity = slice(layout.target.start + 3, layout.target.start + 6)
    projected[:, target_velocity] = observations[:, target_velocity] + ego_velocity
    positions = observations[:, layout.extras["obstacles"]].reshape(len(observations), -1, 3)
    # Padding slots at (-10, -10, -10) keep zero velocity.
    valid = ((positions + 10.).abs().amax(dim=-1) > 1.).to(observations.dtype)
    relative = observations[:, layout.extras["obstacle_vel"]].reshape(len(observations), -1, 2)
    projected[:, layout.extras["obstacle_vel"]] = (
        (relative + ego_velocity[:, None, :2]) * valid[..., None]
    ).reshape(len(observations), -1)
    projected[:, ego + 3:layout.ego.stop] = 0.
    projected[:, ego + 6:ego + 15] = torch.eye(3).reshape(-1).to(observations)  # level attitude
    projected[:, layout.extras["previous_action"]] = 0.
    return projected


class PNCBFCertificate(nn.Module):
    """V(x) per obstacle slot and arena plane, ordered [slots..., planes...].

    One MLP shared by all slots, one for the six planes (lower xyz, upper xyz).
    """

    def __init__(self, obs_layout, *, hidden=(256, 256, 256), device="cpu"):
        super().__init__()
        self.layout = obs_layout
        self.hidden = tuple(int(size) for size in hidden)
        self.num_slots = (obs_layout.extras["obstacles"].stop - obs_layout.extras["obstacles"].start) // 3
        own = 6 + 6  # pursuer position, velocity; target offset, velocity
        self.slot_head = MLP((*self.hidden, 1), own + 6, act=torch.tanh, act_final=False, use_layernorm=False)
        self.plane_head = MLP((*self.hidden, ARENA_PLANES), own, act=torch.tanh, act_final=False, use_layernorm=False)
        self.to(device)

    def valid(self, observations) -> torch.Tensor:
        """Active constraints, shape [B, heads]; padding slots sit at (-10, -10, -10)."""
        offsets = observations[:, self.layout.extras["obstacles"]].reshape(len(observations), -1, 3)
        slots = (offsets + 10.).abs().amax(dim=-1) > 1.
        return torch.cat((slots, slots.new_ones(len(observations), ARENA_PLANES)), dim=-1)

    def _features(self, projected):
        layout = self.layout
        ego = projected[:, layout.ego]
        target = projected[:, layout.target]
        own = torch.cat((ego[:, :3] / POSITION_SCALE, ego[:, 3:6] / VELOCITY_SCALE,
                         target[:, :3] / POSITION_SCALE, target[:, 3:6] / VELOCITY_SCALE), dim=-1)
        offsets = projected[:, layout.extras["obstacles"]].reshape(len(projected), -1, 3)
        velocity = projected[:, layout.extras["obstacle_vel"]].reshape(len(projected), -1, 2)
        distance = (offsets[..., :2].square().sum(dim=-1, keepdim=True) + 1e-12).sqrt()
        slots = torch.cat((offsets / POSITION_SCALE, velocity / VELOCITY_SCALE, distance / POSITION_SCALE), -1)
        return own, slots

    def forward(self, observations):
        """Raw values, shape [B, heads]. Padding stays unmasked for the regression."""
        own, slots = self._features(project_observation(observations, self.layout))
        slot_in = torch.cat((own[:, None].expand(-1, self.num_slots, -1), slots), dim=-1)
        return torch.cat((self.slot_head(slot_in).squeeze(-1), self.plane_head(own)), dim=-1)

    def value(self, observations):
        """Values with padding slots at INACTIVE_VALUE, so they never bind in the QP."""
        values = self(observations)
        return torch.where(self.valid(observations), values, torch.full_like(values, INACTIVE_VALUE))

    def load_checkpoint_state(self, state: dict) -> None:
        # The shipped certificate names its heads slot_heads.0 and plane_heads.0.
        renamed = {key.replace("slot_heads.0.", "slot_head.").replace("plane_heads.0.", "plane_head."): value
                   for key, value in state.items()}
        self.load_state_dict(renamed)

    def config_dict(self):
        return {"kind": "per_constraint", "encoder": "mlp", "hidden": list(self.hidden),
                "ensemble_size": 1, "inputs": "single_integrator"}
