"""Base environment that adds the deployment disturbances around an unspecified vehicle.

It owns everything between the policy and the plant:
  - action interface: normalised [vx, vy, vz, yaw_rate] scaled to a physical command,
  - command delay: the previous command is held for command_delay_s into each step,
  - measurement model: delayed, biased and noisy state for the observations.

The vehicle plant itself (asset, motors, physical randomisation, low-level controller)
is not part of this project. A subclass (or a patch of this class) provides it through
cfg.robot and the two vehicle hooks at the bottom.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .measurement_model import MeasurementModel
from .perturbed_env_cfg import PerturbedBaseEnvCfg

PLANT_MISSING = (
    "The vehicle dynamics are not included in this project. Provide a plant by implementing "
    "PerturbedBaseEnv._apply_vehicle_command and _reset_vehicle, and set cfg.robot."
)


class PerturbedBaseEnv(DirectRLEnv):
    cfg: PerturbedBaseEnvCfg

    def __init__(self, cfg: PerturbedBaseEnvCfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        n, dev = self.num_envs, self.device

        # Policy action a_k (clipped), a_{k-1} and a_k - a_{k-1}.
        self._actions = torch.zeros(n, 4, device=dev)
        self._prev_actions = torch.zeros_like(self._actions)
        self._action_diff = torch.zeros_like(self._actions)
        # Physical command u_k = scale * a_k, and u_{k-1} for the delay.
        self._command = torch.zeros_like(self._actions)
        self._prev_command = torch.zeros_like(self._actions)
        self._command_scale = torch.tensor(
            (*cfg.velocity_command_scale_m_s, cfg.velocity_yaw_rate_scale_rad_s), device=dev
        )
        self._command_delay_steps = max(0, int(round(cfg.command_delay_s / cfg.sim.dt)))
        self._physics_substep = 0

        self._measurement_model = MeasurementModel(
            num_envs=n, device=dev, sim_dt=cfg.sim.dt, policy_dt=1.0 / cfg.policy_rate_hz, cfg=cfg.measurement_model
        )
        self._observed_quat_cache = torch.zeros(n, 4, device=dev)
        self._observed_quat_cache[:, 0] = 1.0
        self._observed_position_local_cache = torch.zeros(n, 3, device=dev)
        self._observed_velocity_world_cache = torch.zeros(n, 3, device=dev)
        self._observed_body_rate_cache = torch.zeros(n, 3, device=dev)
        # The post-step capture in _get_observations is also the first sample of the next
        # step, so the first _apply_action of a step skips it (uniform 500 Hz history).
        self._observation_captured_after_step = False

    @property
    def velocity_command_scale(self) -> torch.Tensor:
        """Physical [vx, vy, vz, yaw_rate] scale of a unit action."""
        return self._command_scale

    def _setup_scene(self) -> None:
        if self.cfg.robot is None:
            raise NotImplementedError(PLANT_MISSING)
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_command.copy_(self._command)
        self._actions.copy_(actions).clamp_(-1.0, 1.0)
        self._action_diff.copy_(self._actions - self._prev_actions)
        self._prev_actions.copy_(self._actions)
        self._command.copy_(self._actions * self._command_scale)
        self._physics_substep = 0

    def _apply_action(self) -> None:
        if self._observation_captured_after_step:
            self._observation_captured_after_step = False
        else:
            self._capture_observation_state()
        delayed = self._physics_substep < self._command_delay_steps
        self._physics_substep += 1
        self._apply_vehicle_command(self._prev_command if delayed else self._command)

    def _capture_observation_state(self) -> None:
        data = self._robot.data
        self._measurement_model.capture(data.root_pos_w, data.root_quat_w, data.root_com_lin_vel_w, data.root_ang_vel_b)

    def _refresh_observation_state(self) -> None:
        quat, position_local, velocity_world, body_rate = self._measurement_model.sample(self._terrain.env_origins)
        self._observed_quat_cache.copy_(quat)
        self._observed_position_local_cache.copy_(position_local)
        self._observed_velocity_world_cache.copy_(velocity_world)
        self._observed_body_rate_cache.copy_(body_rate)

    def _reset_actions(self, env_ids: torch.Tensor) -> None:
        """Zero the action and command history of the reset rows (hover command)."""
        for buffer in (self._actions, self._prev_actions, self._action_diff, self._command, self._prev_command):
            buffer[env_ids] = 0.0

    # Vehicle hooks: the plant is not part of this project.

    def _apply_vehicle_command(self, command: torch.Tensor) -> None:
        """Turn the world-frame command [vx, vy, vz, yaw_rate] into forces on self._robot.

        Called once per physics step (500 Hz). The plant includes the velocity and attitude
        loops, the motor model and its per-episode randomisation.
        """
        raise NotImplementedError(PLANT_MISSING)

    def _reset_vehicle(self, env_ids: torch.Tensor) -> None:
        """Reset the plant state of env_ids (motors at hover, new physical randomisation)."""
        raise NotImplementedError(PLANT_MISSING)
