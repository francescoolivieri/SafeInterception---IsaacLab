"""Quadrotor interception with a kinematic target and cylindrical obstacles."""
from __future__ import annotations


import isaaclab.sim as sim_utils
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.views import XformPrimView
from isaaclab.utils import math as math_utils

from target_interception.scenarios import config as scenario_config
from target_interception.scenarios import generation
from target_interception.scenarios import reference_spline
from target_interception.sim.perturbed_env import PerturbedBaseEnv

from . import curriculum, observations, rewards, scene, target
from .control import arena_limit_masks
from .perception import AlignedEntityObservationModel
from .target_interception_env_cfg import TargetInterceptionEnvCfg

EPISODE_STATUS_LABELS = {
    0: "running",
    1: "capture",
    2: "low_altitude",
    3: "boundary",
    4: "timeout",
    5: "invalid_state",
    6: "collision",
}
REASON_RUNNING = 0
REASON_CAPTURE = 1
REASON_ALTITUDE = 2
REASON_BOUNDARY = 3
REASON_TIMEOUT = 4
REASON_INVALID = 5
REASON_COLLISION = 6


class TargetInterceptionEnv(PerturbedBaseEnv):
    """A physical interceptor following a kinematic target through clutter."""

    cfg: TargetInterceptionEnvCfg
    EPISODE_STATUS_MAP = EPISODE_STATUS_LABELS

    def __init__(self, cfg: TargetInterceptionEnvCfg, **kwargs) -> None:
        valid_obstacle_modes = {"pillars", "none"}
        if cfg.obstacle_observation_mode not in valid_obstacle_modes:
            raise ValueError(
                f"Unsupported obstacle_observation_mode '{cfg.obstacle_observation_mode}'. "
                f"Expected one of {sorted(valid_obstacle_modes)}."
            )
        self._prepare_initial_obstacle_slots(cfg)
        self._validate_obstacle_observation_capacity(cfg)

        self.obs_layout = observations.build_obs_layout(cfg)
        cfg.observation_space = self.obs_layout.obs_dim
        cfg.state_space = cfg.observation_space

        self._ref_markers: VisualizationMarkers | None = None
        super().__init__(cfg, **kwargs)

        self._max_static_obstacles = max(1, int(self.cfg.max_static_obstacles)) if self.cfg.enable_pillars else 0
        self._max_dynamic_obstacles = max(0, int(self.cfg.max_dynamic_obstacles))
        self._spawned_static_obstacles = (
            min(max(0, int(self.cfg.spawn_static_obstacles)), max(0, int(self.cfg.max_static_obstacles)))
            if self.cfg.enable_pillars
            else 0
        )
        self._spawned_dynamic_obstacles = min(max(0, int(self.cfg.spawn_dynamic_obstacles)), self._max_dynamic_obstacles)
        self._static_obstacle_view: XformPrimView | None = None
        self._dynamic_obstacle_view: XformPrimView | None = None
        self._static_obstacle_view_index = torch.full(
            (self.num_envs, self._max_static_obstacles), -1, device=self.device, dtype=torch.long
        )
        self._dynamic_obstacle_view_index = torch.full(
            (self.num_envs, self._max_dynamic_obstacles), -1, device=self.device, dtype=torch.long
        )
        self._setup_obstacle_views()

        # Obstacle prims are only visual (collisions and observations use the state
        # tensors), so their poses are written only when something renders them.
        self._render_visuals = self.sim.has_gui()

        self._step_dt = self.sim.cfg.dt * self.cfg.decimation
        self._target_policy_step_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_policy_step_vel = torch.zeros_like(self._target_policy_step_pos)
        self._target_kinematic_step_vel = torch.zeros_like(self._target_policy_step_pos)
        self._target_kinematic_step_acc = torch.zeros_like(self._target_policy_step_pos)
        self._target_kinematic_step_jerk = torch.zeros_like(self._target_policy_step_pos)
        self._target_command_acc = torch.zeros_like(self._target_policy_step_pos)
        self._target_kinematic_substep = 0

        self._arena_min = torch.tensor(self.cfg.arena_min, device=self.device, dtype=torch.float32)
        self._arena_max = torch.tensor(self.cfg.arena_max, device=self.device, dtype=torch.float32)
        margin = torch.tensor(self.cfg.arena_margin, device=self.device, dtype=torch.float32)
        self._arena_min_safe = self._arena_min + margin
        self._arena_max_safe = self._arena_max - margin
        evaluation_margin_value = self.cfg.evaluation_arena_margin
        if evaluation_margin_value is None:
            evaluation_margin_value = self.cfg.arena_margin
        evaluation_margin = torch.tensor(
            float(evaluation_margin_value), device=self.device, dtype=torch.float32
        )
        self._arena_min_evaluation = self._arena_min + evaluation_margin
        self._arena_max_evaluation = self._arena_max - evaluation_margin
        self._outer_limit_min = self._arena_min.clone()
        self._outer_limit_max = self._arena_max.clone()
        altitude_margin = float(self.cfg.altitude_outer_margin)
        self._outer_limit_min[2] -= altitude_margin
        self._outer_limit_max[2] += altitude_margin
        if self.cfg.enable_walls:
            wall = float(self.cfg.wall_extra_margin + self.cfg.wall_thickness)
            self._outer_limit_min[:2] -= wall
            self._outer_limit_max[:2] += wall

        self._pillar_collision_radius = float(self.cfg.pillar_radius + self.cfg.drone_collision_radius)
        self._pillar_top_z = float(self.cfg.arena_min[2] + self.cfg.pillar_height)

        # Obstacles start parked/inactive; the curriculum fills them on the first reset.
        self._static_obstacle_positions_xy = torch.zeros(self.num_envs, self._max_static_obstacles, 2, device=self.device)
        self._static_obstacle_active = torch.zeros(
            self.num_envs, self._max_static_obstacles, dtype=torch.bool, device=self.device
        )

        # Target-command / measured-state and dynamic-obstacle buffers.
        self._path_steps = int(self.max_episode_length) + 1
        self._path_waypoint_steps = torch.as_tensor(
            reference_spline.knot_steps(self.cfg, self._step_dt, max(0, self._path_steps - 1)),
            device=self.device, dtype=torch.long,
        )
        self._path_waypoint_count = len(self._path_waypoint_steps)
        self._target_waypoints = torch.zeros(self.num_envs, self._path_waypoint_count, 3, device=self.device)
        self._target_command_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_command_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._interceptor_start_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._scenario_phase = torch.ones(self.num_envs, dtype=torch.int64, device=self.device)
        # Evaluation-suite tier tag per env (0 = training/untagged); see target_interception.scenarios.benchmark_suite.
        self._scenario_tier = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        # Optional deterministic eval driver; when set, replaces curriculum.resample on reset.
        self._eval_driver = None

        self._dynamic_obstacle_waypoints = torch.zeros(
            self.num_envs, self._path_waypoint_count, self._max_dynamic_obstacles, 3, device=self.device
        )
        self._dynamic_obstacle_positions = torch.zeros(self.num_envs, self._max_dynamic_obstacles, 3, device=self.device)
        self._dynamic_obstacle_velocities = torch.zeros(self.num_envs, self._max_dynamic_obstacles, 3, device=self.device)
        self._dynamic_obstacle_active = torch.zeros(
            self.num_envs, self._max_dynamic_obstacles, dtype=torch.bool, device=self.device
        )
        self._dynamic_collision_radius = float(self.cfg.pillar_radius + self.cfg.drone_collision_radius)

        self._reference_pos = self._target_pos

        # Target and obstacle channels use the same measurement model and clock as the pursuer.
        observed_obstacles = (
            self._max_static_obstacles + self._max_dynamic_obstacles
            if self.cfg.obstacle_observation_mode == "pillars"
            else 0
        )
        self._entity_observation_model = AlignedEntityObservationModel(
            num_envs=self.num_envs,
            num_entities=1 + observed_obstacles,
            device=self.device,
            sim_dt=float(self.sim.cfg.dt),
            policy_dt=1.0 / float(self.cfg.policy_rate_hz),
            cfg=self.cfg.measurement_model,
        )
        self._observed_target_position_local_cache = torch.zeros_like(self._target_pos)
        self._observed_target_velocity_world_cache = torch.zeros_like(self._target_vel)
        if self.cfg.obstacle_observation_mode == "pillars":
            self._observed_static_obstacle_positions_cache = torch.zeros(
                self.num_envs, self._max_static_obstacles, 3, device=self.device
            )
            self._observed_static_obstacle_velocities_cache = torch.zeros(
                self.num_envs, self._max_static_obstacles, 3, device=self.device
            )
            self._observed_dynamic_obstacle_positions_cache = torch.zeros_like(
                self._dynamic_obstacle_positions
            )
            self._observed_dynamic_obstacle_velocities_cache = torch.zeros_like(
                self._dynamic_obstacle_velocities
            )

        # Training resets draw from a named offline pool. Load it lazily so benchmark
        # evaluation can attach its deterministic suite without requiring a training pool.
        self._scenario_pool = None

        self._success_hold_steps = max(1, int(round(self.cfg.success_hold_time_s / self._step_dt)))
        self._success_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_success_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._last_reward_components: dict[str, torch.Tensor] = {}
        self._last_body_rates = torch.zeros(self.num_envs, 3, device=self.device)
        self._dgppo_transition = {
            "body_rates": torch.zeros_like(self._last_body_rates),
            "action_diff": torch.zeros_like(self._actions),
            "success_event": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "timeout": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "position_error_prev": torch.zeros(self.num_envs, device=self.device),
            "position_error": torch.zeros(self.num_envs, device=self.device),
        }
        self._last_episode_status = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # Pre-reset safety snapshot read by the PPO episode logger. DirectRLEnv
        # auto-resets completed rows before the trainer receives the transition.
        self._last_safety_violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_step_snapshot: dict[str, torch.Tensor] = {}
        self._last_position_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)


        self._setup_visualizers()

    @staticmethod
    def _prepare_initial_obstacle_slots(cfg: TargetInterceptionEnvCfg) -> None:
        """Resolve how many obstacle prim slots to spawn at startup (lazy = phase-1 only)."""
        max_static = max(0, int(cfg.max_static_obstacles)) if cfg.enable_pillars else 0
        max_dynamic = max(0, int(cfg.max_dynamic_obstacles))
        if not bool(cfg.lazy_obstacle_spawning):
            cfg.spawn_static_obstacles = max_static
            cfg.spawn_dynamic_obstacles = max_dynamic
            return
        phase_static, phase_dynamic = scenario_config.phase_obstacle_limits(cfg, 1)
        cfg.spawn_static_obstacles = min(max_static, max(max(0, int(cfg.spawn_static_obstacles)), phase_static))
        cfg.spawn_dynamic_obstacles = min(max_dynamic, max(max(0, int(cfg.spawn_dynamic_obstacles)), phase_dynamic))

    @staticmethod
    def _validate_obstacle_observation_capacity(cfg: TargetInterceptionEnvCfg) -> None:
        """Reject structural obstacle counts that cannot fit the policy observation."""
        if cfg.obstacle_observation_mode != "pillars":
            return
        capacity = observations.obstacle_observation_capacity(cfg)
        static_slots = max(1, int(cfg.max_static_obstacles)) if cfg.enable_pillars else 0
        dynamic_slots = max(0, int(cfg.max_dynamic_obstacles))
        required = static_slots + dynamic_slots
        if required > capacity:
            raise ValueError(
                f"max_static_obstacles + max_dynamic_obstacles requires {required} observation slots, "
                f"but obstacle_observation_capacity is {capacity}"
            )

    # IsaacLab interface

    def _setup_scene(self) -> None:
        super()._setup_scene()  # vehicle, ground and cloned envs
        num_envs = self.scene.cfg.num_envs
        if self.cfg.enable_walls:
            scene.spawn_arena_walls(num_envs, self.cfg)
        if self.cfg.enable_pillars:
            scene.spawn_static_obstacle_slots(num_envs, self.cfg, 0, self.cfg.spawn_static_obstacles)
        scene.spawn_dynamic_obstacle_slots(num_envs, self.cfg, 0, self.cfg.spawn_dynamic_obstacles)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._set_camera_view()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Ground-truth pre-step error, the same pos_error_old the PPO reward uses.
        position_error_prev = self._last_position_error.detach().clone()
        # Scenario motion is task state; the base env owns the action, command delay and measurements.
        target.update_episode_motion(self)
        self._refresh_target_measured_state()
        super()._pre_physics_step(actions)

        self._last_body_rates = self._robot.data.root_ang_vel_b.detach().clone()
        self._dgppo_transition = {
            "body_rates": self._last_body_rates,
            "action_diff": self._action_diff.detach().clone(),
            "position_error_prev": position_error_prev,
        }
        self._update_visualizers()

    def _apply_action(self) -> None:
        target.update_kinematic_command_substep(self)
        self._target_pos.copy_(self._target_command_pos)
        self._target_vel.copy_(self._target_command_vel)
        super()._apply_action()

    def _get_observations(self) -> dict:
        # Record this physics step in the 500 Hz sensor history, then read the measurements.
        self._refresh_target_measured_state()
        self._capture_observation_state()
        self._observation_captured_after_step = True
        self._refresh_observation_state()
        return observations.build_observation(self)

    def _entity_observation_ground_truth(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        target_position_w: torch.Tensor | None = None,
        target_velocity_w: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble target/obstacle world states in a stable entity order."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        origins = self._terrain.env_origins[env_ids]
        if target_position_w is None or target_velocity_w is None:
            runtime_position_w, runtime_velocity_w = self._target_observation_state_w(env_ids)
            target_position_w = runtime_position_w if target_position_w is None else target_position_w
            target_velocity_w = runtime_velocity_w if target_velocity_w is None else target_velocity_w

        positions = [target_position_w.unsqueeze(1)]
        velocities = [target_velocity_w.unsqueeze(1)]
        if self.cfg.obstacle_observation_mode == "pillars":
            if self._max_static_obstacles > 0:
                static_xy = self._static_obstacle_positions_xy[env_ids]
                static_z = static_xy.new_full(
                    (env_ids.numel(), self._max_static_obstacles, 1),
                    float(self.cfg.arena_min[2]) + 0.5 * float(self.cfg.pillar_height),
                )
                positions.append(torch.cat((static_xy, static_z), dim=-1) + origins[:, None, :])
                velocities.append(torch.zeros_like(positions[-1]))
            if self._max_dynamic_obstacles > 0:
                positions.append(self._dynamic_obstacle_positions[env_ids] + origins[:, None, :])
                velocities.append(self._dynamic_obstacle_velocities[env_ids])
        return torch.cat(positions, dim=1), torch.cat(velocities, dim=1)

    def _capture_observation_state(self) -> None:
        super()._capture_observation_state()
        positions, velocities = self._entity_observation_ground_truth()
        self._entity_observation_model.capture(positions, velocities)

    def _refresh_observation_state(self) -> None:
        super()._refresh_observation_state()
        positions, velocities = self._entity_observation_model.sample(self._terrain.env_origins, self._measurement_model)
        self._observed_target_position_local_cache.copy_(positions[:, 0])
        self._observed_target_velocity_world_cache.copy_(velocities[:, 0])
        if self.cfg.obstacle_observation_mode != "pillars":
            return
        cursor = 1
        static_stop = cursor + self._max_static_obstacles
        self._observed_static_obstacle_positions_cache.copy_(positions[:, cursor:static_stop])
        self._observed_static_obstacle_velocities_cache.copy_(velocities[:, cursor:static_stop])
        cursor = static_stop
        self._observed_dynamic_obstacle_positions_cache.copy_(positions[:, cursor:])
        self._observed_dynamic_obstacle_velocities_cache.copy_(velocities[:, cursor:])

    def _get_rewards(self) -> torch.Tensor:
        self._refresh_target_measured_state()
        return rewards.compute_rewards(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_target_measured_state()
        timeout = self.episode_length_buf >= self.max_episode_length
        pos_local = self._clip_agent_root_state_in_sim()

        altitude_limit, xy_limit = self._arena_limit_masks(pos_local)
        collision = self._collision_mask(pos_local)
        outer_boundary = self._outer_boundary_mask(pos_local)
        invalid = ~torch.isfinite(self._robot.data.root_state_w).all(dim=-1)
        success = self._update_success_flags(pos_local)  # interception
        self._last_success = success
        self._last_success_event = success & (self._success_counter == self._success_hold_steps)

        safety_violation = altitude_limit | xy_limit | collision
        self._last_safety_violation = safety_violation.detach().clone()
        terminated = invalid.clone()
        if self.cfg.terminate_on_safety_violation:
            terminated |= safety_violation
        if self.cfg.terminate_on_out_of_boundaries:
            terminated |= outer_boundary
        if self.cfg.terminate_on_success:
            terminated |= success
        truncated = timeout & (~terminated)
        self._last_body_rates = self._robot.data.root_ang_vel_b.detach().clone()
        self._dgppo_transition["body_rates"] = self._last_body_rates
        self._dgppo_transition["success_event"] = self._last_success_event.detach().clone()
        self._dgppo_transition["timeout"] = truncated.detach().clone()
        self._dgppo_transition["position_error"] = torch.linalg.vector_norm(
            self._reference_pos - pos_local, dim=-1
        ).detach()
        if self.cfg.enable_closing_speed_capture_penalty:
            self._dgppo_transition["capture_closing_speed_factor"] = rewards.capture_closing_speed_factor(
                pos_local,
                self._robot.data.root_lin_vel_w,
                self._target_pos,
                self._target_vel,
                safe_speed=self.cfg.closing_speed_safe_mps,
                alpha=self.cfg.closing_speed_alpha,
            ).detach()

        episode_status = torch.full((self.num_envs,), REASON_RUNNING, dtype=torch.int32, device=self.device)
        if self.cfg.terminate_on_success:
            episode_status[terminated & success] = REASON_CAPTURE
        if self.cfg.terminate_on_safety_violation:
            episode_status[terminated & altitude_limit] = REASON_ALTITUDE
            episode_status[terminated & xy_limit] = REASON_BOUNDARY
            episode_status[terminated & collision] = REASON_COLLISION
        if self.cfg.terminate_on_out_of_boundaries:
            episode_status[terminated & outer_boundary] = REASON_BOUNDARY
        episode_status[truncated] = REASON_TIMEOUT
        episode_status[terminated & invalid] = REASON_INVALID
        self._last_episode_status = episode_status
        self._last_step_snapshot = {
            # Must be captured before DirectRLEnv auto-resets completed rows.
            "pncbf_constraints": self.pncbf_constraints().detach().clone(),
            "pos_local": pos_local.detach().clone(),
            "vel_world": self._robot.data.root_lin_vel_w.detach().clone(),
            "ref_pos": self._reference_pos.detach().clone(),
            "evader_vel": self._target_vel.detach().clone(),
            "scenario_tier": self._scenario_tier.detach().clone(),
            "safety_violation": safety_violation.detach().clone(),
            "collision": collision.detach().clone(),
            "minimum_safety_margin": self._minimum_safety_margin(pos_local).detach().clone(),
        }
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._robot.reset(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)

        # Sample new interception scenarios (obstacles, target path, interceptor spawn). At eval an
        # attached deterministic driver walks a fixed suite instead of random pool draws.
        if self._eval_driver is not None:
            self._eval_driver.assign(self, env_ids)
        else:
            self._ensure_scenario_pool_loaded()
            curriculum.resample(self, env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = self._interceptor_start_pos[env_ids] + self._terrain.env_origins[env_ids]
        default_root_state[:, 3:7] = self._spawn_yaw_quat(env_ids)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._clip_agent_root_state_in_sim(env_ids)

        target_position_w, target_velocity_w, *_ = target.eval_target_steps(
            self, self._target_waypoints, env_ids, torch.zeros_like(env_ids)
        )
        target_position_w = target_position_w + self._terrain.env_origins[env_ids]
        body_rate = torch.zeros(env_ids.numel(), 3, device=self.device)
        self._measurement_model.reset(
            env_ids, default_root_state[:, :3], default_root_state[:, 3:7], default_root_state[:, 7:10], body_rate
        )
        positions, velocities = self._entity_observation_ground_truth(
            env_ids, target_position_w=target_position_w, target_velocity_w=target_velocity_w
        )
        self._entity_observation_model.reset(env_ids, positions, velocities, self._measurement_model)

        self._reset_actions(env_ids)
        self._success_counter[env_ids] = 0
        self._last_success[env_ids] = False
        self._last_success_event[env_ids] = False

        # Reset the kinematic reference on the shared scenario clock.
        target.update_episode_motion(self, env_ids)
        self._refresh_target_measured_state(env_ids)
        pos_local = self._robot.data.root_pos_w[env_ids] - self._terrain.env_origins[env_ids]
        self._last_position_error[env_ids] = torch.norm(self._target_pos[env_ids] - pos_local, dim=-1)

        # Plant reset: motors at hover and new physical randomisation (not included).
        self._reset_vehicle(env_ids)

    # Control + geometry helpers

    def _arena_limit_masks(self, pos_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return arena_limit_masks(
            pos_local,
            self._arena_min_evaluation,
            self._arena_max_evaluation,
        )

    def _outer_boundary_mask(self, pos_local: torch.Tensor) -> torch.Tensor:
        below = pos_local < self._outer_limit_min
        above = pos_local > self._outer_limit_max
        return torch.any(below | above, dim=-1)

    def _ensure_scenario_pool_loaded(self) -> None:
        if self._scenario_pool is not None:
            return
        from target_interception.scenarios.pool import load_scenario_pools

        self._scenario_pool = load_scenario_pools(self.cfg, self.device)

    def _collision_mask(self, pos_local: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_pillars or self._max_static_obstacles == 0:
            hit_static = torch.zeros(pos_local.shape[0], dtype=torch.bool, device=self.device)
        else:
            xy = pos_local[:, None, :2]
            dxy = torch.linalg.vector_norm(xy - self._static_obstacle_positions_xy, dim=-1)
            inside_radius = (dxy <= self._pillar_collision_radius) & self._static_obstacle_active
            inside_height = (pos_local[:, 2] >= self.cfg.arena_min[2]) & (pos_local[:, 2] <= self._pillar_top_z)
            hit_static = torch.any(inside_radius, dim=1) & inside_height

        if self._max_dynamic_obstacles == 0:
            return hit_static

        xy = pos_local[:, None, :2]
        dyn_dxy = torch.linalg.vector_norm(xy - self._dynamic_obstacle_positions[:, :, :2], dim=-1)
        dyn_inside = (dyn_dxy <= self._dynamic_collision_radius) & self._dynamic_obstacle_active
        dyn_top_z = float(self.cfg.arena_min[2] + self.cfg.dynamic_obstacle_height)
        dyn_height = (pos_local[:, 2] >= self.cfg.arena_min[2]) & (pos_local[:, 2] <= dyn_top_z)
        return hit_static | (torch.any(dyn_inside, dim=1) & dyn_height)

    def _minimum_safety_margin(self, pos_local: torch.Tensor) -> torch.Tensor:
        """Signed clearance from the drone collision body to the closest obstacle."""
        xy = pos_local[:, :2]
        best = pos_local.new_full((pos_local.shape[0],), float("inf"))

        if self.cfg.enable_pillars and self._max_static_obstacles > 0:
            distance = torch.linalg.vector_norm(xy[:, None, :] - self._static_obstacle_positions_xy, dim=-1)
            margin = distance - self._pillar_collision_radius
            margin = torch.where(self._static_obstacle_active, margin, torch.full_like(margin, float("inf")))
            best = torch.minimum(best, margin.min(dim=1).values)

        if self._max_dynamic_obstacles > 0:
            distance = torch.linalg.vector_norm(xy[:, None, :] - self._dynamic_obstacle_positions[:, :, :2], dim=-1)
            margin = distance - self._dynamic_collision_radius
            margin = torch.where(self._dynamic_obstacle_active, margin, torch.full_like(margin, float("inf")))
            best = torch.minimum(best, margin.min(dim=1).values)

        if self.cfg.enable_walls:
            wall_margin = torch.stack(
                [
                    xy[:, 0] - self._arena_min[0],
                    self._arena_max[0] - xy[:, 0],
                    xy[:, 1] - self._arena_min[1],
                    self._arena_max[1] - xy[:, 1],
                ],
                dim=-1,
            ) - float(self.cfg.drone_collision_radius)
            best = torch.minimum(best, wall_margin.min(dim=1).values)

        return best

    def _clip_boundary_velocity(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        lower_hit = pos < self._outer_limit_min
        upper_hit = pos > self._outer_limit_max
        vel = torch.where(lower_hit & (vel < 0.0), torch.zeros_like(vel), vel)
        vel = torch.where(upper_hit & (vel > 0.0), torch.zeros_like(vel), vel)
        return vel

    def _update_success_flags(self, pos_local: torch.Tensor) -> torch.Tensor:
        # Interception: measured drone centers stay within the capture radius.
        pos_error = torch.norm(self._reference_pos - pos_local, dim=-1)
        within = pos_error < self.cfg.pos_tolerance
        self._success_counter = torch.where(within, self._success_counter + 1, torch.zeros_like(self._success_counter))
        return self._success_counter >= self._success_hold_steps

    def _spawn_yaw_quat(self, env_ids: torch.Tensor) -> torch.Tensor:
        delta = self._target_waypoints[env_ids, 0] - self._interceptor_start_pos[env_ids]
        yaw = torch.atan2(delta[:, 1], delta[:, 0])
        zeros = torch.zeros_like(yaw)
        return math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    # Obstacle prim views (move sampled obstacles into place each scenario)

    def _setup_obstacle_views(self) -> None:
        self._static_obstacle_view = None
        self._dynamic_obstacle_view = None
        self._static_obstacle_view_index.fill_(-1)
        self._dynamic_obstacle_view_index.fill_(-1)

        if self._spawned_static_obstacles > 0 and self.cfg.enable_pillars:
            self._static_obstacle_view = XformPrimView(
                "/World/envs/env_.*/Pillars/Pillar.*", device=self.device, validate_xform_ops=False
            )
            self._static_obstacle_view_index[:, : self._spawned_static_obstacles] = self._obstacle_view_indices(
                self._static_obstacle_view, "Pillar", self._spawned_static_obstacles
            )

        if self._spawned_dynamic_obstacles > 0:
            self._dynamic_obstacle_view = XformPrimView(
                "/World/envs/env_.*/DynamicObstacles/Dynamic.*", device=self.device, validate_xform_ops=False
            )
            self._dynamic_obstacle_view_index[:, : self._spawned_dynamic_obstacles] = self._obstacle_view_indices(
                self._dynamic_obstacle_view, "Dynamic", self._spawned_dynamic_obstacles
            )

    def _obstacle_view_indices(self, view: XformPrimView, prefix: str, slots: int) -> torch.Tensor:
        indices = torch.full((self.num_envs, slots), -1, dtype=torch.long, device=self.device)
        for idx, path in enumerate(view.prim_paths):
            env_id = -1
            slot_id = -1
            for part in path.split("/"):
                if part.startswith("env_"):
                    env_id = int(part[4:])
                elif part.startswith(prefix) and part[len(prefix) :].isdigit():
                    slot_id = int(part[len(prefix) :])
            if 0 <= env_id < self.num_envs and 0 <= slot_id < slots:
                indices[env_id, slot_id] = idx
        if torch.any(indices < 0):
            raise RuntimeError(f"Obstacle view for {prefix} did not find all {self.num_envs} x {slots} prims.")
        return indices

    def _ensure_obstacle_slots_for_phase(self, phase: int) -> None:
        if not bool(self.cfg.lazy_obstacle_spawning):
            return

        static_need, dynamic_need = scenario_config.phase_obstacle_limits(self.cfg, phase)
        static_need = min(static_need, self._max_static_obstacles)
        dynamic_need = min(dynamic_need, self._max_dynamic_obstacles)
        if static_need <= self._spawned_static_obstacles and dynamic_need <= self._spawned_dynamic_obstacles:
            return

        old_static = self._spawned_static_obstacles
        old_dynamic = self._spawned_dynamic_obstacles
        self._spawned_static_obstacles = max(self._spawned_static_obstacles, static_need)
        self._spawned_dynamic_obstacles = max(self._spawned_dynamic_obstacles, dynamic_need)
        self.cfg.spawn_static_obstacles = self._spawned_static_obstacles
        self.cfg.spawn_dynamic_obstacles = self._spawned_dynamic_obstacles

        num_envs = self.scene.cfg.num_envs
        if self._spawned_static_obstacles > old_static:
            scene.spawn_static_obstacle_slots(num_envs, self.cfg, old_static, self._spawned_static_obstacles)
        if self._spawned_dynamic_obstacles > old_dynamic:
            scene.spawn_dynamic_obstacle_slots(num_envs, self.cfg, old_dynamic, self._spawned_dynamic_obstacles)

        self._setup_obstacle_views()
        all_envs = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._move_static_obstacles(all_envs)
        self._move_dynamic_obstacles(all_envs)

    def _move_static_obstacles(self, env_ids: torch.Tensor) -> None:
        slots = self._spawned_static_obstacles
        if self._static_obstacle_view is None or slots == 0 or not self._render_visuals:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        active = self._static_obstacle_active[env_ids, :slots]
        xy = self._static_obstacle_positions_xy[env_ids, :slots]
        inactive = generation.inactive_obstacle_xy(self, slots).view(1, -1, 2)
        xy = torch.where(active.unsqueeze(-1), xy, inactive.expand_as(xy))
        pos = torch.zeros(env_ids.shape[0], slots, 3, device=self.device)
        pos[:, :, :2] = xy
        pos[:, :, 2] = generation.static_center_z(self)
        pos = pos + self._terrain.env_origins[env_ids].view(-1, 1, 3)
        indices = self._static_obstacle_view_index[env_ids, :slots].reshape(-1)
        self._static_obstacle_view.set_world_poses(pos.reshape(-1, 3), indices=indices.detach().cpu().tolist())

    def _move_dynamic_obstacles(self, env_ids: torch.Tensor) -> None:
        slots = self._spawned_dynamic_obstacles
        if self._dynamic_obstacle_view is None or slots == 0 or not self._render_visuals:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        active = self._dynamic_obstacle_active[env_ids, :slots]
        pos = self._dynamic_obstacle_positions[env_ids, :slots]
        inactive = generation.inactive_dynamic_pos(self, slots).view(1, -1, 3)
        pos = torch.where(active.unsqueeze(-1), pos, inactive.expand_as(pos))
        pos = pos + self._terrain.env_origins[env_ids].view(-1, 1, 3)
        indices = self._dynamic_obstacle_view_index[env_ids, :slots].reshape(-1)
        self._dynamic_obstacle_view.set_world_poses(pos.reshape(-1, 3), indices=indices.detach().cpu().tolist())

    # Visualization helpers

    def _setup_visualizers(self) -> None:
        if not self.cfg.debug_vis:
            return
        ref_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TargetInterception/TargetCommand",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.06,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.1)),
                ),
            },
        )
        self._ref_markers = VisualizationMarkers(ref_cfg)

    def _update_visualizers(self) -> None:
        # Marker prims are purely visual; skip the per-step USD pose writes when
        # nothing renders them (headless, no GUI).
        if not self._render_visuals:
            return
        if not (self.cfg.debug_vis and self._ref_markers is not None):
            return
        self._ref_markers.visualize(translations=self._target_command_pos + self._terrain.env_origins)

    # DG-PPO reads obs_layout and ground-truth costs, then scores the saved transition.

    def compute_dgppo_signals(
        self,
        observations: torch.Tensor | dict,
        actions: torch.Tensor,
        constraint_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reward and shaped costs for the latest state-t transition."""
        reward = rewards.compute_dgppo_reward(self, actions=actions)
        return reward, rewards.shape_dgppo_costs(constraint_values, eps=float(self.cfg.dgppo_cost_margin))

    def get_curriculum_metrics(self) -> dict[str, float]:
        metrics = {"progress": curriculum.progress(self)}
        for phase in range(1, len(self.cfg.curriculum_phase_fractions) + 1):
            metrics[f"phase_{phase}_fraction"] = float((self._scenario_phase == phase).float().mean().item())
        return metrics


    def _refresh_target_measured_state(self, env_ids=None) -> None:
        if env_ids is None:
            self._target_pos.copy_(self._target_command_pos)
            self._target_vel.copy_(self._target_command_vel)
        else:
            self._target_pos[env_ids] = self._target_command_pos[env_ids]
            self._target_vel[env_ids] = self._target_command_vel[env_ids]

    def _target_observation_state_w(self, env_ids):
        return (
            self._target_pos[env_ids] + self._terrain.env_origins[env_ids],
            self._target_vel[env_ids],
        )


    def pncbf_constraints(self) -> torch.Tensor:
        """Per-constraint ground truth [num_envs, allocated obstacles + 6], in metres.

        Every allocated obstacle (static then dynamic, the identity order of
        _obstacle_observation_identity) at pncbf_obstacle_clearance or contact distance,
        then the six arena planes (lower xyz, upper xyz). Inactive obstacles read -1,
        a safe filler that no observation slot shows.
        """
        from target_interception.safety.constraints import (
            arena_constraint_components,
            obstacle_constraint_components,
        )

        position, arena_min, arena_max, obstacles = self._constraint_geometry()
        clearance = self.cfg.pncbf_obstacle_clearance
        if clearance is not None and float(clearance) <= 0.0:
            raise ValueError(f"pncbf_obstacle_clearance must be positive, got {clearance}")
        parts = []
        for centers, active, contact in obstacles:
            if centers.shape[1]:
                distance = contact if clearance is None else float(clearance)
                cost = obstacle_constraint_components(centers - position[:, None, :2], distance)
                parts.append(torch.where(active.to(torch.bool), cost, torch.full_like(cost, -1.0)))
        parts.append(arena_constraint_components(position, arena_min, arena_max))
        return torch.cat(parts, dim=1)

    def dgppo_constraint_values(self) -> torch.Tensor:
        """Ground truth of DG-PPO's V_h head [num_envs, 1]: the worst of the arena and
        every active obstacle, in DG-PPO length units (see _dgppo_constraint_geometry)."""
        from target_interception.safety.constraints import safety_constraint_value

        return safety_constraint_value(*self._dgppo_constraint_geometry())

    def _dgppo_constraint_geometry(self) -> tuple:
        """Constraint geometry with x/y divided by dgppo_length_scale_xy and z by dgppo_length_scale_z.

        Scaling coordinates, arena bounds, pillar centres and contact distances
        divides every signed constraint by its axis length scale, so DG-PPO's
        cost margin and clip act in those units. The defaults (1 m) keep metres.
        """
        position, arena_min, arena_max, obstacles = self._constraint_geometry()
        l_xy = float(self.cfg.dgppo_length_scale_xy)
        l_z = float(self.cfg.dgppo_length_scale_z)
        if l_xy <= 0.0 or l_z <= 0.0:
            raise ValueError(f"DG-PPO length scales must be positive, got xy={l_xy}, z={l_z}")
        if l_xy == 1.0 and l_z == 1.0:
            return position, arena_min, arena_max, obstacles
        scale = position.new_tensor([1.0 / l_xy, 1.0 / l_xy, 1.0 / l_z])
        bounds = [torch.as_tensor(b, device=position.device, dtype=position.dtype) * scale for b in (arena_min, arena_max)]
        obstacles = [(centers / l_xy, active, contact / l_xy) for centers, active, contact in obstacles]
        return position * scale, bounds[0], bounds[1], obstacles

    def _constraint_geometry(self) -> tuple:
        """(position, arena_min, arena_max, obstacles) of the current simulator state."""
        position = self._robot.data.root_pos_w - self._terrain.env_origins
        obstacles = []
        if self.cfg.enable_pillars and self._max_static_obstacles > 0:
            obstacles.append(
                (self._static_obstacle_positions_xy, self._static_obstacle_active, self._pillar_collision_radius)
            )
        if self._max_dynamic_obstacles > 0:
            obstacles.append(
                (
                    self._dynamic_obstacle_positions[..., :2],
                    self._dynamic_obstacle_active,
                    self._dynamic_collision_radius,
                )
            )
        return position, self._arena_min_safe, self._arena_max_safe, obstacles

    def _clip_agent_root_state_in_sim(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        origins = self._terrain.env_origins[env_ids]
        state = self._robot.data.root_state_w[env_ids].clone()
        position = state[:, :3] - origins
        if not self.cfg.enable_clip_states:
            return position

        clipped = torch.clamp(position, min=self._outer_limit_min, max=self._outer_limit_max)
        changed = torch.isfinite(position).all(dim=-1) & torch.any(position != clipped, dim=-1)
        if not changed.any():
            return position

        changed_ids = env_ids[changed]
        pose = state[changed, :7].clone()
        pose[:, :3] = clipped[changed] + self._terrain.env_origins[changed_ids]
        self._robot.write_root_pose_to_sim(pose, changed_ids)

        velocity = state[changed, 7:].clone()
        velocity[:, :3] = self._clip_boundary_velocity(position[changed], velocity[:, :3])
        self._robot.write_root_velocity_to_sim(velocity, changed_ids)
        position[changed] = clipped[changed]
        return position

    def _set_camera_view(self) -> None:
        if not self.cfg.debug_vis:
            return
        from isaacsim.core.utils.viewports import set_camera_view

        origins = self._terrain.env_origins
        # Top view over all env copies.
        arena_min = torch.as_tensor(self.cfg.arena_min, device=origins.device, dtype=origins.dtype)
        arena_max = torch.as_tensor(self.cfg.arena_max, device=origins.device, dtype=origins.dtype)
        env_min = origins.amin(dim=0) + arena_min
        env_max = origins.amax(dim=0) + arena_max
        center = 0.5 * (env_min + env_max)
        extent = env_max - env_min
        height = 1.35 * max(float(extent[0]), float(extent[1]))
        eye = (float(center[0]), float(center[1]), float(center[2]) + height + float(extent[2]))
        target = tuple(float(value) for value in center)
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")

    def get_last_episode_status(self) -> torch.Tensor:
        return self._last_episode_status

    def get_last_safety_violation(self) -> torch.Tensor:
        """Return the latest pre-reset altitude, boundary, or collision mask."""
        return self._last_safety_violation

    def get_last_step_snapshot(self) -> dict[str, torch.Tensor]:
        return self._last_step_snapshot
