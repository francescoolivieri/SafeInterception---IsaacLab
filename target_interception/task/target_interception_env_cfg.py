"""Configuration of the target-interception tasks."""
from __future__ import annotations

from typing import Literal

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from target_interception.scenarios.config import ScenarioGenerationCfg
from target_interception.scenarios.large_arena import (
    LARGE_ARENA_FRAME_ID,
    LARGE_ARENA_MAX_DYNAMIC_OBSTACLES,
    LARGE_ARENA_MAX_M,
    LARGE_ARENA_MAX_STATIC_OBSTACLES,
    LARGE_ARENA_MIN_M,
    LARGE_ARENA_PHASE_OBSTACLE_COUNTS,
    LARGE_ARENA_POOL_SIZES,
    NOMINAL_POOL_SIZES,
)
from target_interception.sim.perturbed_env_cfg import PerturbedBaseEnvCfg
from target_interception.world_frame import ARENA_MAX_M, ARENA_MIN_M, WORLD_FRAME_ID


def arena_env_spacing(arena_min, arena_max, wall_extra_margin: float, wall_thickness: float) -> float:
    """Env grid spacing that fits the largest arena side plus both walls."""
    return max((arena_max[axis] - arena_min[axis]) + 2 * wall_extra_margin + 2 * wall_thickness for axis in (0, 1))


@configclass
class TargetInterceptionEnvCfg(PerturbedBaseEnvCfg):
    """Pursuer intercepting a kinematic evader in a cylinder-cluttered arena."""

    observation_space = 0  # resolved from ObsLayout by the environment
    state_space = 0

    # Arena (room frame by default; the Large tasks switch to the training arena).
    world_frame_id: str = WORLD_FRAME_ID
    arena_min = ARENA_MIN_M
    arena_max = ARENA_MAX_M
    arena_margin: float = 0.0
    # None uses arena_margin. Evaluation sets 0.0 so violations are scored at the physical arena.
    evaluation_arena_margin: float | None = None
    # Episodes end (terminate_on_out_of_boundaries) this far outside the floor/ceiling.
    altitude_outer_margin: float = 0.2

    enable_walls: bool = True
    wall_thickness: float = 0.2
    wall_extra_margin: float = 0.0

    # Every obstacle is a vertical cylinder of radius r_y and height 1.9 m.
    enable_pillars: bool = True
    pillar_radius: float = 0.2
    pillar_height: float = 1.9
    drone_collision_radius: float = 0.09

    env_spacing = arena_env_spacing(arena_min, arena_max, wall_extra_margin, wall_thickness)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=env_spacing, replicate_physics=True)

    # "pillars": obstacle block with a fixed number of slots; "none": nominal policy.
    obstacle_observation_mode: Literal["pillars", "none"] = "pillars"
    obstacle_observation_capacity: int = 16

    debug_vis: bool = True

    # Task reward (thesis, Table "Reward coefficients").
    reward_pos: float = 3.7384779440286717
    reward_body_rates_roll_pitch: float = 0.0005172705819670797
    reward_body_rates_yaw: float = 0.004909551246339162
    reward_action_smoothness: float = 0.01
    reward_success: float = 31.852436859999765
    penalty_timeout: float = 10.0
    # Capture bonus scaled by phi = exp(-alpha * max(0, closing_speed - v_safe)).
    # On for the DG-PPO (M5) runs, off for the nominal PPO (M2, M3) and M4.
    enable_closing_speed_capture_penalty: bool = False
    closing_speed_safe_mps: float = 2.0
    closing_speed_alpha: float = 3.0

    # M4 only: w_safe on every step with a violation.
    enable_collision_penalty: bool = False
    penalty_safety: float = 4.06893384906473

    # Capture: centre distance below pos_tolerance (r_cap).
    pos_tolerance: float = 0.15
    success_hold_time_s: float = 0.01
    terminate_on_success: bool = True

    # DG-PPO constraint shaping: x/y and z divided by the length scales, pushed
    # dgppo_cost_margin away from zero on its own side, then clipped.
    dgppo_cost_margin: float = 0.2
    dgppo_length_scale_xy: float = 2.0
    dgppo_length_scale_z: float = 0.72
    # PNCBF obstacle constraint distance; None is physical contact (r_y + r_p).
    pncbf_obstacle_clearance: float | None = None

    terminate_on_safety_violation: bool = False
    terminate_on_out_of_boundaries: bool = False
    # Clamp the pursuer back inside the outer limits (nominal training only).
    enable_clip_states: bool = True

    # Curriculum: train.py fills curriculum_total_steps from the trainer budget.
    curriculum_total_steps: int = 0
    # Phases 1..4: static target, light, medium, dense clutter.
    curriculum_phase_fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
    curriculum_blend_fraction: float = 0.2
    # Spawn only the obstacle prims needed by the phases reached so far.
    lazy_obstacle_spawning: bool = True
    spawn_static_obstacles: int = 0
    spawn_dynamic_obstacles: int = 0
    # Obstacle slots for training (evaluation adopts the suite's own counts).
    max_static_obstacles: int = 6
    max_dynamic_obstacles: int = 3
    dynamic_obstacle_height: float = 1.9
    # Evader waypoint interval (knots of the cubic spline replayed at runtime).
    path_waypoint_dt: float = 0.2
    # Knots are denser over the launch, which is shorter than two coarse knot intervals.
    path_launch_time: float = 0.4
    path_launch_waypoint_dt: float = 0.04

    scenario: ScenarioGenerationCfg = ScenarioGenerationCfg()


@configclass
class TargetInterceptionStateCfg(TargetInterceptionEnvCfg):
    """Room arena (out-of-distribution evaluation)."""

    terminate_on_out_of_boundaries = True
    scenario: ScenarioGenerationCfg = ScenarioGenerationCfg(pool_name="clutter_train_pool_aggressive")


@configclass
class TargetInterceptionStateLargeCfg(TargetInterceptionStateCfg):
    """Large training arena: area x1.5 of the room, up to 11 static + 5 moving cylinders (M3, M5)."""

    world_frame_id = LARGE_ARENA_FRAME_ID
    arena_min = LARGE_ARENA_MIN_M
    arena_max = LARGE_ARENA_MAX_M
    max_static_obstacles = LARGE_ARENA_MAX_STATIC_OBSTACLES
    max_dynamic_obstacles = LARGE_ARENA_MAX_DYNAMIC_OBSTACLES
    scenario: ScenarioGenerationCfg = ScenarioGenerationCfg(
        phase_obstacle_counts=LARGE_ARENA_PHASE_OBSTACLE_COUNTS, pool_name="clutter_train_pool_large",
        pool_size_by_phase=LARGE_ARENA_POOL_SIZES,
    )

    def __post_init__(self):
        self.env_spacing = arena_env_spacing(self.arena_min, self.arena_max, self.wall_extra_margin, self.wall_thickness)
        self.scene.env_spacing = self.env_spacing


@configclass
class TargetInterceptionStatePenaltyLargeCfg(TargetInterceptionStateLargeCfg):
    """Large arena with the safety penalty in the reward (M4)."""

    enable_collision_penalty = True


@configclass
class TargetInterceptionNominalLargeCfg(TargetInterceptionEnvCfg):
    """Obstacle-free large arena for the nominal PPO policy (M2, M3)."""

    world_frame_id = LARGE_ARENA_FRAME_ID
    arena_min = LARGE_ARENA_MIN_M
    arena_max = LARGE_ARENA_MAX_M
    curriculum_phase_fractions = (0.2, 0.8)  # static target, then moving target
    obstacle_observation_mode = "none"
    enable_walls = False
    enable_pillars = False
    max_static_obstacles = 0
    max_dynamic_obstacles = 0
    scenario: ScenarioGenerationCfg = ScenarioGenerationCfg(
        pool_name="clutter_nominal_pool_large", pool_size_by_phase=NOMINAL_POOL_SIZES
    )

    def __post_init__(self):
        self.env_spacing = arena_env_spacing(self.arena_min, self.arena_max, self.wall_extra_margin, self.wall_thickness)
        self.scene.env_spacing = self.env_spacing
