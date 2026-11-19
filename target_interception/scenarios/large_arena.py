"""Simulation-only large, dense interception arena.

1.5x the area of the mockup room (each side x sqrt(1.5), 4.4 x 3.6 m -> 5.39 x 4.41 m,
same centre and heights), with about twice the obstacles in the curriculum. It has its
own world-frame ID, so its pools and checkpoints can't be mixed up with the room's
(world_frame.py).
"""

from __future__ import annotations

LARGE_ARENA_FRAME_ID = "sim_large_arena_x1p5_20261007"
LARGE_ARENA_MIN_M = (-2.694, -2.205, 0.05)
LARGE_ARENA_MAX_M = (2.694, 2.205, 1.90)

# 11 + 5 fills the policy's 16 obstacle observation slots exactly.
LARGE_ARENA_MAX_STATIC_OBSTACLES = 11
LARGE_ARENA_MAX_DYNAMIC_OBSTACLES = 5

# Default counts doubled and capped at 11 static / 5 moving; phase 4 is
# (10,5), (11,5), (11,4).
LARGE_ARENA_PHASE_OBSTACLE_COUNTS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((2, 2), (4, 2), (6, 2)),
    ((6, 4), (6, 5), (8, 4), (8, 5), (10, 4), (10, 5)),
    ((10, 5), (11, 5), (11, 4)),
)
# Rows per curriculum phase of the thesis pools: the four clutter phases (M3 certificate,
# M5) and the two obstacle-free phases of the nominal policy.
LARGE_ARENA_POOL_SIZES: tuple[tuple[int, int], ...] = ((1, 2000), (2, 5000), (3, 10000), (4, 25000))
NOMINAL_POOL_SIZES: tuple[tuple[int, int], ...] = ((1, 5000), (2, 25000))


# Evaluation clutter densities (static, moving) for the light / medium / dense levels:
# 5, 10 and 15 obstacles, evenly spaced. 6+4 and 10+5 are training counts; 3+2 lies
# within the phase-2 range but is not sampled exactly.
LARGE_ARENA_EVALUATION_CLUTTER: tuple[tuple[int, int], ...] = ((3, 2), (6, 4), (10, 5))

# The large arena is the in-distribution benchmark, the room the out-of-distribution one.
LARGE_ARENA_SUITE_NAME = "thesis_eval_large_final_id"
ROOM_SUITE_NAME = "thesis_eval_room_final_ood"
# Every large-arena task is evaluated out of distribution on the room task (rewards are
# not scored, so the penalty and nominal variants map to the same task).
ROOM_TASK_FOR_LARGE_TASK = {
    "TargetInterception-State-Large-v0": "TargetInterception-State-v0",
    "TargetInterception-State-Penalty-Large-v0": "TargetInterception-State-v0",
    "TargetInterception-Nominal-Large-v0": "TargetInterception-State-v0",
}


def apply_large_arena(cfg) -> None:
    """Switch an interception env or offline cfg to the large dense arena."""
    cfg.world_frame_id = LARGE_ARENA_FRAME_ID
    cfg.arena_min = LARGE_ARENA_MIN_M
    cfg.arena_max = LARGE_ARENA_MAX_M
    cfg.max_static_obstacles = LARGE_ARENA_MAX_STATIC_OBSTACLES
    cfg.max_dynamic_obstacles = LARGE_ARENA_MAX_DYNAMIC_OBSTACLES
    cfg.scenario.phase_obstacle_counts = LARGE_ARENA_PHASE_OBSTACLE_COUNTS
    cfg.scenario.pool_size_by_phase = LARGE_ARENA_POOL_SIZES
