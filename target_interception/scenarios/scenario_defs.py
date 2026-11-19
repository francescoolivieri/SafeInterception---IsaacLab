"""Evaluation scenario codes and structured challenge layouts.

In-distribution trials use random clutter; OOD trials use walls, narrow passages
and moving obstacles. Coordinates are metres in the room frame."""
from __future__ import annotations

import random
from dataclasses import dataclass

SCENARIO_WALL = 1
SCENARIO_NARROW_PASSAGE = 3
SCENARIO_DYNAMIC_OBSTACLES = 4
SCENARIO_CLUTTER_LIGHT = 30
SCENARIO_CLUTTER_MEDIUM = 31
SCENARIO_CLUTTER_DENSE = 32
SCENARIO_STATIONARY = 33

# In-distribution: random clutter from the training sampler. OOD: structured challenges.
IN_DISTRIBUTION = "in_distribution"
OOD = "ood"
FAMILIES = (IN_DISTRIBUTION, OOD)

# code -> (static, moving) pillars and training curriculum phase.
CLUTTER_LEVELS = {
    SCENARIO_CLUTTER_LIGHT: ((2, 1), 2),
    SCENARIO_CLUTTER_MEDIUM: ((4, 2), 3),
    SCENARIO_CLUTTER_DENSE: ((5, 3), 4),
}
CLUTTER_CODES = tuple(CLUTTER_LEVELS)
CHALLENGE_CODES = (SCENARIO_WALL, SCENARIO_NARROW_PASSAGE, SCENARIO_DYNAMIC_OBSTACLES)
# Families whose target is parked: one row per layout, recorded with speed 0.
STATIC_CHALLENGE_CODES = (SCENARIO_WALL, SCENARIO_NARROW_PASSAGE)
EVALUATION_SCENARIO_CODES = CLUTTER_CODES + CHALLENGE_CODES + (SCENARIO_STATIONARY,)

SCENARIO_LABELS = {
    SCENARIO_CLUTTER_LIGHT: "clutter_light",
    SCENARIO_CLUTTER_MEDIUM: "clutter_medium",
    SCENARIO_CLUTTER_DENSE: "clutter_dense",
    SCENARIO_WALL: "wall",
    SCENARIO_NARROW_PASSAGE: "narrow_passage",
    SCENARIO_DYNAMIC_OBSTACLES: "dynamic_obstacles",
    SCENARIO_STATIONARY: "stationary_evader",
}
SCENARIO_FAMILY = {code: IN_DISTRIBUTION for code in CLUTTER_CODES} | {
    code: OOD for code in CHALLENGE_CODES
} | {SCENARIO_STATIONARY: IN_DISTRIBUTION}
SCENARIO_PHASE = {code: phase for code, (_, phase) in CLUTTER_LEVELS.items()} | {
    SCENARIO_WALL: 3, SCENARIO_NARROW_PASSAGE: 3, SCENARIO_DYNAMIC_OBSTACLES: 4, SCENARIO_STATIONARY: 3,
}

Point = tuple[float, float]
Rail = tuple[Point, Point]


@dataclass(frozen=True)
class ScenarioSpec:
    code: int
    static_xy: tuple[Point, ...]          # pillar centers
    dynamic_rails: tuple[Rail, ...]       # each rail: obstacle oscillates start<->end
    dynamic_phases: tuple[float, ...]     # initial phase along each rail, in [0, 1)
    target_route: tuple[Point, ...]       # the route polyline, re-parameterized by arc length
    interceptor_xy: Point
    target_speed: float
    # Motion pattern per rail: "rail", "wobble" or "stop_and_go".
    dynamic_patterns: tuple[str, ...] = ()
    # Extra moving obstacles that keep this XY offset from the target at every waypoint.
    dynamic_follow_offsets: tuple[Point, ...] = ()


def _mirror_y(spec: ScenarioSpec) -> ScenarioSpec:
    def m(p: Point) -> Point:
        return (p[0], -p[1])

    return ScenarioSpec(
        code=spec.code,
        static_xy=tuple(m(p) for p in spec.static_xy),
        dynamic_rails=tuple((m(a), m(b)) for a, b in spec.dynamic_rails),
        dynamic_phases=spec.dynamic_phases,
        dynamic_patterns=spec.dynamic_patterns,
        dynamic_follow_offsets=tuple(m(p) for p in spec.dynamic_follow_offsets),
        target_route=tuple(m(p) for p in spec.target_route),
        interceptor_xy=m(spec.interceptor_xy),
        target_speed=spec.target_speed,
    )


# Structured challenges: parameter draws and builders


def sample_challenge_params(code: int, rng: random.Random) -> dict:
    """One layout of a challenge family; the same dict is replayed at every speed."""
    u = rng.uniform
    if code == SCENARIO_WALL:
        return {"shift_x": u(-0.18, 0.18), "pillars": rng.choice((4, 5, 6)), "wall_y": u(-0.2, 0.2),
                "target_x": u(0.85, 1.6), "target_y": u(-0.8, 0.8),
                "interceptor_y": u(-0.3, 0.3), "mirror": rng.random() < 0.5}
    if code == SCENARIO_NARROW_PASSAGE:
        return {"shift_x": u(-0.06, 0.06), "gap": u(0.38, 0.44), "target_x": u(0.8, 1.6),
                "target_y": u(-0.9, 0.9), "interceptor_y": u(-0.25, 0.25), "mirror": rng.random() < 0.5}
    if code == SCENARIO_DYNAMIC_OBSTACLES:
        patterns = ["rail", "wobble", "stop_and_go"]
        rng.shuffle(patterns)
        return {"rail_shift": u(-0.06, 0.06), "rail_phases": [u(0.0, 1.0) for _ in range(3)],
                "patterns": patterns, "target_x": u(1.5, 1.75), "half_span": u(0.7, 1.0),
                "center_y": u(-0.15, 0.15), "interceptor_y": u(-0.3, 0.3), "mirror": rng.random() < 0.5}
    raise ValueError(f"Unknown challenge code {code}")


def _wall_spec(p: dict, speed: float) -> ScenarioSpec:
    # A straight wall of pillars between the interceptor and a parked target.
    # The target sits behind the wall, within the wall's own vertical extent.
    n = int(p["pillars"])
    half = 0.40 * 0.5 * (n - 1)
    return ScenarioSpec(
        code=SCENARIO_WALL,
        static_xy=tuple((p["shift_x"], p["wall_y"] + 0.40 * (i - 0.5 * (n - 1))) for i in range(n)),
        dynamic_rails=(),
        dynamic_phases=(),
        target_route=((p["shift_x"] + p["target_x"], p["wall_y"] + p["target_y"] * half),),
        interceptor_xy=(p["shift_x"] - 1.76, p["interceptor_y"]),
        target_speed=0.0,
    )


def _narrow_passage_spec(p: dict, speed: float) -> ScenarioSpec:
    # Two L-shaped pillar groups leave a horizontal passage of half-width p["gap"]
    # between centre rows; the target is parked beyond it.
    arm_x, inner_x, mouth_x = -0.44 + p["shift_x"], -0.088 + p["shift_x"], 0.264 + p["shift_x"]
    gap = p["gap"]
    static = []
    for sign in (-1.0, 1.0):
        static += [(arm_x, sign * (gap + 0.4 * k)) for k in range(3)]
        static += [(inner_x, sign * gap), (mouth_x, sign * gap)]
    return ScenarioSpec(
        code=SCENARIO_NARROW_PASSAGE,
        static_xy=tuple(static),
        dynamic_rails=(),
        dynamic_phases=(),
        target_route=((p["target_x"] + p["shift_x"], p["target_y"]),),
        interceptor_xy=(-1.54, p["interceptor_y"]),
        target_speed=0.0,
    )


def _dynamic_spec(p: dict, speed: float) -> ScenarioSpec:
    # Three long rails (3.0 m, longer than training's 1.8 m) sweep the space
    # between interceptor and target. The target shuttles vertically on the
    # right; a fourth obstacle shields it at a fixed offset toward the interceptor.
    rails = tuple(((x + p["rail_shift"], -1.50), (x + p["rail_shift"], 1.50)) for x in (-0.92, -0.22, 0.30))
    x, c, h = p["target_x"], p["center_y"], p["half_span"]
    return ScenarioSpec(
        code=SCENARIO_DYNAMIC_OBSTACLES,
        static_xy=(),
        dynamic_rails=rails,
        dynamic_phases=tuple(p["rail_phases"]),
        dynamic_patterns=tuple(p["patterns"]),
        dynamic_follow_offsets=((-0.616, 0.0),),
        target_route=((x, c - h), (x, c + h), (x, c - h)),
        interceptor_xy=(-1.85, p["interceptor_y"]),
        target_speed=speed,
    )


CHALLENGE_BUILDERS = {
    SCENARIO_WALL: _wall_spec,
    SCENARIO_NARROW_PASSAGE: _narrow_passage_spec,
    SCENARIO_DYNAMIC_OBSTACLES: _dynamic_spec,
}


def challenge_spec(code: int, params: dict, speed: float) -> ScenarioSpec:
    spec = CHALLENGE_BUILDERS[code](params, speed)
    return _mirror_y(spec) if params["mirror"] else spec
