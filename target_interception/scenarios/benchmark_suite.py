"""Build, save and replay fixed evaluation suites.

Rows use the training waypoint timing. Suite obstacle counts set the evaluation
environment's capacity; replay draws no new scenarios."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from target_interception.scenarios import generation
from target_interception.world_frame import WORLD_FRAME_ID

from . import path_motion
from .context import ScenarioContext
from .large_arena import LARGE_ARENA_FRAME_ID, LARGE_ARENA_SUITE_NAME, ROOM_SUITE_NAME
from .scenario_defs import SCENARIO_LABELS, SCENARIO_PHASE, ScenarioSpec

SUITE_SCHEMA_VERSION = 13
DEFAULT_SUITE_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmark_suite"
# The large arena is the in-distribution default; the room suite is the OOD check.
DEFAULT_SUITE_NAME = LARGE_ARENA_SUITE_NAME
SUITE_NAME_BY_WORLD_FRAME = {LARGE_ARENA_FRAME_ID: LARGE_ARENA_SUITE_NAME, WORLD_FRAME_ID: ROOM_SUITE_NAME}
# An authored route may be slowed by at most this many 0.05 m/s steps to pass
# the kinematic gate; beyond that the layout is rejected, never parked.
MAX_RETIME_STEPS = 6

ROW_KEYS = (
    "target_waypoints",
    # Stored, not read by the env.
    "target_yaw",
    "interceptor_start",
    "static_xy",
    "static_active",
    "dynamic_waypoints",
    "dynamic_active",
    "path_type",
)
def _repeat_closed_route(xy: torch.Tensor, required: float) -> torch.Tensor:
    """Lap a closed authored route until it covers required metres; open routes stay as they are."""
    if xy.shape[0] <= 2 or float(torch.linalg.vector_norm(xy[0] - xy[-1]).item()) > 1e-4:
        return xy
    length = float(torch.linalg.vector_norm(xy[1:] - xy[:-1], dim=-1).sum().item())
    if length <= 1e-6 or length >= required:
        return xy
    extra_laps = max(1, int((required - length) / length) + 1)
    return torch.cat([xy] + [xy[1:] for _ in range(extra_laps)], dim=0)


def _smooth_route_polyline(ctx: ScenarioContext, xy: torch.Tensor) -> torch.Tensor:
    """Resample and round the corners of an authored route polyline."""
    scenario = ctx.cfg.scenario
    xy = path_motion.resample(xy, float(getattr(scenario, "smooth_evader_spacing", 0.45)))
    return path_motion.round_corners(xy, int(getattr(scenario, "smooth_evader_chaikin_passes", 2)))


def _evader_mid_z(ctx: ScenarioContext) -> float:
    from target_interception.task import target

    clearance = target.wall_clearance(ctx)
    lo = float(ctx._arena_min_safe[2]) + clearance
    hi = float(ctx._arena_max_safe[2]) - clearance
    if hi <= lo:
        lo, hi = float(ctx._arena_min_safe[2]), float(ctx._arena_max_safe[2])
    return 0.5 * (lo + hi)


def _rail_pattern(spec: ScenarioSpec, i: int) -> str:
    return spec.dynamic_patterns[i] if i < len(spec.dynamic_patterns) else "rail"


def evaluation_target_obstacle_clearance(ctx: ScenarioContext) -> float:
    """Target-to-pillar centre distance for evaluation: collision radii, no training reserves."""
    from target_interception.task import target

    return float(ctx.cfg.pillar_radius + target.target_radius(ctx))


def evaluation_target_trajectory_min_clearance(
    ctx: ScenarioContext,
    target_waypoints: torch.Tensor,
    static_xy: torch.Tensor,
    static_active: torch.Tensor,
    dynamic_waypoints: torch.Tensor,
    dynamic_active: torch.Tensor,
) -> torch.Tensor:
    """Signed clearance from exact physical collisions in a benchmark row."""

    required = evaluation_target_obstacle_clearance(ctx)
    target_waypoints, dynamic_waypoints = generation.executed_target_schedules(ctx, target_waypoints, dynamic_waypoints)
    static = generation.target_static_min_separation(target_waypoints, static_xy, static_active) - required
    dynamic = generation.target_dynamic_min_separation(target_waypoints, dynamic_waypoints, dynamic_active) - required
    return torch.minimum(static, dynamic)


def _retime_to_reference_envelope(ctx: ScenarioContext, waypoints: torch.Tensor) -> torch.Tensor:
    """Slow an authored route until it passes the target limits.

    Generation's speed profile first, then up to MAX_RETIME_STEPS clock backoffs, else rejected.
    """
    from target_interception.task import target

    for _ in range(MAX_RETIME_STEPS + 1):
        if target.reference_kinematics_ok(ctx, waypoints):
            return waypoints
        waypoints = target.reduce_waypoint_speed(ctx, waypoints, decrement=0.05)
    raise ValueError("authored target route cannot be retimed into the reference envelope")


def _manual_route_waypoints(ctx: ScenarioContext, spec: ScenarioSpec, z: float) -> torch.Tensor:
    from target_interception.scenarios import routes as evader

    route_xy = torch.tensor(spec.target_route, device=ctx.device, dtype=torch.float32)
    if spec.target_speed <= 0.0:  # parked target
        waypoints = torch.zeros(ctx._path_waypoint_count, 3, device=ctx.device)
        waypoints[:, :2] = route_xy[0]
        waypoints[:, 2] = z
        return waypoints
    step_dist = evader.evader_step_distances(ctx, spec.target_speed)
    # Repeat before smoothing so the lap seams get rounded like any corner; the
    # headroom covers the length smoothing removes.
    route_xy = _repeat_closed_route(route_xy, 1.15 * float(step_dist.sum().item()) + 1.0)
    route_xy = _smooth_route_polyline(ctx, route_xy)
    waypoints = path_motion.profiled_waypoints(ctx, route_xy, step_dist, z)
    return _retime_to_reference_envelope(ctx, waypoints)


def _author_row(ctx: ScenarioContext, spec: ScenarioSpec) -> dict[str, torch.Tensor]:
    from target_interception.task import target

    device = ctx.device
    max_static = ctx._max_static_obstacles
    max_dynamic = ctx._max_dynamic_obstacles
    z = _evader_mid_z(ctx)

    static_xy = generation.inactive_obstacle_xy(ctx, max_static).clone()
    static_active = torch.zeros(max_static, dtype=torch.bool, device=device)
    for i, point in enumerate(spec.static_xy):
        static_xy[i] = torch.tensor(point, device=device, dtype=torch.float32)
        static_active[i] = True

    dynamic_waypoints = generation.inactive_dynamic_pos(ctx, max_dynamic).view(1, max_dynamic, 3).repeat(
        ctx._path_waypoint_count, 1, 1
    ).clone()
    dynamic_active = torch.zeros(max_dynamic, dtype=torch.bool, device=device)
    for i, (start, end) in enumerate(spec.dynamic_rails):
        phase = spec.dynamic_phases[i] if i < len(spec.dynamic_phases) else 0.0
        start_xy = torch.tensor(start, device=device, dtype=torch.float32)
        end_xy = torch.tensor(end, device=device, dtype=torch.float32)
        dynamic_waypoints[:, i] = generation.dynamic_rail_waypoints(
            ctx, start_xy, end_xy, pattern=_rail_pattern(spec, i), phase0=phase
        )
        dynamic_active[i] = True

    evader_waypoints = _manual_route_waypoints(ctx, spec, z)
    slot = len(spec.dynamic_rails)
    for offset in spec.dynamic_follow_offsets:
        offset_xy = torch.tensor(offset, device=device, dtype=torch.float32)
        dynamic_waypoints[:, slot, :2] = evader_waypoints[:, :2] + offset_xy
        dynamic_waypoints[:, slot, 2] = generation.dynamic_center_z(ctx)
        dynamic_active[slot] = True
        slot += 1

    _validate_row(ctx, spec, evader_waypoints, static_xy, static_active, dynamic_waypoints, dynamic_active)

    first_vel = target.first_velocity(ctx, evader_waypoints)
    yaw = float(torch.atan2(first_vel[1], first_vel[0]).item())
    pursuer_start = torch.tensor([spec.interceptor_xy[0], spec.interceptor_xy[1], z], device=device)
    return {
        "target_waypoints": evader_waypoints,
        "target_yaw": torch.tensor(yaw, device=device),
        "interceptor_start": pursuer_start,
        "static_xy": static_xy,
        "static_active": static_active,
        "dynamic_waypoints": dynamic_waypoints,
        "dynamic_active": dynamic_active,
        # Authored routes are continuous cruise references; a parked target is a hover row.
        "path_type": torch.tensor(1 if spec.target_speed > 0.0 else target.MOTION_MODE_ID["hover"],
                                  device=device, dtype=torch.long),
    }


def _validate_row(
    ctx: ScenarioContext, spec: ScenarioSpec, evader_waypoints, static_xy, static_active, dynamic_waypoints, dynamic_active
) -> None:
    label = SCENARIO_LABELS[spec.code]
    from target_interception.task import target

    # Exact collision body, without the training reserves.
    radius = target.target_radius(ctx)
    lo = ctx._arena_min_safe + radius
    hi = ctx._arena_max_safe - radius
    if not bool(torch.isfinite(evader_waypoints).all().item()):
        raise ValueError(f"{label}: target route contains non-finite values.")
    executed, _ = generation.executed_target_schedules(ctx, evader_waypoints)
    if not bool(torch.all((executed >= lo) & (executed <= hi)).item()):
        raise ValueError(f"{label}: target route leaves the safe arena.")
    xy = evader_waypoints[:, :2]
    interceptor_xy = torch.tensor(spec.interceptor_xy, device=xy.device)
    interceptor_clearance = float(ctx.cfg.scenario.interceptor_wall_clearance)
    interceptor_lo = ctx._arena_min_safe[:2] + interceptor_clearance
    interceptor_hi = ctx._arena_max_safe[:2] - interceptor_clearance
    if not bool(torch.all((interceptor_xy >= interceptor_lo) & (interceptor_xy <= interceptor_hi)).item()):
        raise ValueError(f"{label}: interceptor start leaves the safe arena.")
    if float(torch.linalg.vector_norm(xy[0] - interceptor_xy).item()) < float(target.start_distance_min(ctx)):
        raise ValueError(f"{label}: interceptor starts too close to the target.")
    clearance = float(
        evaluation_target_trajectory_min_clearance(
            ctx, evader_waypoints, static_xy, static_active, dynamic_waypoints, dynamic_active
        ).item()
    )
    kinematics_ok = target.reference_kinematics_ok(ctx, evader_waypoints)
    if clearance < -1e-6 or not kinematics_ok:
        raise ValueError(
            f"{label}: finalized target route is invalid "
            f"(clearance {clearance:.2f} m, kinematics_ok={kinematics_ok})."
        )


@dataclass
class BenchmarkSuite:
    rows: dict[str, torch.Tensor]
    tiers: torch.Tensor
    metadata: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return int(self.tiers.shape[0])

    # Obstacle slot counts baked into the rows; the eval env adopts these
    # (cfg.max_*_obstacles) so any scenario size fits regardless of training caps.
    @property
    def max_static_obstacles(self) -> int:
        return int(self.rows["static_active"].shape[1])

    @property
    def max_dynamic_obstacles(self) -> int:
        return int(self.rows["dynamic_active"].shape[1])

    def to(self, device: torch.device | str) -> "BenchmarkSuite":
        device = torch.device(device)
        return BenchmarkSuite(
            rows={key: value.to(device) for key, value in self.rows.items()},
            tiers=self.tiers.to(device),
            metadata=self.metadata,
        )


def _world_frame_id(cfg: Any | None) -> str:
    return str(getattr(cfg, "world_frame_id", WORLD_FRAME_ID)) if cfg is not None else LARGE_ARENA_FRAME_ID


def default_suite_name(cfg: Any | None = None) -> str:
    """The frozen suite of the cfg's arena (no cfg: the in-distribution large arena)."""
    return SUITE_NAME_BY_WORLD_FRAME.get(_world_frame_id(cfg), DEFAULT_SUITE_NAME)


def _suite_name(name: str | None = None) -> str:
    name = (name or DEFAULT_SUITE_NAME).strip()
    if not name:
        raise ValueError("benchmark suite name must not be empty.")
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("benchmark suite name must be a filename, not a path.")
    return name if name.endswith(".pt") else f"{name}.pt"


def suite_cache_path(cfg: Any | None = None, suite_name: str | None = None) -> Path:
    return DEFAULT_SUITE_DIR / _suite_name(suite_name or default_suite_name(cfg))


def generate_command_hint(cfg: Any | None = None, suite_name: str | None = None) -> str:
    task = "TargetInterception-State-v0" if _world_frame_id(cfg) == WORLD_FRAME_ID else "TargetInterception-State-Large-v0"
    return (
        f"python scripts/generate_benchmark_suite.py --task {task} "
        f"--suite-name {Path(_suite_name(suite_name or default_suite_name(cfg))).stem}"
    )


def save_benchmark_suite(cfg: Any, suite: BenchmarkSuite, suite_name: str | None = None) -> Path:
    cache_path = suite_cache_path(cfg, suite_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {SCENARIO_LABELS[c]: int((suite.tiers == c).sum().item()) for c in sorted(set(suite.tiers.tolist()))}

    tmp = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema_version": SUITE_SCHEMA_VERSION,
            "suite_name": Path(cache_path).stem,
            "counts_by_scenario": counts,
            "rows": {key: value.cpu() for key, value in suite.rows.items()},
            "tiers": suite.tiers.cpu(),
            "metadata": suite.metadata,
        },
        tmp,
    )
    tmp.replace(cache_path)

    manifest = cache_path.with_suffix(".json")
    manifest_tmp = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    manifest_tmp.write_text(
        json.dumps(
            {
                "suite_file": cache_path.name,
                "schema_version": SUITE_SCHEMA_VERSION,
                "suite_name": Path(cache_path).stem,
                "total": suite.total,
                "counts_by_scenario": counts,
                "metadata": suite.metadata,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest)
    return cache_path


def load_benchmark_suite(
    cfg: Any,
    device: torch.device | str = "cpu",
    suite_name: str | None = None,
) -> BenchmarkSuite:
    cache_path = suite_cache_path(cfg, suite_name)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Benchmark evaluation suite is missing:\n  {cache_path}\n"
            "evaluate.py scores every method on the same fixed suite. Generate it first with:\n"
            f"  {generate_command_hint(cfg, suite_name)}"
        )

    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    schema_version = int(payload.get("schema_version", -1))
    if schema_version != SUITE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Benchmark suite has schema version {schema_version}, expected {SUITE_SCHEMA_VERSION}; "
            f"regenerate it: {cache_path}"
        )
    rows = payload["rows"]
    # Replay on the suite's own knots (no launch zone without path_launch_time).
    cfg.path_launch_time = float(payload.get("metadata", {}).get("path_launch_time", 0.0))
    if set(rows) != set(ROW_KEYS):
        raise RuntimeError(f"Benchmark suite has an unexpected tensor schema: {cache_path}")
    total = int(payload["tiers"].shape[0])
    for key, tensor in rows.items():
        if int(tensor.shape[0]) != total:
            raise RuntimeError(f"Benchmark tensor '{key}' has {int(tensor.shape[0])} rows, expected {total}.")
    _validate_suite_shapes(rows, cfg, cache_path)
    metadata = payload.get("metadata", {})
    # Scoring a suite in another arena would mix frames.
    suite_frame = metadata.get("world_frame")
    if suite_frame is not None and suite_frame != _world_frame_id(cfg):
        raise RuntimeError(
            f"Benchmark suite world frame {suite_frame!r} does not match the env's "
            f"{_world_frame_id(cfg)!r}: {cache_path}"
        )
    if metadata and len(metadata.get("trials", [])) != total:
        raise RuntimeError("Benchmark trial metadata must match the tensor row count")
    return BenchmarkSuite(rows=rows, tiers=payload["tiers"], metadata=metadata).to(device)


def _validate_suite_shapes(rows: dict[str, torch.Tensor], cfg: Any, path: Path) -> None:
    """Check row shapes; only the waypoint count has to match the env timing.

    Obstacle slot counts come from the suite (evaluate.py adopts them).
    """
    n_static = rows["static_active"].shape[1]
    n_dynamic = rows["dynamic_active"].shape[1]
    ctx = ScenarioContext.from_cfg(cfg, "cpu")
    expected = {
        "target_waypoints": (ctx._path_waypoint_count, 3),
        "target_yaw": (),
        "interceptor_start": (3,),
        "static_xy": (n_static, 2),
        "static_active": (n_static,),
        "dynamic_waypoints": (ctx._path_waypoint_count, n_dynamic, 3),
        "dynamic_active": (n_dynamic,),
        "path_type": (),
    }
    for key, tail in expected.items():
        actual = tuple(rows[key].shape[1:])
        if actual != tail:
            raise RuntimeError(f"Benchmark tensor '{key}' has shape tail {actual}, expected {tail}: {path}")


class EvalScenarioDriver:
    """Assign each fixed suite row exactly once during evaluation."""

    def __init__(self, suite: BenchmarkSuite, device: torch.device | str) -> None:
        self.suite = suite.to(device)
        self.total = suite.total
        self._cursor = 0
        self.assigned_indices: dict[int, int] = {}

    def _phase_for(self, code: int) -> int:
        return SCENARIO_PHASE[int(code)]

    def assign(self, env, env_ids: torch.Tensor) -> None:
        from target_interception.task import curriculum

        for env_id in env_ids.tolist():
            env_id = int(env_id)
            if self._cursor < self.total:
                idx = self._cursor
                row = {key: self.suite.rows[key][idx] for key in ROW_KEYS}
                code = int(self.suite.tiers[idx].item())
                curriculum.assign_scenario_rows(env, env_id, row, self._phase_for(code))
                env._scenario_tier[env_id] = code
                self._cursor += 1
                self.assigned_indices[env_id] = idx
            else:
                env._scenario_tier[env_id] = 0
                self.assigned_indices[env_id] = -1

        env._move_static_obstacles(env_ids)
        env._move_dynamic_obstacles(env_ids)
