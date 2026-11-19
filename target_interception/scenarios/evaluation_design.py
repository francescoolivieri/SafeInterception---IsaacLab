"""Build fixed in-distribution and OOD evaluation suites.

Each moving-target layout is replayed at every configured speed and kept only
if all replays pass. Stationary targets share the random-clutter layouts."""
from __future__ import annotations

import copy
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import torch

from target_interception.scenarios import generation
from target_interception.world_frame import WORLD_FRAME_ID

from .benchmark_suite import (
    ROW_KEYS,
    BenchmarkSuite,
    _author_row,
    evaluation_target_trajectory_min_clearance,
)
from .context import ScenarioContext
from .coverage import build_coverage_schedule
from .large_arena import LARGE_ARENA_EVALUATION_CLUTTER, LARGE_ARENA_FRAME_ID
from .pool import _sample_request
from .scenario_defs import (
    CHALLENGE_CODES,
    CLUTTER_CODES,
    CLUTTER_LEVELS,
    OOD,
    SCENARIO_FAMILY,
    SCENARIO_LABELS,
    SCENARIO_PHASE,
    SCENARIO_STATIONARY,
    STATIC_CHALLENGE_CODES,
    challenge_spec,
    sample_challenge_params,
)

EVALUATION_SPEEDS = (1.2, 1.6, 2.0, 2.2)
CLUTTER_LAYOUTS = 200
CHALLENGE_LAYOUTS = 50
EVALUATION_SEED = 20261006
# Room layouts all succeed within 200 attempts, so raising the cap leaves that suite
# unchanged; the dense large-arena clutter needs more (layout_attempts records it).
LAYOUT_ATTEMPTS = 1000
STATIONARY_MIN_SEPARATION = 2.5  # m, pursuer start to parked evader (training sampler minimum: 1.2)


def clutter_levels(cfg) -> dict[int, tuple[tuple[int, int], int]]:
    """Clutter counts and phases for the configured arena."""
    if getattr(cfg, "world_frame_id", WORLD_FRAME_ID) != LARGE_ARENA_FRAME_ID:
        return CLUTTER_LEVELS
    return {code: (counts, phase)
            for (code, (_, phase)), counts in zip(CLUTTER_LEVELS.items(), LARGE_ARENA_EVALUATION_CLUTTER)}


def _layout_seed(seed: int, code: int, layout: int, attempt: int) -> int:
    return seed + 1_000_003 * code + 7_919 * layout + 104_729 * attempt


def _trial_meta(code: int, layout: int, speed: float, **extra) -> dict:
    label = SCENARIO_LABELS[code]
    return {
        "trial_id": f"{label}_{layout:03d}" + (f"_v{speed:g}" if speed > 0 else "_static"), "category": label, "code": code,
        "family": SCENARIO_FAMILY[code], "phase": SCENARIO_PHASE[code], "requested_speed": speed,
        "layout_id": f"{label}_{layout:03d}", **extra,
    }


def _replay_clutter_route(ctx, row: dict, route_xy: torch.Tensor, speeds) -> list[torch.Tensor] | None:
    """Retime one sampled route at every speed; None if any replay fails the audit."""
    from target_interception.scenarios import routes as evader
    from target_interception.task import target

    from . import path_motion

    z = float(row["target_waypoints"][0, 2])
    replays = []
    for speed in speeds:
        waypoints = path_motion.profiled_waypoints(ctx, route_xy, evader.evader_step_distances(ctx, speed), z)
        exhausted = float(torch.linalg.vector_norm(waypoints[-1] - waypoints[-2])) <= 1e-6
        if (exhausted or not target.reference_kinematics_ok(ctx, waypoints)
                or not generation.target_trajectory_in_arena(ctx, waypoints)
                or float(generation.target_trajectory_min_clearance(
                    ctx, waypoints, row["static_xy"], row["static_active"],
                    row["dynamic_waypoints"], row["dynamic_active"])) < -1e-6):
            return None
        replays.append(waypoints)
    return replays


def _parked_waypoints(ctx, row: dict, route: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    """The first route point the evader can hold for the whole episode, far from the pursuer.

    Moving pillars were timed against the moving evader, so the route start can be crossed
    later in the episode; then a later route point is used. None if no point qualifies.
    """

    for k, point in enumerate(route):
        if float(torch.linalg.vector_norm(point - row["interceptor_start"])) < STATIONARY_MIN_SEPARATION:
            continue
        waypoints = point.expand_as(route).clone()
        if generation.target_trajectory_in_arena(ctx, waypoints) and float(generation.target_trajectory_min_clearance(
                ctx, waypoints, row["static_xy"], row["static_active"],
                row["dynamic_waypoints"], row["dynamic_active"])) >= -1e-6:
            return waypoints, k
    return None


def generate_clutter_layout(cfg, code: int, layout: int, seed: int, speeds=EVALUATION_SPEEDS,
                            stationary: bool = False):
    """One random-clutter layout sampled at the top speed and replayed at every speed.

    With stationary=True the layout also yields a parked-evader row (last), and a layout
    without a valid parking point is resampled.
    """
    from target_interception.task import target

    torch.set_num_threads(1)
    (counts, phase) = clutter_levels(cfg)[code]
    cfg = copy.deepcopy(cfg)
    top = max(speeds)
    cfg.scenario.target_speed_range = (top, top)
    ctx = ScenarioContext.from_cfg(cfg, "cpu")
    for attempt in range(LAYOUT_ATTEMPTS):
        row_seed = _layout_seed(seed, code, layout, attempt)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(row_seed)
            request = build_coverage_schedule(ctx, phase, 1, obstacle_choices=(counts,))[0]
        try:
            row, _ = _sample_request(ctx, phase, request, int(cfg.scenario.scenario_attempts), row_seed)
        except RuntimeError:
            continue
        route_xy = getattr(ctx, "last_accepted_route_xy", None)
        if route_xy is None or not generation.final_obstacle_layout_valid(
            ctx, row["static_xy"], row["static_active"], row["dynamic_waypoints"], row["dynamic_active"]
        ):
            continue
        replays = _replay_clutter_route(ctx, row, route_xy, speeds)
        # The top-speed replay must reproduce the sampled row exactly.
        if replays is None or not torch.allclose(replays[speeds.index(top)], row["target_waypoints"], atol=1e-5):
            continue
        # The slowest replay samples the route most densely.
        parked = _parked_waypoints(ctx, row, replays[speeds.index(min(speeds))]) if stationary else None
        if stationary and parked is None:
            continue
        rows, metas = [], []
        for speed, waypoints in zip(speeds, replays):
            first_vel = target.first_velocity(ctx, waypoints)
            rows.append({**{k: torch.as_tensor(v).cpu() for k, v in row.items()},
                         "target_waypoints": waypoints.cpu(),
                         "target_yaw": torch.atan2(first_vel[1], first_vel[0]).cpu()})
            metas.append(_trial_meta(code, layout, speed, generation_seed=row_seed, layout_attempts=attempt + 1,
                                     coverage_request=asdict(request)))
        if parked is not None:
            waypoints, k_park = parked
            rows.append({**{k: torch.as_tensor(v).cpu() for k, v in row.items()},
                         "target_waypoints": waypoints.cpu(), "target_yaw": torch.tensor(0.0),
                         "path_type": torch.tensor(target.MOTION_MODE_ID["hover"], dtype=torch.long)})
            meta = _trial_meta(SCENARIO_STATIONARY, layout, 0.0, generation_seed=row_seed,
                               layout_attempts=attempt + 1, coverage_request=asdict(request))
            meta.update(trial_id=f"{SCENARIO_LABELS[SCENARIO_STATIONARY]}_{SCENARIO_LABELS[code]}_{layout:03d}",
                        layout_id=metas[0]["layout_id"], phase=SCENARIO_PHASE[code],
                        clutter_level=SCENARIO_LABELS[code],
                        parked_at_route_time=float(ctx._path_waypoint_steps[k_park]) * ctx._step_dt)  # on the slowest replay
            metas.append(meta)
        return rows, metas
    raise RuntimeError(f"{SCENARIO_LABELS[code]} layout {layout}: no replayable layout in {LAYOUT_ATTEMPTS} attempts")


def challenge_cfg(cfg):
    """Cfg for the authored rows: physical arena, obstacle slots sized for every family."""
    cfg = copy.deepcopy(cfg)
    cfg.arena_margin = 0.0
    cfg.max_static_obstacles = max(int(cfg.max_static_obstacles), 10)
    cfg.max_dynamic_obstacles = max(int(cfg.max_dynamic_obstacles), 4)
    # The dynamic-obstacle shuttle reverses in a tight U-turn; at 7.0 m/s^2 the brake,
    # turn and restart fall within one or two 0.2 s knots and the cubic reconstruction
    # overshoots the executed-acceleration gate. 3.0 spreads them over more knots.
    cfg.scenario.target_acceleration_limit = 3.0
    return cfg


def generate_challenge_layout(cfg, code: int, layout: int, seed: int, speeds=EVALUATION_SPEEDS):
    """One parametrised challenge layout, compiled at every speed."""
    torch.set_num_threads(1)
    ctx = ScenarioContext.from_cfg(challenge_cfg(cfg), "cpu")
    for attempt in range(LAYOUT_ATTEMPTS):
        layout_seed = _layout_seed(seed, code, layout, attempt)
        params = sample_challenge_params(code, random.Random(layout_seed))
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(layout_seed)
                rows = [_author_row(ctx, challenge_spec(code, params, speed)) for speed in speeds]
        except ValueError:
            continue
        metas = [_trial_meta(code, layout, speed, generation_seed=layout_seed, layout_attempts=attempt + 1,
                             params=params) for speed in speeds]
        return [{k: v.cpu() for k, v in row.items()} for row in rows], metas
    raise RuntimeError(f"{SCENARIO_LABELS[code]} layout {layout}: no valid layout in {LAYOUT_ATTEMPTS} attempts")


def _pad_row(row, n_static, n_dynamic):
    """Pad inactive obstacle slots up to the suite's shared capacity."""

    row = dict(row)
    pad_static = n_static - row["static_xy"].shape[0]
    pad_dynamic = n_dynamic - row["dynamic_active"].shape[0]
    if pad_static:
        row["static_xy"] = torch.cat((row["static_xy"], torch.zeros(pad_static, 2)))
        row["static_active"] = torch.cat((row["static_active"], torch.zeros(pad_static, dtype=torch.bool)))
    if pad_dynamic:
        waypoints = row["dynamic_waypoints"]
        row["dynamic_waypoints"] = torch.cat((waypoints, torch.zeros(waypoints.shape[0], pad_dynamic, 3)), dim=1)
        row["dynamic_active"] = torch.cat((row["dynamic_active"], torch.zeros(pad_dynamic, dtype=torch.bool)))
    return row


def audit_rows(rows, metadata, cfg) -> dict:
    """Recheck the finished rows against their family's limits, independently of generation."""
    from target_interception.task import target

    contexts = {"in_distribution": ScenarioContext.from_cfg(cfg, "cpu"),
                "ood": ScenarioContext.from_cfg(challenge_cfg(cfg), "cpu")}
    clearances = []
    for row, meta in zip(rows, metadata):
        ctx = contexts[meta["family"]]
        waypoints = row["target_waypoints"]
        if not all(bool(torch.isfinite(value).all()) for value in row.values()):
            raise ValueError(f"{meta['trial_id']}: nonfinite row")
        if not target.reference_kinematics_ok(ctx, waypoints):
            raise ValueError(f"{meta['trial_id']}: reference kinematics failed")
        clutter = meta["family"] != OOD
        clearance_fn = generation.target_trajectory_min_clearance if clutter else evaluation_target_trajectory_min_clearance
        clearance = float(clearance_fn(ctx, waypoints, row["static_xy"], row["static_active"],
                                       row["dynamic_waypoints"], row["dynamic_active"]))
        if clearance < -1e-5:
            raise ValueError(f"{meta['trial_id']}: invalid target-obstacle clearance {clearance}")
        executed, _ = generation.executed_target_schedules(ctx, waypoints)
        radius = target.target_radius(ctx)
        in_arena = generation.target_trajectory_in_arena(ctx, waypoints) if clutter else bool(
            ((executed >= ctx._arena_min_safe + radius) & (executed <= ctx._arena_max_safe - radius)).all())
        if not in_arena:
            raise ValueError(f"{meta['trial_id']}: reference outside arena")
        meta["minimum_reference_clearance"] = clearance
        clearances.append(clearance)
    return {"validated_rows": len(rows), "minimum_reference_clearance": min(clearances),
            "checks": ["finite", "kinematics", "target_obstacle_clearance", "arena"]}


def build_evaluation_suite(cfg, *, workers: int = 4, seed: int = EVALUATION_SEED,
                           clutter_layouts: int = CLUTTER_LAYOUTS, challenge_layouts: int = CHALLENGE_LAYOUTS,
                           speeds=EVALUATION_SPEEDS, stationary: bool = True) -> BenchmarkSuite:
    """Generate the suite without mutating the caller config."""
    if not cfg.enable_pillars or cfg.max_static_obstacles < 5 or cfg.max_dynamic_obstacles < 3:
        raise ValueError("the evaluation suite needs at least 5 static / 3 dynamic obstacle slots")
    speeds = tuple(float(v) for v in speeds)
    jobs = [(generate_clutter_layout, code, layout) for code in CLUTTER_CODES for layout in range(clutter_layouts)]
    n_clutter_jobs = len(jobs)
    # A parked target has no speed levels: those families get as many distinct
    # layouts as a moving family has trials, each recorded once at speed 0.
    jobs += [(generate_challenge_layout, code, layout, (0.0,)) for code in STATIC_CHALLENGE_CODES
             for layout in range(challenge_layouts * len(speeds))]
    jobs += [(generate_challenge_layout, code, layout) for code in CHALLENGE_CODES
             if code not in STATIC_CHALLENGE_CODES for layout in range(challenge_layouts)]
    jobs = [job if len(job) == 4 else (*job, speeds) for job in jobs]
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, cfg, code, layout, seed, job_speeds, *((stationary,) if i < n_clutter_jobs else ())): i
                   for i, (fn, code, layout, job_speeds) in enumerate(jobs)}
        for count, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if count % 25 == 0 or count == len(jobs):
                print(f"[INFO] Generated layout {count}/{len(jobs)}", flush=True)
    pairs = [pair for layout_rows, layout_metas in results for pair in zip(layout_rows, layout_metas)]
    # Parked-evader rows come last, in clutter-layout order.
    pairs = ([p for p in pairs if p[1]["code"] != SCENARIO_STATIONARY]
             + [p for p in pairs if p[1]["code"] == SCENARIO_STATIONARY])
    rows, metadata = [p[0] for p in pairs], [p[1] for p in pairs]
    n_static = max(row["static_xy"].shape[0] for row in rows)
    n_dynamic = max(row["dynamic_active"].shape[0] for row in rows)
    rows = [_pad_row(row, n_static, n_dynamic) for row in rows]
    audit = audit_rows(rows, metadata, cfg)

    ctx = ScenarioContext.from_cfg(cfg, "cpu")
    dt = (ctx._path_waypoint_steps[1:] - ctx._path_waypoint_steps[:-1]).float() * ctx._step_dt
    for row, meta in zip(rows, metadata):
        speed = torch.linalg.vector_norm(torch.diff(row["target_waypoints"][:, :2], dim=0), dim=-1) / dt
        meta.update(realized_mean_speed=float((speed * dt).sum() / dt.sum()),
                    realized_peak_speed=float(speed.max()),
                    initial_separation=float(torch.linalg.vector_norm(
                        row["target_waypoints"][0] - row["interceptor_start"])),
                    static_count=int(row["static_active"].sum()),
                    dynamic_count=int(row["dynamic_active"].sum()))
    stacked = {key: torch.stack([row[key] for row in rows]).cpu() for key in ROW_KEYS}
    scenario = cfg.scenario
    return BenchmarkSuite(stacked, torch.tensor([meta["code"] for meta in metadata]), metadata={
        "design": "clutter_and_challenges_v3",
        "suite_seed": seed,
        "world_frame": getattr(cfg, "world_frame_id", WORLD_FRAME_ID),
        "clutter_levels": {SCENARIO_LABELS[code]: list(counts) for code, (counts, _) in clutter_levels(cfg).items()},
        "requested_speeds": list(speeds),
        "static_target_categories": [SCENARIO_LABELS[c] for c in STATIC_CHALLENGE_CODES],
        "layouts": {"in_distribution_per_level": clutter_layouts, "ood_per_family": challenge_layouts,
                    "stationary_per_level": clutter_layouts if stationary else 0},
        "stationary_min_separation": STATIONARY_MIN_SEPARATION,
        "evader_limits": {
            "acceleration_limit": scenario.target_acceleration_limit,
            "max_acceleration": scenario.target_max_acceleration,
            "max_speed": scenario.target_max_speed,
        },
        "episode_length_s": cfg.episode_length_s,
        "path_waypoint_dt": cfg.path_waypoint_dt,
        "path_launch_time": cfg.path_launch_time,
        "path_launch_waypoint_dt": cfg.path_launch_waypoint_dt,
        "pillar_radius": cfg.pillar_radius,
        "clutter_arena_margin": cfg.arena_margin,
        "challenge_arena_margin": 0.0,
        "generator_config": scenario.to_dict(),
        "audit": audit,
        "trials": metadata,
    })
