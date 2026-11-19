"""Offline scenario pools, grouped by curriculum phase.

Training loads a named pool and augments sampled rows at reset. Generation and
the pool audit use the same geometry and timing as the environment."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import torch

from target_interception.scenarios import config as scenario_config
from target_interception.scenarios import generation
from target_interception.world_frame import WORLD_FRAME_ID

from . import penalty_curriculum
from .context import ScenarioContext
from .generation_stats import GenerationStats

SCENARIO_POOL_SCHEMA_VERSION = 3

# Default pool location: <repo-root>/data/scenario_pools.
DEFAULT_POOL_DIR = Path(__file__).resolve().parents[2] / "data" / "scenario_pools"

# Tensors read at runtime; loading ignores any other stored field.
RUNTIME_POOL_KEYS = (
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
POOL_KEYS = RUNTIME_POOL_KEYS

def _world_frame_id(cfg: Any) -> str:
    """Frame of the configured arena; simulation-only variants carry their own ID."""
    return str(getattr(cfg, "world_frame_id", WORLD_FRAME_ID))


def enabled_pool_phases(cfg: Any) -> list[int]:
    """1-indexed curriculum phases with a non-zero fraction (always at least phase 1)."""
    phases = [
        phase
        for phase, fraction in enumerate(cfg.curriculum_phase_fractions, start=1)
        if float(fraction) > 0.0
    ]
    return phases or [1]


def _pool_sizes(cfg: Any) -> dict[int, int]:
    sizes = {phase: max(1, int(cfg.scenario.pool_size)) for phase in enabled_pool_phases(cfg)}
    for phase, rows in getattr(cfg.scenario, "pool_size_by_phase", ()) or ():
        phase, rows = int(phase), int(rows)
        if phase not in sizes:
            raise ValueError(
                f"Pool-size override was provided for disabled phase {phase}; "
                f"enabled phases are {sorted(sizes)}."
            )
        if rows <= 0:
            raise ValueError(f"Pool-size override for phase {phase} must be positive.")
        sizes[phase] = rows
    return sizes


def _pool_dir(cfg: Any) -> Path:
    configured = getattr(cfg.scenario, "pool_dir", None)
    return Path(configured) if configured else DEFAULT_POOL_DIR


def _pool_name_or_none(cfg: Any) -> str | None:
    name = getattr(cfg.scenario, "pool_name", None)
    if name is None:
        return None
    name = str(name).strip()
    if not name:
        return None
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("scenario.pool_name must be a filename, not a path.")
    return name if name.endswith(".pt") else f"{name}.pt"


def _pool_name(cfg: Any) -> str:
    name = _pool_name_or_none(cfg)
    if name is None:
        raise ValueError(
            "scenario.pool_name is required. Generate a named pool with:\n"
            "  python scripts/generate_scenarios.py --task <YourTask> --pool-name <pool_name>\n"
            "then select it in the training yaml with:\n"
            "  env:\n"
            "    scenario:\n"
            "      pool_name: <pool_name>"
        )
    return name


def scenario_pool_cache_path(cfg: Any) -> Path:
    return _pool_dir(cfg) / _pool_name(cfg)


def generate_command_hint(cfg: Any) -> str:
    command = (
        "python scripts/generate_scenarios.py --task <YourTask> "
        f"--pool-name {Path(_pool_name(cfg)).stem} --pool-size {int(cfg.scenario.pool_size)}"
    )
    overrides = getattr(cfg.scenario, "pool_size_by_phase", ()) or ()
    if overrides:
        pairs = ",".join(f"{int(phase)}:{int(rows)}" for phase, rows in overrides)
        command += f" --pool-size-by-phase {pairs}"
    return command


# Build

def _row_seed(seed: int, phase: int, row: int) -> int:
    return (int(seed) + 1_000_003 * int(phase) + 97_003 * int(row)) % (2**63 - 1)


def _count_request(stats: GenerationStats | None, request, outcome: str) -> None:
    if stats is None:
        return
    stats.count(f"coverage.obstacle_counts.{request.static_count}+{request.dynamic_count}.{outcome}")
    stats.count(f"coverage.target_speed.{request.speed_bin}.{outcome}")


def _sample_request(ctx, phase: int, request, attempts_per_row: int, seed: int) -> tuple[dict, int]:

    started = time.perf_counter() if ctx.generation_stats is not None else None
    torch.manual_seed(seed)
    for attempt in range(1, attempts_per_row + 1):
        _count_request(ctx.generation_stats, request, "attempts")
        scenario = generation.sample_scenario(ctx, phase, coverage=request)
        if scenario is not None:
            # Keep the requested bucket while retrying rare final-geometry
            # failures. The independent whole-pool audit still runs before save.
            layout = (scenario["static_xy"], scenario["static_active"],
                      scenario["dynamic_waypoints"], scenario["dynamic_active"])
            if not (
                generation.final_obstacle_layout_valid(ctx, *layout)
                and generation.final_target_trajectory_valid(ctx, scenario["target_waypoints"], *layout)
                and generation.point_free(ctx, scenario["interceptor_start"], *layout)
            ):
                _count_request(ctx.generation_stats, request, "final_geometry_rejected")
                continue
        if scenario is not None:
            _count_request(ctx.generation_stats, request, "accepted")
            if started is not None:
                elapsed = time.perf_counter() - started
                ctx.generation_stats.add_time("rows.total", elapsed)
                ctx.generation_stats.add_time(
                    f"coverage.obstacle_counts.{request.static_count}+{request.dynamic_count}", elapsed
                )
                ctx.generation_stats.add_time(f"coverage.target_speed.{request.speed_bin}", elapsed)
            return {
                "target_waypoints": scenario["target_waypoints"].cpu(),
                "target_yaw": scenario["target_yaw"],
                "interceptor_start": scenario["interceptor_start"].cpu(),
                "static_xy": scenario["static_xy"].cpu(),
                "static_active": scenario["static_active"].cpu(),
                "dynamic_waypoints": scenario["dynamic_waypoints"].cpu(),
                "dynamic_active": scenario["dynamic_active"].cpu(),
                "path_type": scenario["path_type"],
            }, attempt
    raise RuntimeError(
        f"Could not fill phase-{phase} coverage request {request} after "
        f"{attempts_per_row} attempts. Loosen that bucket or raise scenario_attempts."
    )


def _sample_request_chunk(
    cfg, phase: int, jobs: list[tuple[int, Any, int]], attempts_per_row: int, diagnostics: bool
):
    """Process-worker entry point returning one packed tensor bank.

    Packing here is important for large pools: sending each row's many small
    tensors separately creates tens of thousands of shared-memory mappings.
    """
    torch.set_num_threads(1)
    ctx = ScenarioContext.from_cfg(cfg, "cpu")
    if diagnostics:
        ctx.generation_stats = GenerationStats()
    rows: dict[str, list[torch.Tensor]] = {key: [] for key in POOL_KEYS}
    row_indices: list[int] = []
    attempt_counts: list[int] = []
    for row, request, seed in jobs:
        scenario, attempts = _sample_request(ctx, phase, request, attempts_per_row, seed)
        _append_scenario(rows, scenario, ctx.device)
        row_indices.append(row)
        attempt_counts.append(attempts)
    return (
        row_indices,
        {key: torch.stack(values) for key, values in rows.items()},
        attempt_counts,
        ctx.generation_stats.to_dict() if ctx.generation_stats is not None else None,
    )


def _append_scenario(rows: dict[str, list[torch.Tensor]], scenario: dict, device: torch.device) -> None:
    rows["target_waypoints"].append(scenario["target_waypoints"].to(device))
    rows["target_yaw"].append(torch.tensor(float(scenario["target_yaw"]), device=device))
    rows["interceptor_start"].append(scenario["interceptor_start"].to(device))
    rows["static_xy"].append(scenario["static_xy"].to(device))
    rows["static_active"].append(scenario["static_active"].to(device))
    rows["dynamic_waypoints"].append(scenario["dynamic_waypoints"].to(device))
    rows["dynamic_active"].append(scenario["dynamic_active"].to(device))
    rows["path_type"].append(torch.tensor(int(scenario["path_type"]), device=device, dtype=torch.long))


def build_scenario_pools(
    cfg: Any,
    device: torch.device | str = "cpu",
    *,
    workers: int = 1,
    generation_stats: GenerationStats | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Fill every coverage request of every enabled phase."""
    from target_interception.scenarios.coverage import (
        build_coverage_schedule,
        quota_histogram,
    )

    ctx = ScenarioContext.from_cfg(cfg, device)
    ctx.generation_stats = generation_stats
    pool_sizes = _pool_sizes(cfg)
    attempts_per_row = max(1, int(cfg.scenario.scenario_attempts))
    seed = int(cfg.scenario.pool_seed)
    workers = max(1, int(workers))
    if workers > 1 and ctx.device.type != "cpu":
        raise ValueError("Parallel scenario generation is CPU-only; use --device cpu or --workers 1.")

    pools: dict[int, dict[str, torch.Tensor]] = {}
    start = time.perf_counter()
    print(
        f"[INFO] Generating interception scenario pools: rows={dict(sorted(pool_sizes.items()))}, "
        f"seed={seed}, workers={workers}"
    )

    fork_devices = [ctx.device.index or 0] if ctx.device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        for phase, pool_size in sorted(pool_sizes.items()):
            phase_cfg = penalty_curriculum.generation_cfg(cfg, phase)
            ctx = ScenarioContext.from_cfg(phase_cfg, device)
            ctx.generation_stats = generation_stats
            rows: dict[str, list[torch.Tensor]] = {key: [] for key in POOL_KEYS}
            torch.manual_seed(_row_seed(seed, phase, -1))
            requests = build_coverage_schedule(ctx, phase, pool_size)
            print(f"[INFO] Phase {phase} coverage quotas: {quota_histogram(requests)}")
            report_stride = max(1, pool_size // 10)
            jobs = [
                (row, request, _row_seed(seed, phase, row))
                for row, request in enumerate(requests)
            ]
            results: list[tuple[dict, int] | None] = [None] * pool_size
            packed_pool: dict[str, torch.Tensor] | None = None
            attempt_counts: list[int] = [0] * pool_size
            attempts = 0
            accepted = 0
            next_report = report_stride
            if workers == 1:
                for row, request, row_seed in jobs:
                    results[row] = _sample_request(ctx, phase, request, attempts_per_row, row_seed)
                    attempts += results[row][1]
                    attempt_counts[row] = results[row][1]
                    accepted += 1
                    if accepted >= next_report or accepted == pool_size:
                        print(f"[INFO] Phase {phase}: {accepted}/{pool_size} accepted from {attempts} attempts")
                        next_report += report_stride
            else:
                chunk_size = max(1, min(32, (pool_size + workers * 20 - 1) // (workers * 20)))
                chunks = [jobs[index : index + chunk_size] for index in range(0, pool_size, chunk_size)]
                context = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                    # Keep only a small bounded set of completed tensor banks in
                    # flight. This prevents worker output from accumulating in
                    # RAM when later chunks finish before earlier ones.
                    chunk_iter = iter(chunks)
                    pending = {
                        executor.submit(
                            _sample_request_chunk,
                            phase_cfg,
                            phase,
                            chunk,
                            attempts_per_row,
                            generation_stats is not None,
                        )
                        for chunk in [next(chunk_iter, None) for _ in range(workers * 2)]
                        if chunk is not None
                    }
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            row_indices, chunk_pool, chunk_attempts, chunk_stats = future.result()
                            if generation_stats is not None and chunk_stats is not None:
                                generation_stats.merge(chunk_stats)
                            if packed_pool is None:
                                packed_pool = {
                                    key: torch.empty(
                                        (pool_size, *values.shape[1:]),
                                        dtype=values.dtype,
                                        device=ctx.device,
                                    )
                                    for key, values in chunk_pool.items()
                                }
                            index = torch.tensor(row_indices, dtype=torch.long, device=ctx.device)
                            for key, values in chunk_pool.items():
                                packed_pool[key].index_copy_(0, index, values.to(ctx.device))
                            for row, row_attempts in zip(row_indices, chunk_attempts):
                                attempt_counts[row] = row_attempts
                            attempts += sum(chunk_attempts)
                            accepted += len(row_indices)
                        if accepted >= next_report or accepted == pool_size:
                            print(
                                f"[INFO] Phase {phase}: {accepted}/{pool_size} accepted "
                                f"from {attempts} attempts"
                            )
                            while next_report <= accepted:
                                next_report += report_stride

                        for _ in done:
                            chunk = next(chunk_iter, None)
                            if chunk is not None:
                                pending.add(
                                    executor.submit(
                                        _sample_request_chunk,
                                        phase_cfg,
                                        phase,
                                        chunk,
                                        attempts_per_row,
                                        generation_stats is not None,
                                    )
                                )

            if workers == 1:
                for result in results:
                    assert result is not None
                    _append_scenario(rows, result[0], ctx.device)
                packed_pool = {key: torch.stack(values).cpu() for key, values in rows.items()}
            assert packed_pool is not None

            sorted_attempt_counts = sorted(attempt_counts)
            if generation_stats is not None:
                generation_stats.count("rows.accepted", pool_size)
                generation_stats.count("rows.attempts_total", sum(attempt_counts))
                for row_attempts in attempt_counts:
                    generation_stats.count(f"rows.attempts_histogram.{row_attempts}")
            p95 = sorted_attempt_counts[
                min(len(sorted_attempt_counts) - 1, int(0.95 * len(sorted_attempt_counts)))
            ]
            p50 = sorted_attempt_counts[(len(sorted_attempt_counts) - 1) // 2]
            print(
                f"[INFO] Phase {phase} attempts/row: p50={p50}, p95={p95}, max={sorted_attempt_counts[-1]}, "
                f"mean={sum(sorted_attempt_counts) / len(sorted_attempt_counts):.1f}"
            )

            pools[phase] = {key: values.cpu() for key, values in packed_pool.items()}

    print(f"[INFO] Generated interception scenario pools in {time.perf_counter() - start:.1f}s")
    return pools


def save_scenario_pools(
    cfg: Any,
    pools: dict[int, dict[str, torch.Tensor]],
    *,
    generation_stats: GenerationStats | None = None,
) -> Path:
    serialization_start = time.perf_counter()
    cache_path = scenario_pool_cache_path(cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pools = {
        phase: _runtime_pool(pool)
        for phase, pool in pools.items()
    }
    rows_by_phase = {
        str(phase): int(pool["path_type"].shape[0])
        for phase, pool in sorted(pools.items())
    }
    pool_name = Path(cache_path).stem

    tmp = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema_version": SCENARIO_POOL_SCHEMA_VERSION,
            "pool_name": pool_name,
            "curriculum_profile": getattr(cfg.scenario, "curriculum_profile", None),
            "world_frame_id": _world_frame_id(cfg),
            "arena_min": tuple(float(value) for value in cfg.arena_min),
            "arena_max": tuple(float(value) for value in cfg.arena_max),
            "rows_by_phase": rows_by_phase,
            "target_interpolation": "cubic",
            "pools": {str(phase): pool for phase, pool in pools.items()},
        },
        tmp,
    )
    tmp.replace(cache_path)

    manifest = cache_path.with_suffix(".json")
    manifest_tmp = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    manifest_tmp.write_text(
        json.dumps(
            {
                "scenario_file": cache_path.name,
                "schema_version": SCENARIO_POOL_SCHEMA_VERSION,
                "pool_name": pool_name,
                "curriculum_profile": getattr(cfg.scenario, "curriculum_profile", None),
                "world_frame_id": _world_frame_id(cfg),
                "arena_min": tuple(float(value) for value in cfg.arena_min),
                "arena_max": tuple(float(value) for value in cfg.arena_max),
                "rows_by_phase": rows_by_phase,
                "target_interpolation": "cubic",
                "phases": sorted(pools),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest)
    if generation_stats is not None:
        generation_stats.add_time("serialization", time.perf_counter() - serialization_start)
    return cache_path


# Load (training)

def load_scenario_pools(cfg: Any, device: torch.device | str = "cpu") -> dict[int, dict[str, torch.Tensor]]:
    """Load the named pool selected by cfg.scenario.pool_name onto device."""
    cache_path = scenario_pool_cache_path(cfg)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Target-interception scenario pool is missing:\n  {cache_path}\n"
            "Training needs a pre-generated scenario pool. Generate it first with:\n"
            f"  {generate_command_hint(cfg)}"
        )

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load scenario pool {cache_path}: {exc}") from exc

    if int(payload.get("schema_version", -1)) != SCENARIO_POOL_SCHEMA_VERSION:
        raise RuntimeError(f"Scenario pool has the wrong schema version: {cache_path}")

    stored_name = payload.get("pool_name")
    if stored_name is not None and str(stored_name) != cache_path.stem:
        print(
            f"[WARN] Scenario pool filename '{cache_path.stem}' does not match its stored name "
            f"'{stored_name}': {cache_path}"
        )
    if payload.get("world_frame_id") != _world_frame_id(cfg):
        raise RuntimeError(
            f"Scenario pool uses world frame {payload.get('world_frame_id')!r}, expected "
            f"{_world_frame_id(cfg)!r}: {cache_path}"
        )
    for field, expected in (("arena_min", cfg.arena_min), ("arena_max", cfg.arena_max)):
        stored = payload.get(field)
        if stored is None or len(stored) != 3 or not torch.allclose(
            torch.as_tensor(stored, dtype=torch.float64),
            torch.as_tensor(expected, dtype=torch.float64),
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError(
                f"Scenario pool {field}={stored!r} does not match the current "
                f"{tuple(float(value) for value in expected)!r}: {cache_path}"
            )

    if payload.get("target_interpolation") != "cubic":
        raise RuntimeError(f"Scenario pool was not generated for cubic target replay; regenerate it: {cache_path}")

    expected_profile = getattr(cfg.scenario, "curriculum_profile", None)
    if payload.get("curriculum_profile") != expected_profile:
        raise RuntimeError(f"Scenario pool curriculum profile does not match {expected_profile!r}: {cache_path}")
    pools = {int(phase): _runtime_pool(pool) for phase, pool in payload["pools"].items()}
    missing = [phase for phase in enabled_pool_phases(cfg) if phase not in pools]
    if missing:
        raise RuntimeError(
            f"Scenario pool '{Path(cache_path).stem}' does not contain enabled phase(s) {missing}: {cache_path}"
        )

    ctx = ScenarioContext.from_cfg(cfg, "cpu")
    device = torch.device(device)
    out: dict[int, dict[str, torch.Tensor]] = {}
    rows_by_phase: dict[int, int] = {}
    for phase, pool in pools.items():
        if set(pool) != set(RUNTIME_POOL_KEYS):
            raise RuntimeError(f"Scenario pool phase {phase} has an unexpected tensor schema: {cache_path}")
        actual_rows = int(pool["path_type"].shape[0])
        if actual_rows <= 0:
            raise RuntimeError(f"Scenario pool phase {phase} is empty: {cache_path}")
        _validate_pool_shapes(pool, ctx, cache_path, phase)
        rows_by_phase[phase] = actual_rows
        out[phase] = {key: tensor.to(device) for key, tensor in pool.items()}

    size_mib = sum(t.numel() * t.element_size() for pool in pools.values() for t in pool.values()) / (1024.0 * 1024.0)
    print(
        f"[INFO] Loaded interception scenario pools: rows={dict(sorted(rows_by_phase.items()))}, "
        f"device={device}, memory={size_mib:.1f} MiB"
    )
    return out


def _validate_pool_shapes(pool: dict[str, torch.Tensor], ctx: ScenarioContext, path: Path, phase: int) -> None:
    """Fail early when a named pool was generated for an incompatible env shape."""
    rows = int(pool["path_type"].shape[0])
    expected = {
        "target_waypoints": (ctx._path_waypoint_count, 3),
        "target_yaw": (),
        "interceptor_start": (3,),
        "static_xy": (ctx._max_static_obstacles, 2),
        "static_active": (ctx._max_static_obstacles,),
        "dynamic_waypoints": (ctx._path_waypoint_count, ctx._max_dynamic_obstacles, 3),
        "dynamic_active": (ctx._max_dynamic_obstacles,),
        "path_type": (),
    }
    for key, tail in expected.items():
        if int(pool[key].shape[0]) != rows:
            raise RuntimeError(
                f"Scenario pool phase {phase} tensor '{key}' has {int(pool[key].shape[0])} rows, "
                f"expected {rows}: {path}"
            )
        actual = tuple(pool[key].shape[1:])
        if actual != tail:
            raise RuntimeError(
                f"Scenario pool phase {phase} tensor '{key}' has shape tail {actual}, "
                f"expected {tail}: {path}"
            )


def _runtime_pool(pool: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep the runtime tensors and discard optional stored descriptors."""
    missing = [key for key in RUNTIME_POOL_KEYS if key not in pool]
    if missing:
        raise RuntimeError(f"Scenario pool is missing runtime tensor(s): {missing}")
    return {key: pool[key] for key in RUNTIME_POOL_KEYS}


def validate_scenario_pools(
    cfg: Any,
    pools: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Recheck every saved row with the continuous geometry gates, independently of generation."""
    from target_interception.task import target

    ctx = ScenarioContext.from_cfg(cfg, "cpu")
    report: dict[str, Any] = {
        "valid": True,
        "world_frame_id": _world_frame_id(cfg),
        "arena_min": tuple(float(value) for value in cfg.arena_min),
        "arena_max": tuple(float(value) for value in cfg.arena_max),
        "phases": {},
    }
    failures: list[str] = []
    for phase, source_pool in sorted(pools.items()):
        ctx = ScenarioContext.from_cfg(penalty_curriculum.generation_cfg(cfg, phase), "cpu")
        pool = _runtime_pool(source_pool)
        rows = int(pool["path_type"].shape[0])
        phase_report = {
            "rows": rows,
            "invalid_rows": 0,
            "minimum_target_clearance": None,
            "minimum_obstacle_separation": None,
            "minimum_start_distance_xy": None,
        }
        min_target_clearance = float("inf")
        min_obstacle_separation = float("inf")
        min_start_distance = float("inf")
        # Same fallback as the coverage schedule: an obstacle-free task has no valid pair.
        allowed_counts = set(scenario_config.phase_obstacle_count_choices(ctx, phase)) or {(0, 0)}
        for row in range(rows):
            row_failures: list[str] = []
            target_wp = pool["target_waypoints"][row]
            interceptor = pool["interceptor_start"][row]
            static_xy = pool["static_xy"][row]
            static_active = pool["static_active"][row].to(torch.bool)
            dynamic_wp = pool["dynamic_waypoints"][row]
            dynamic_active = pool["dynamic_active"][row].to(torch.bool)
            static_count = int(static_active.sum().item())
            dynamic_count = int(dynamic_active.sum().item())

            interceptor_clearance = float(ctx.cfg.scenario.interceptor_wall_clearance)
            interceptor_lo = ctx._arena_min_safe + interceptor_clearance
            interceptor_hi = ctx._arena_max_safe - interceptor_clearance
            interceptor_in_arena = torch.all(
                (interceptor >= interceptor_lo) & (interceptor <= interceptor_hi)
            )
            if not bool(interceptor_in_arena.item()):
                row_failures.append("interceptor start is outside the arena")

            if dynamic_count:
                dynamic_z = dynamic_wp[:, dynamic_active, 2]
                expected_z = generation.dynamic_center_z(ctx)
                if not bool(torch.all(torch.abs(dynamic_z - expected_z) <= 1e-5).item()):
                    row_failures.append(
                        f"dynamic obstacle center height does not match {expected_z:.4f} m"
                    )

            if (static_count, dynamic_count) not in allowed_counts:
                row_failures.append(f"invalid phase count {static_count}+{dynamic_count}")
            is_static = float(torch.linalg.vector_norm(target_wp[1:] - target_wp[:-1], dim=-1).max().item()) <= 1e-5
            stationary = penalty_curriculum.BANKS[phase - 1].speed is None if penalty_curriculum.enabled(cfg) else phase == 1
            if stationary != is_static:
                row_failures.append("target motion does not match the phase")
            if not generation.final_obstacle_layout_valid(
                ctx, static_xy, static_active, dynamic_wp, dynamic_active
            ):
                row_failures.append("obstacle layout is invalid")
            if not generation.final_target_trajectory_valid(
                ctx,
                target_wp,
                static_xy,
                static_active,
                dynamic_wp,
                dynamic_active,
            ):
                row_failures.append("target trajectory is invalid")

            start_distance = float(
                torch.linalg.vector_norm(target_wp[0, :2] - interceptor[:2]).item()
            )
            min_start_distance = min(min_start_distance, start_distance)
            if start_distance < target.start_distance_min(ctx) - 1e-6:
                row_failures.append(f"start distance {start_distance:.4f} m is too small")
            if penalty_curriculum.enabled(cfg):
                upper = penalty_curriculum.BANKS[phase - 1].distance[1]
                if upper is not None and start_distance >= upper + 1e-6:
                    row_failures.append(f"start distance {start_distance:.4f} m exceeds bank maximum")
            first_velocity = target.first_velocity(ctx, target_wp)
            initial_motion = target.initial_motion_bucket(interceptor, target_wp, first_velocity)
            toward_motion_min_start_distance = target.min_start_distance_for_toward_motion(ctx)
            if phase > 1 and start_distance < toward_motion_min_start_distance and initial_motion == "toward":
                row_failures.append("target starts toward the interceptor below its required separation")
            if not generation.point_free(
                ctx, interceptor, static_xy, static_active, dynamic_wp, dynamic_active
            ):
                row_failures.append("interceptor start overlaps an obstacle")
            if not bool(torch.isfinite(pool["target_yaw"][row]).item()):
                row_failures.append("target yaw is not finite")

            target_clearance = float(
                generation.target_trajectory_min_clearance(
                    ctx, target_wp, static_xy, static_active, dynamic_wp, dynamic_active
                ).item()
            )
            min_target_clearance = min(min_target_clearance, target_clearance)
            obstacle_separations = (
                generation.static_static_min_separation(static_xy, static_active),
                generation.dynamic_static_min_separation(
                    dynamic_wp, dynamic_active, static_xy, static_active
                ),
                generation.dynamic_dynamic_min_separation(dynamic_wp, dynamic_active),
            )
            min_obstacle_separation = min(
                min_obstacle_separation,
                *(float(value.item()) for value in obstacle_separations),
            )

            if dynamic_count:
                bins = tuple(getattr(ctx.cfg.scenario, "dynamic_rail_length_bins", ()))
                minimum_authored_length = min(
                    (min(float(lo), float(hi)) for lo, hi in bins),
                    default=0.0,
                )
                minimum_effective_span = float(
                    getattr(ctx.cfg.scenario, "dynamic_min_effective_span_fraction", 0.60)
                ) * minimum_authored_length
                for slot in torch.nonzero(dynamic_active, as_tuple=False).flatten().tolist():
                    actual_span = (
                        float(torch.pdist(dynamic_wp[:, slot, :2]).max().item())
                        if dynamic_wp.shape[0] > 1
                        else 0.0
                    )
                    if actual_span + 1e-5 < minimum_effective_span:
                        row_failures.append(f"dynamic slot {slot} has insufficient realized span")

            if row_failures:
                phase_report["invalid_rows"] += 1
                if len(failures) < 20:
                    failures.append(f"phase {phase} row {row}: " + "; ".join(row_failures))

        phase_report["minimum_target_clearance"] = (
            min_target_clearance if math.isfinite(min_target_clearance) else None
        )
        phase_report["minimum_obstacle_separation"] = (
            min_obstacle_separation if math.isfinite(min_obstacle_separation) else None
        )
        phase_report["minimum_start_distance_xy"] = (
            min_start_distance if math.isfinite(min_start_distance) else None
        )
        report["phases"][str(phase)] = phase_report
        report["valid"] &= phase_report["invalid_rows"] == 0

    if failures:
        raise RuntimeError(
            f"Scenario-pool validation failed ({sum(p['invalid_rows'] for p in report['phases'].values())} invalid rows):\n"
            + "\n".join(failures)
        )
    return report
