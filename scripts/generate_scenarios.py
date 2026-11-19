"""Generate and validate named scenario pools without launching Isaac Sim."""

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Each generator worker is one process; BLAS threads on top of that only fight over the cores.
for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")


def _phase_size_overrides(text: str) -> tuple[tuple[int, int], ...]:
    overrides: list[tuple[int, int]] = []
    seen: set[int] = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise argparse.ArgumentTypeError(f"Invalid phase size {chunk!r}; expected PHASE:ROWS.")
        phase_text, rows_text = chunk.split(":", 1)
        try:
            phase, rows = int(phase_text), int(rows_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid phase size {chunk!r}; expected integer PHASE:ROWS.") from exc
        if phase <= 0 or rows <= 0:
            raise argparse.ArgumentTypeError(f"Invalid phase size {chunk!r}; phase and rows must be positive.")
        if phase in seen:
            raise argparse.ArgumentTypeError(f"Duplicate pool-size override for phase {phase}.")
        seen.add(phase)
        overrides.append((phase, rows))
    if not overrides:
        raise argparse.ArgumentTypeError("Expected at least one PHASE:ROWS entry.")
    return tuple(overrides)


parser = argparse.ArgumentParser(description="Generate an offline target-interception pool.")
parser.add_argument("--task", type=str, required=True, help="Gym task id (e.g. TargetInterception-State-v0).")
parser.add_argument("--curriculum-profile", choices=["penalty_staged"], default=None)
parser.add_argument(
    "--phases", type=int, nargs="+", default=None,
    help="Generate only these enabled curriculum phases (e.g. --phases 4).",
)
parser.add_argument(
    "--pool-size",
    type=int,
    default=None,
    help="Scenarios generated for every enabled curriculum phase (default: the thesis size of each phase).",
)
parser.add_argument(
    "--pool-size-by-phase",
    type=_phase_size_overrides,
    default=None,
    help="Comma-separated PHASE:ROWS overrides; e.g. 1:2000,2:5000. Unlisted phases use --pool-size/default.",
)
parser.add_argument(
    "--pool-name",
    type=str,
    required=True,
    help="Human-readable pool filename. Training selects it with env.scenario.pool_name.",
)
parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="Return successfully if the named scenario pool already exists.",
)
parser.add_argument("--seed", type=int, default=None, help="Pool sampling seed (default: scenario.pool_seed).")
parser.add_argument(
    "--workers",
    type=int,
    default=0,
    help="CPU generator processes; 0 uses up to eight logical CPUs (default: 0).",
)
parser.add_argument(
    "--diagnostics",
    action="store_true",
    help="Collect aggregate stage timings/rejection counters and save <pool>.generation.json.",
)
args_cli = parser.parse_args()

# Multiprocessing's resource sharer creates temporary sockets and files. On the
# cluster, putting those on NFS leaves busy .nfs* tombstones when workers exit.
_temp_dir = os.environ.get("SLURM_TMPDIR")
if _temp_dir:
    Path(_temp_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(Path(_temp_dir).resolve())
    tempfile.tempdir = os.environ["TMPDIR"]


from target_interception.scenarios.pool import (
    build_scenario_pools, save_scenario_pools, scenario_pool_cache_path, validate_scenario_pools,
)
from target_interception.scenarios.generation_stats import GenerationStats
from target_interception.scenarios.context import ScenarioContext
from target_interception.scenarios.offline_config import offline_cfg_for_task


def main() -> None:
    env_cfg = offline_cfg_for_task(args_cli.task)
    if args_cli.curriculum_profile:
        from target_interception.scenarios import penalty_curriculum
        penalty_curriculum.apply(env_cfg)

    if args_cli.phases is not None:
        enabled = {i for i, fraction in enumerate(env_cfg.curriculum_phase_fractions, 1) if fraction > 0}
        if not set(args_cli.phases) <= enabled:
            parser.error(f"--phases must be selected from {sorted(enabled)} for this task")
        env_cfg.curriculum_phase_fractions = tuple(
            fraction if phase in args_cli.phases else 0.0
            for phase, fraction in enumerate(env_cfg.curriculum_phase_fractions, 1)
        )
        env_cfg.scenario.pool_size_by_phase = tuple(
            (phase, rows) for phase, rows in env_cfg.scenario.pool_size_by_phase if phase in args_cli.phases
        )

    if args_cli.pool_size is not None:
        # One size for every phase, in place of the per-phase defaults of the task.
        env_cfg.scenario.pool_size = max(1, int(args_cli.pool_size))
        env_cfg.scenario.pool_size_by_phase = ()
    if args_cli.pool_size_by_phase is not None:
        env_cfg.scenario.pool_size_by_phase = args_cli.pool_size_by_phase
    if args_cli.pool_name is not None:
        env_cfg.scenario.pool_name = args_cli.pool_name
    if args_cli.seed is not None:
        env_cfg.scenario.pool_seed = args_cli.seed
    env_cfg.sim.device = "cpu"

    cache_path = scenario_pool_cache_path(env_cfg)
    if cache_path.is_file():
        if args_cli.skip_existing:
            print(f"[INFO] Scenario pool already ready: {cache_path}")
            return
        print(f"[INFO] Scenario pool already exists, overwriting: {cache_path}")

    workers = args_cli.workers if args_cli.workers > 0 else min(8, os.cpu_count() or 1)
    stats = GenerationStats() if args_cli.diagnostics else None
    started = time.perf_counter()
    pools = build_scenario_pools(
        env_cfg, "cpu", workers=workers, generation_stats=stats
    )
    generation_seconds = time.perf_counter() - started
    audit_started = time.perf_counter()
    validate_scenario_pools(env_cfg, pools)
    audit_seconds = time.perf_counter() - audit_started
    saved = save_scenario_pools(env_cfg, pools, generation_stats=stats)
    print(f"[INFO] Scenario pool ready (every row audited): {saved}")
    if stats is not None:
        stats.add_time("generation_wall", generation_seconds)
        stats.add_time("final_audit", audit_seconds)
        report_path = saved.with_suffix(".generation.json")
        rows = {str(phase): int(bank["path_type"].shape[0]) for phase, bank in pools.items()}
        context = ScenarioContext.from_cfg(env_cfg)
        report = {
            "task": args_cli.task,
            "seed": int(env_cfg.scenario.pool_seed),
            "frames": 0,
            "reference_control_steps": sum(rows.values()) * (context._path_steps - 1),
            "checkpoint": None,
            "evaluation_mode": "offline generation, integrated audit",
            "rows_by_phase": rows,
            "workers": workers,
            "device": "cpu",
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpus": os.cpu_count(),
            },
            "thread_settings": {
                key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "scenario_config": env_cfg.scenario.to_dict(),
            "diagnostics": stats.to_dict(),
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[INFO] Generation diagnostics: {report_path}")


if __name__ == "__main__":
    main()
