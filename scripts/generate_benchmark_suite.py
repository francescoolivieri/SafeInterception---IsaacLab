"""Generate a fixed evaluation suite for the requested task and arena."""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Each generator worker is one process; BLAS threads on top of that only fight over the cores.
for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

parser = argparse.ArgumentParser(description="Generate the evaluation suite for a task.")
parser.add_argument("--task", type=str, required=True, help="Gym task id (e.g. TargetInterception-State-Large-v0).")
parser.add_argument("--suite-name", type=str, default=None,
                    help="Benchmark-suite filename used by evaluate.py (default: the task arena's suite).")
parser.add_argument("--skip-existing", action="store_true",
                    help="Return successfully if the named benchmark suite already exists.")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=None, help="Suite seed (default: EVALUATION_SEED).")
parser.add_argument("--clutter-layouts", type=int, default=None, help="Layouts per random-clutter level.")
parser.add_argument("--challenge-layouts", type=int, default=None, help="Layouts per challenge family.")
args_cli = parser.parse_args()


from target_interception.scenarios import evaluation_design as design
from target_interception.scenarios.benchmark_suite import save_benchmark_suite, suite_cache_path
from target_interception.scenarios.offline_config import offline_cfg_for_task


def main() -> None:
    env_cfg = offline_cfg_for_task(args_cli.task)
    cache_path = suite_cache_path(env_cfg, args_cli.suite_name)
    if cache_path.is_file():
        if args_cli.skip_existing:
            print(f"[INFO] Benchmark suite already ready: {cache_path}")
            return
        print(f"[INFO] Benchmark suite already exists, overwriting: {cache_path}")
    suite = design.build_evaluation_suite(
        env_cfg,
        workers=args_cli.workers,
        seed=design.EVALUATION_SEED if args_cli.seed is None else args_cli.seed,
        clutter_layouts=design.CLUTTER_LAYOUTS if args_cli.clutter_layouts is None else args_cli.clutter_layouts,
        challenge_layouts=design.CHALLENGE_LAYOUTS if args_cli.challenge_layouts is None else args_cli.challenge_layouts,
    )
    saved = save_benchmark_suite(env_cfg, suite, args_cli.suite_name)
    print(f"[INFO] Benchmark suite ready ({suite.total} trials): {saved}")


if __name__ == "__main__":
    main()
