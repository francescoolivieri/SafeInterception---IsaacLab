"""Run experiment evaluations and aggregate their metrics.

Both arenas run by default. This script launches evaluate.py in subprocesses
and can aggregate existing results without Isaac Sim."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
EVALUATE = os.path.join(REPO_ROOT, "scripts", "evaluate.py")
SOURCE_ROOT = REPO_ROOT
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from target_interception.scenarios.large_arena import ROOM_TASK_FOR_LARGE_TASK
from target_interception.scenarios.scenario_defs import EVALUATION_SCENARIO_CODES, FAMILIES, SCENARIO_LABELS

# Overall-metric columns: (metric key, column label, format, std key for "mean +/- std" or None).
METRIC_COLUMNS = [
    ("capture_rate", "capture", "{:.3f}", None),
    ("collision_rate", "collision", "{:.3f}", None),
    ("time_to_capture_mean_s", "ttc (s)", "{:.2f}", "time_to_capture_std_s"),
    ("min_safety_margin_mean", "margin", "{:.3f}", "min_safety_margin_std"),
    ("min_safety_margin_min", "margin worst", "{:.3f}", None),
    ("path_length_mean", "path len", "{:.2f}", "path_length_std"),
    ("filter_intervened_fraction", "filter frac", "{:.3f}", None),
]

SCENARIOS = [SCENARIO_LABELS[code] for code in EVALUATION_SCENARIO_CODES]


def parse_args():
    p = argparse.ArgumentParser(description="Run and aggregate interception experiments.")
    p.add_argument("--experiments", nargs="+", default=None, help="Experiment yamls (default: all configs/*.yaml).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Root for per-method runs and tables (default outputs/benchmark, or outputs/benchmark_ood_room).")
    p.add_argument("--aggregate-only", action="store_true", help="Skip rollouts; only read existing metrics.json.")
    p.add_argument("--across-seeds", action="store_true",
                   help="Only read <output_dir>/seed_*/<method>/metrics.json and write mean +/- std over seeds.")
    p.add_argument("--ood-room", action="store_true", help="Evaluate every method in the room only (evaluate.py --ood-room).")
    p.add_argument("--large-only", action="store_true", help="Evaluate in the large arena only, skipping the room.")
    p.add_argument("--headless", action="store_true", help="Forwarded to evaluate.py (run Isaac Sim headless).")
    p.add_argument("--device", type=str, default=None, help="Forwarded to evaluate.py (e.g. cuda:0).")
    args = p.parse_args()
    if args.ood_room and args.large_only:
        p.error("--ood-room and --large-only are mutually exclusive.")
    if args.across_seeds and not args.output_dir:
        p.error("--across-seeds needs --output_dir, the folder that holds the seed_* folders.")
    return args


def discover_experiments(selected) -> list[str]:
    if selected:
        return [os.path.abspath(p) for p in selected]
    files = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "*.yaml")))
    if not files:
        raise SystemExit("no configs/*.yaml found.")
    return files


def load_yaml(path: str) -> dict:
    import yaml  # local import: aggregation path needs yaml, not Isaac

    with open(path) as f:
        return yaml.safe_load(f) or {}


def run_experiment(exp_file: str, out_dir: str, args) -> None:
    """Roll one experiment through evaluate.py (one deterministic run on the fixed suite)."""
    cmd = [sys.executable, EVALUATE, "--experiment", exp_file, "--output_dir", out_dir, "--metrics-only"]
    if args.ood_room:
        cmd.append("--ood-room")
    if args.headless:
        cmd.append("--headless")
    if args.device:
        cmd += ["--device", args.device]
    print(f"[BENCH] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def read_metrics(metrics_path: str) -> dict | None:
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        return (json.load(f) or {}).get("metrics", {})


def fmt_cell(value, fmt: str, std=None) -> str:
    if value is None:
        return "-"
    cell = fmt.format(value)
    if std:
        cell += " +/- " + fmt.format(std)
    return cell


def write_tables_and_csv(rows: list[dict], out_dir: str) -> str:
    """Emit the markdown comparison (overall + per-scenario) and a flat csv."""
    md_path = os.path.join(out_dir, "comparison.md")
    csv_path = os.path.join(out_dir, "comparison.csv")
    lines = ["# Target-interception benchmark comparison (fixed evaluation suite)", ""]

    # Overall table
    header = ["method", "task", "episodes"] + [label for _, label, _, _ in METRIC_COLUMNS]
    lines += ["## Overall", "", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        m = r["metrics"]
        cells = [r["name"], r["task"], str(m.get("num_episodes", 0))]
        cells += [fmt_cell(m.get(key), fmt, m.get(std) if std else None) for key, _, fmt, std in METRIC_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    for family in FAMILIES:
        if not any(family in r["metrics"].get("by_family", {}) for r in rows):
            continue
        lines += ["", f"## {family.replace('_', ' ').capitalize()} (separate benchmark population)", "",
                  "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        for r in rows:
            m = r["metrics"].get("by_family", {}).get(family, {})
            cells = [r["name"], r["task"], str(m.get("num_episodes", 0))]
            cells += [fmt_cell(m.get(key), fmt, m.get(std) if std else None) for key, _, fmt, std in METRIC_COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")

    # Per-scenario capture, collision and successful-capture time
    scenario_header = ["method"] + [f"{name} {w}" for name in SCENARIOS for w in ("cap", "coll", "ttc s")]
    lines += ["", "## By Scenario", "",
              "| " + " | ".join(scenario_header) + " |", "|" + "|".join(["---"] * len(scenario_header)) + "|"]
    for r in rows:
        by = r["metrics"].get("by_scenario", {})
        cells = [r["name"]]
        for name in SCENARIOS:
            scenario = by.get(name, {})
            cells += [
                fmt_cell(scenario.get("capture_rate"), "{:.3f}"),
                fmt_cell(scenario.get("collision_rate"), "{:.3f}"),
                fmt_cell(scenario.get("time_to_capture_mean_s"), "{:.2f}"),
            ]
        lines.append("| " + " | ".join(cells) + " |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Flat CSV: overall + per-scenario
    fields = ["method", "task", "episodes"] + [key for key, _, _, _ in METRIC_COLUMNS]
    for name in SCENARIOS:
        fields += [f"{name}_capture_rate", f"{name}_collision_rate", f"{name}_time_to_capture_mean_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            m = r["metrics"]
            row = {"method": r["name"], "task": r["task"], "episodes": m.get("num_episodes", 0)}
            for key, _, _, _ in METRIC_COLUMNS:
                row[key] = m.get(key, "")
            by = m.get("by_scenario", {})
            for name in SCENARIOS:
                scenario = by.get(name, {})
                row[f"{name}_capture_rate"] = scenario.get("capture_rate", "")
                row[f"{name}_collision_rate"] = scenario.get("collision_rate", "")
                row[f"{name}_time_to_capture_mean_s"] = scenario.get("time_to_capture_mean_s", "")
            w.writerow(row)
    print(f"[BENCH] wrote {csv_path}")
    return md_path


def seed_cell(values: list, fmt: str) -> str:
    """Mean +/- sample std over seeds; a metric a seed doesn't have (no captures, say) is left out."""
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    return fmt_cell(statistics.mean(values), fmt, statistics.stdev(values) if len(values) > 1 else None)


def write_seed_summary(root: str) -> str:
    """One row per method over the training seeds found under root, as markdown."""
    runs: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(root, "seed_*", "*", "metrics.json"))):
        runs.setdefault(os.path.basename(os.path.dirname(path)), []).append(read_metrics(path))
    if not runs:
        raise SystemExit(f"no seed_*/<method>/metrics.json under {root}")
    labels = [label for _, label, _, _ in METRIC_COLUMNS]
    lines = ["# Mean +/- std over training seeds", "", "| method | seeds | episodes per seed | " + " | ".join(labels) + " |",
             "|" + "|".join(["---"] * (len(labels) + 3)) + "|"]
    for method, seeds in runs.items():
        cells = [seed_cell([m.get(key) for m in seeds], fmt) for key, _, fmt, _ in METRIC_COLUMNS]
        lines.append(f"| {method} | {len(seeds)} | {seeds[0].get('num_episodes', 0)} | " + " | ".join(cells) + " |")
    scenarios = sorted({name for seeds in runs.values() for m in seeds for name in m.get("by_scenario", {})})
    lines += ["", "## By scenario", "", "| method | " + " | ".join(f"{n} {w}" for n in scenarios for w in ("cap", "coll")) + " |",
              "|" + "|".join(["---"] * (2 * len(scenarios) + 1)) + "|"]
    for method, seeds in runs.items():
        by = [m.get("by_scenario", {}) for m in seeds]
        cells = [seed_cell([b.get(n, {}).get(key) for b in by], "{:.3f}")
                 for n in scenarios for key in ("capture_rate", "collision_rate")]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    path = os.path.join(root, "comparison_seeds.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def plot_comparison(rows: list[dict], out_dir: str) -> list[str]:
    """Plot the safety-performance tradeoff and the time to capture of every method."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[BENCH] matplotlib unavailable ({exc}); skipping plots.")
        return []

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    import numpy as np
    from matplotlib.patches import Patch

    rows = [row for row in rows if int(row["metrics"].get("num_episodes", 0)) > 0]
    if not rows:
        return []
    colors = [plt.get_cmap("tab10")(i) for i in range(len(rows))]
    method_handles = [Patch(facecolor=color, label=row["name"]) for row, color in zip(rows, colors)]
    saved: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for method_idx, (row, color) in enumerate(zip(rows, colors)):
        metrics = row["metrics"]
        capture = 100.0 * float(metrics.get("capture_rate", 0.0))
        collision = 100.0 * float(metrics.get("collision_rate", 0.0))
        margin = metrics.get("min_safety_margin_mean")
        margin_std = metrics.get("min_safety_margin_std")
        ttc = metrics.get("time_to_capture_mean_s")
        # Method identity comes from the color legend; the short TTC tag alternates
        # above/below the marker so clustered methods stay readable.
        ttc_tag = "" if ttc is None else f"{float(ttc):.1f} s"
        label_kw = {
            "textcoords": "offset points",
            "fontsize": 8,
            "ha": "center",
            "color": "#404040",
            "xytext": (0, 9) if method_idx % 2 == 0 else (0, -16),
        }
        axes[0].scatter(collision, capture, s=90, color=color, edgecolors="black", linewidths=0.6)
        axes[0].annotate(ttc_tag, (collision, capture), **label_kw)
        if margin is not None:
            if margin_std is not None:
                # Spread of the per-episode minimum margin across the suite (mean +/- std).
                axes[1].errorbar(float(margin), capture, xerr=float(margin_std), fmt="none",
                                 ecolor=color, elinewidth=1.1, capsize=3, alpha=0.55, zorder=2)
            axes[1].scatter(float(margin), capture, s=90, color=color, edgecolors="black",
                            linewidths=0.6, zorder=3)
            axes[1].annotate(ttc_tag, (float(margin), capture), **label_kw)
    axes[0].set_xlabel("collision rate [%] (lower is safer)")
    axes[1].set_xlabel("min safety margin [m], mean +/- std over episodes (higher is safer)")
    for ax in axes:
        ax.set_ylabel("capture rate [%]")
        ax.set_ylim(-3.0, 103.0)
        ax.grid(True, alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_title("Capture versus collisions")
    axes[1].set_title("Capture versus clearance")
    fig.suptitle("Overall safety-performance tradeoff (marker tag: mean TTC)")
    fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=min(5, len(rows)), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    path = os.path.join(plot_dir, "overall_tradeoff.png")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    specs = [
        ("capture_rate", "Capture rate", "[%]", 100.0, None),
        ("collision_rate", "Collision rate", "[%]", 100.0, None),
        ("time_to_capture_mean_s", "Time to capture (mean +/- std)", "[s]", 1.0, "time_to_capture_std_s"),
    ]
    x = np.arange(len(SCENARIOS))
    width = 0.8 / len(rows)
    for method_idx, (row, color) in enumerate(zip(rows, colors)):
        offset = (method_idx - (len(rows) - 1) / 2) * width
        by_scenario = row["metrics"].get("by_scenario", {})
        for ax, (key, _title, _unit, scale, std_key) in zip(axes, specs):
            values = []
            stds = []
            for name in SCENARIOS:
                scenario = by_scenario.get(name, {}) or {}
                value = scenario.get(key)
                values.append(np.nan if value is None else float(value) * scale)
                std = scenario.get(std_key) if std_key else None
                stds.append(0.0 if std is None else float(std) * scale)
            yerr = stds if std_key and any(stds) else None
            ax.bar(x + offset, values, width, label=row["name"], color=color,
                   yerr=yerr, capsize=2, error_kw={"lw": 0.9, "ecolor": "#404040", "alpha": 0.85})
    for ax, (_key, title, unit, _scale, _std_key) in zip(axes, specs):
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("_", "\n") for name in SCENARIOS])
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(True, axis="y", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylim(0.0, 100.0)
    axes[1].set_ylim(0.0, 100.0)
    fig.suptitle("Performance by scenario", y=0.99)
    fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=min(5, len(rows)), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    path = os.path.join(plot_dir, "performance_by_scenario.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    return saved


def main() -> None:
    args = parse_args()
    if args.across_seeds:
        print(f"[BENCH] wrote {write_seed_summary(os.path.abspath(args.output_dir))}")
        return
    if args.ood_room:
        arenas = [True]
    elif args.large_only:
        arenas = [False]
    else:
        arenas = [False, True]
    for ood_room in arenas:
        args.ood_room = ood_room
        run_benchmark(args, both=len(arenas) == 2)


def run_benchmark(args, both: bool) -> None:
    exp_files = discover_experiments(args.experiments)
    output_dir = args.output_dir or ("outputs/benchmark_ood_room" if args.ood_room else "outputs/benchmark")
    if both and args.output_dir and args.ood_room:
        output_dir = f"{args.output_dir.rstrip('/')}_ood_room"
    out_root = output_dir if os.path.isabs(output_dir) else os.path.join(REPO_ROOT, output_dir)
    os.makedirs(out_root, exist_ok=True)

    rows = []
    for exp_file in exp_files:
        exp = load_yaml(exp_file)
        name = exp.get("name") or os.path.splitext(os.path.basename(exp_file))[0]
        task = exp.get("task", "?")
        if args.ood_room:
            task = ROOM_TASK_FOR_LARGE_TASK.get(task, task)
        print(f"[BENCH] === {name} ({task}) ===")
        out_dir = os.path.join(out_root, name)
        if not args.aggregate_only:
            run_experiment(exp_file, out_dir, args)
        metrics = read_metrics(os.path.join(out_dir, "metrics.json"))
        if metrics is None:
            print(f"[BENCH] WARNING: no metrics.json for '{name}' - leaving its row blank.")
            metrics = {"num_episodes": 0}
        rows.append({"name": name, "task": task, "metrics": metrics})

    md_path = write_tables_and_csv(rows, out_root)
    print(f"[BENCH] wrote {md_path}")
    for p in plot_comparison(rows, out_root):
        print(f"[BENCH] wrote {p}")

    with open(md_path) as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
