"""
run_multirun.py

Runs run_seq_replay_v3.py across N_RUNS random seeds, then aggregates
results into a summary table with mean ± std for each metric.

Usage (submit via SLURM using the same launch.sh, just swap the script):
    python run_multirun.py

Or directly:
    python run_multirun.py --n_runs 5 --base_seed 0

Each run gets its own subdirectory under outputs/runs/run_{seed}/
so figures and per-run metrics.txt are all preserved separately.

The summary is written to:
    outputs/multirun_summary.txt
    outputs/multirun_summary.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_DIR  = "/home/ao1g22/comp6228/irp"
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
RUNS_DIR  = os.path.join(OUT_DIR,  "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# Seeds to use — deterministic, spread across space
DEFAULT_SEEDS = [0, 7, 42, 123, 999]

METRIC_KEYS = [
    "iid_next_frame_mse",
    "seq_next_frame_mse",
    "seq_img_pred_mse",
    "iid_recon_mse",
    "seq_recon_mse",
    "iid_schema_distortion",
    "seq_schema_distortion",
]

METRIC_LABELS = {
    "iid_next_frame_mse":    "Next-frame MSE — IID (Ridge probe)",
    "seq_next_frame_mse":    "Next-frame MSE — Sequential (Ridge probe)",
    "seq_img_pred_mse":      "Image prediction MSE — Sequential only",
    "iid_recon_mse":         "Reconstruction MSE — IID",
    "seq_recon_mse":         "Reconstruction MSE — Sequential",
    "iid_schema_distortion": "Schema distortion ratio — IID",
    "seq_schema_distortion": "Schema distortion ratio — Sequential",
}


# ─────────────────────────────────────────────
# Import the experiment
# ─────────────────────────────────────────────
# Add the irp directory to path so we can import run_seq_replay_v3
sys.path.insert(0, BASE_DIR)
from run_seq_replay_v3 import main as run_experiment


# ─────────────────────────────────────────────
# Aggregation and reporting
# ─────────────────────────────────────────────
def aggregate(all_results):
    """Compute mean and std across runs for each metric."""
    summary = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in all_results]
        summary[key] = {
            "values": vals,
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals)),
            "min":    float(np.min(vals)),
            "max":    float(np.max(vals)),
        }
    return summary


def write_summary_txt(summary, seeds, path):
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Can Machines Dream? — Multi-Run Results Summary\n")
        f.write(f"Seeds: {seeds}  |  N runs: {len(seeds)}\n")
        f.write("=" * 70 + "\n\n")

        f.write("PRIMARY METRIC: Next-frame prediction MSE (identical Ridge probe)\n")
        f.write("-" * 70 + "\n")
        iid = summary["iid_next_frame_mse"]
        seq = summary["seq_next_frame_mse"]
        f.write(f"  IID       mean ± std : {iid['mean']:.4f} ± {iid['std']:.4f}"
                f"  [min={iid['min']:.4f}, max={iid['max']:.4f}]\n")
        f.write(f"  Sequential mean ± std : {seq['mean']:.4f} ± {seq['std']:.4f}"
                f"  [min={seq['min']:.4f}, max={seq['max']:.4f}]\n")
        delta_mean = iid['mean'] - seq['mean']
        pct = 100 * delta_mean / iid['mean']
        f.write(f"  Delta (IID - Seq) mean: {delta_mean:.4f}  ({pct:.1f}% reduction)\n")

        # Check consistency: did sequential win every run?
        per_run_iid = summary["iid_next_frame_mse"]["values"]
        per_run_seq = summary["seq_next_frame_mse"]["values"]
        wins = sum(s < i for s, i in zip(per_run_seq, per_run_iid))
        f.write(f"  Sequential < IID in {wins}/{len(seeds)} runs\n\n")

        f.write("PER-RUN BREAKDOWN (next-frame MSE)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Seed':>6}  {'IID':>10}  {'Sequential':>12}  {'Delta':>10}  {'Seq wins?':>10}\n")
        for seed, i, s in zip(seeds, per_run_iid, per_run_seq):
            d = i - s
            win = "✓" if s < i else "✗"
            f.write(f"  {seed:>6}  {i:>10.4f}  {s:>12.4f}  {d:>10.4f}  {win:>10}\n")
        f.write("\n")

        f.write("ALL METRICS — MEAN ± STD\n")
        f.write("-" * 70 + "\n")
        for key in METRIC_KEYS:
            s = summary[key]
            label = METRIC_LABELS[key]
            f.write(f"  {label}\n")
            f.write(f"    {s['mean']:.6f} ± {s['std']:.6f}"
                    f"  [min={s['min']:.6f}, max={s['max']:.6f}]\n")
        f.write("\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        if wins == len(seeds):
            f.write("  Sequential replay produced lower next-frame prediction MSE\n")
            f.write(f"  in ALL {len(seeds)} runs. The result is consistent across seeds.\n")
            f.write(f"  Mean improvement: {pct:.1f}% reduction in MSE.\n")
            f.write("  This robustly supports the claim that sequential predictive\n")
            f.write("  replay consolidates temporally ordered structure better than\n")
            f.write("  unordered IID replay, as measured by an identical linear probe.\n")
        elif wins > len(seeds) // 2:
            f.write(f"  Sequential replay won {wins}/{len(seeds)} runs.\n")
            f.write("  Result is directionally consistent but not perfectly stable.\n")
            f.write("  Consider increasing STUDENT_EPOCHS or kl_weight.\n")
        else:
            f.write(f"  Sequential replay only won {wins}/{len(seeds)} runs.\n")
            f.write("  Result is not stable. Review architecture and training config.\n")


def save_summary_plot(summary, seeds, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Can Machines Dream? — Multi-Run Results\n"
        f"(N={len(seeds)} seeds, mean ± std)",
        fontsize=11
    )

    # Panel 1: Primary metric — next-frame MSE per run + mean
    ax = axes[0]
    iid_vals = summary["iid_next_frame_mse"]["values"]
    seq_vals = summary["seq_next_frame_mse"]["values"]
    x = np.arange(len(seeds))
    ax.bar(x - 0.2, iid_vals, 0.35, label="IID",        color="steelblue", alpha=0.7)
    ax.bar(x + 0.2, seq_vals, 0.35, label="Sequential",  color="coral",     alpha=0.7)
    ax.axhline(summary["iid_next_frame_mse"]["mean"], color="steelblue",
               linestyle="--", linewidth=1.2, alpha=0.8, label="IID mean")
    ax.axhline(summary["seq_next_frame_mse"]["mean"], color="coral",
               linestyle="--", linewidth=1.2, alpha=0.8, label="Seq mean")
    ax.set_xticks(x)
    ax.set_xticklabels([f"s={s}" for s in seeds], fontsize=8)
    ax.set_title("Next-frame MSE\n(primary, Ridge probe)", fontsize=9)
    ax.set_ylabel("MSE")
    ax.legend(fontsize=7)

    # Panel 2: Mean ± std bar chart for primary + schema distortion
    ax = axes[1]
    metrics_to_show = [
        ("iid_next_frame_mse",    "IID NF-MSE",    "steelblue"),
        ("seq_next_frame_mse",    "Seq NF-MSE",    "coral"),
        ("iid_schema_distortion", "IID Schema",    "steelblue"),
        ("seq_schema_distortion", "Seq Schema",    "coral"),
    ]
    labels_  = [m[1] for m in metrics_to_show]
    means_   = [summary[m[0]]["mean"] for m in metrics_to_show]
    stds_    = [summary[m[0]]["std"]  for m in metrics_to_show]
    colours_ = [m[2] for m in metrics_to_show]
    xpos = np.arange(len(labels_))
    bars = ax.bar(xpos, means_, yerr=stds_, color=colours_, alpha=0.75,
                  capsize=5, error_kw={"linewidth": 1.5})
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels_, fontsize=7, rotation=20, ha="right")
    ax.set_title("Mean ± std\n(primary + schema)", fontsize=9)
    ax.set_ylabel("Score")
    for bar, mean in zip(bars, means_):
        ax.annotate(f"{mean:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7)

    # Panel 3: Reconstruction MSE mean ± std
    ax = axes[2]
    rc_metrics = [
        ("iid_recon_mse", "IID Recon",  "steelblue"),
        ("seq_recon_mse", "Seq Recon",  "coral"),
    ]
    labels_r  = [m[1] for m in rc_metrics]
    means_r   = [summary[m[0]]["mean"] for m in rc_metrics]
    stds_r    = [summary[m[0]]["std"]  for m in rc_metrics]
    colours_r = [m[2] for m in rc_metrics]
    xr = np.arange(len(labels_r))
    bars_r = ax.bar(xr, means_r, yerr=stds_r, color=colours_r, alpha=0.75,
                    capsize=5, error_kw={"linewidth": 1.5}, width=0.4)
    ax.set_xticks(xr)
    ax.set_xticklabels(labels_r, fontsize=9)
    ax.set_title("Reconstruction MSE\n(secondary)", fontsize=9)
    ax.set_ylabel("MSE")
    for bar, mean in zip(bars_r, means_r):
        ax.annotate(f"{mean:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Summary plot saved to {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs",    type=int, default=5,
                        help="Number of runs (uses first n seeds from DEFAULT_SEEDS)")
    parser.add_argument("--base_seed", type=int, default=None,
                        help="If set, use seeds base_seed+0, base_seed+1, ...")
    args = parser.parse_args()

    if args.base_seed is not None:
        seeds = list(range(args.base_seed, args.base_seed + args.n_runs))
    else:
        seeds = DEFAULT_SEEDS[:args.n_runs]

    print("=" * 70)
    print("Can Machines Dream? — Multi-Run Experiment")
    print(f"Running {len(seeds)} seeds: {seeds}")
    print("=" * 70)

    all_results = []
    for i, seed in enumerate(seeds):
        run_dir = os.path.join(RUNS_DIR, f"run_{seed}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"RUN {i+1}/{len(seeds)}  seed={seed}  → {run_dir}")
        print(f"{'='*70}")
        results = run_experiment(seed=seed, run_dir=run_dir)
        all_results.append(results)
        print(f"  Run {i+1} complete.")

    print(f"\n{'='*70}")
    print("All runs complete. Aggregating...")

    summary = aggregate(all_results)

    summary_txt = os.path.join(OUT_DIR, "multirun_summary.txt")
    write_summary_txt(summary, seeds, summary_txt)
    print(f"  Summary written to {summary_txt}")

    # Print summary to stdout too
    with open(summary_txt) as f:
        print(f.read())

    summary_png = os.path.join(OUT_DIR, "multirun_summary.png")
    save_summary_plot(summary, seeds, summary_png)


if __name__ == "__main__":
    main()