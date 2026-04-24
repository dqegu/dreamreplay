"""
run_all.py
─────────────────────────────────────────────────────────────────────────────
Main entry point. Runs the full training and evaluation pipeline across
multiple seeds and aggregates results.

Usage:
  python run_all.py [--seeds 0 7 42] [--force_teacher] [--force_students]
"""

import argparse
import json
import os
import numpy as np

from config import OUT_DIR, SEEDS, PROBE_FACTORS, CHECKPOINT_EPOCHS
from data import load_shapes3d
from training import run_training_pipeline
import experiment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",          nargs="*", type=int, default=SEEDS)
    p.add_argument("--force_teacher",  action="store_true")
    p.add_argument("--force_students", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading Shapes3D...")
    (train_imgs, train_seqs, train_labels,
     test_imgs,  test_seqs,  test_labels) = load_shapes3d()

    all_results = []

    for seed in args.seeds:
        print(f"\n{'#'*70}")
        print(f"#  SEED {seed}")
        print(f"{'#'*70}")

        models = run_training_pipeline(
            train_imgs, train_seqs, seed=seed,
            force_teacher=args.force_teacher,
            force_students=args.force_students)

        print("\nRunning experiments...")
        res = experiment.run(
            models,
            train_imgs, train_seqs, train_labels,
            test_imgs,  test_seqs,  test_labels,
            seed=seed)
        all_results.append(res)

    # Aggregate across seeds
    agg = _aggregate(all_results)
    agg_path = os.path.join(OUT_DIR, "multirun_summary.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\nAggregated results saved to {agg_path}")
    _print_aggregate(agg)


def _agg_scalar(all_results, *keys):
    vals = []
    for r in all_results:
        v = r
        for k in keys:
            v = v[k]
        vals.append(float(v))
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def _aggregate(all_results):
    agg = {"recall": {}}

    for cond in ("sequential", "shuffled", "iid"):
        agg["recall"][cond] = {}
        for metric in ("pixel_mse", "pixel_mse_std",
                        "latent_mse", "latent_mse_std"):
            agg["recall"][cond][metric] = _agg_scalar(
                all_results, cond, metric)

    for key in ("seq_wins_vs_shuffled", "seq_wins_vs_iid",
                "shuf_wins_vs_iid", "n_seqs"):
        agg["recall"][key] = _agg_scalar(all_results, key)

    return agg


def _print_aggregate(agg):
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS  (mean ± std across seeds)")
    print(f"{'='*70}")

    r = agg["recall"]
    print(f"\n  Partial-cue recall — gap frames 5–9 masked")
    print(f"  All three students: identical sequence VAE architecture")
    print(f"  Only replay structure differs\n")
    print(f"  {'Condition':<14}  {'Pixel MSE':>16}  {'Latent MSE':>16}")
    print("  " + "─" * 52)
    for cond in ("sequential", "shuffled", "iid"):
        print(f"  {cond:<14}  "
              f"{r[cond]['pixel_mse']['mean']:>8.5f}±"
              f"{r[cond]['pixel_mse']['std']:.5f}  "
              f"{r[cond]['latent_mse']['mean']:>8.5f}±"
              f"{r[cond]['latent_mse']['std']:.5f}")
    n = r['n_seqs']['mean']
    print(f"\n  Win rates on latent MSE (lower = better consolidation):")
    print(f"    Seq vs Shuffled: "
          f"{r['seq_wins_vs_shuffled']['mean']/n*100:.1f}% seq wins")
    print(f"    Seq vs IID:      "
          f"{r['seq_wins_vs_iid']['mean']/n*100:.1f}% seq wins")
    print(f"    Shuf vs IID:     "
          f"{r['shuf_wins_vs_iid']['mean']/n*100:.1f}% shuf wins")
    print(f"    Seq vs IID:      {adv_iid:+.3f}  "
          f"({'SEQ MORE CANONICAL ✓' if adv_iid < 0 else 'IID more canonical'})")


if __name__ == "__main__":
    main()