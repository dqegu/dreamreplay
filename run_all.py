"""
Main entry point.

Runs: teacher training → both student trainings → linear probe evaluation.
Repeated over multiple seeds; results aggregated and saved.

Usage
  python run_all.py [--seeds 0 7 42] [--n_train 4000] [--n_test 1000]
                    [--force_teacher] [--force_students]
"""

import argparse
import json
import os
import numpy as np

from config import OUT_DIR, N_TRAIN_GROUPS, N_TEST_GROUPS, SEEDS, PROBE_FACTORS
from data import load_shapes3d
from training import run_training_pipeline
import experiment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",          nargs="*", type=int, default=SEEDS)
    p.add_argument("--n_train",        type=int,  default=N_TRAIN_GROUPS)
    p.add_argument("--n_test",         type=int,  default=N_TEST_GROUPS)
    p.add_argument("--force_teacher",  action="store_true")
    p.add_argument("--force_students", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load data (streaming; done once, reused across seeds) ─────────────────
    print("Loading Shapes3D...")
    (train_imgs, train_seqs, train_labels,
     test_imgs,  test_seqs,  test_labels) = load_shapes3d(
        n_train=args.n_train, n_test=args.n_test, seed=0)

    all_results = []

    for seed in args.seeds:
        print(f"\n{'#'*70}")
        print(f"#  SEED {seed}")
        print(f"{'#'*70}")

        models = run_training_pipeline(
            train_imgs, train_seqs, seed=seed,
            force_teacher=args.force_teacher,
            force_students=args.force_students,
        )

        print("\nRunning probe evaluation...")
        res = experiment.run(
            models,
            train_imgs, train_seqs, train_labels,
            test_imgs,  test_seqs,  test_labels,
            seed=seed,
        )
        all_results.append(res)

    # ── Aggregate across seeds ────────────────────────────────────────────────
    agg = _aggregate(all_results)
    agg_path = os.path.join(OUT_DIR, "multirun_summary.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregated results saved to {agg_path}")
    _print_aggregate(agg)


def _aggregate(all_results):
    """Mean ± std across seeds for every metric."""
    agg = {}
    students = ["teacher", "iid_matched", "iid_mhn", "sequential"]
    factors  = list(PROBE_FACTORS.keys())

    for student in students:
        agg[student] = {"recon_mse": {}, "probes": {}}

        recon_vals = [r[student]["recon_mse"] for r in all_results]
        agg[student]["recon_mse"] = {
            "mean": float(np.mean(recon_vals)),
            "std":  float(np.std(recon_vals)),
        }

        for factor in factors:
            agg[student]["probes"][factor] = {}
            for key in ("frame_acc", "frame_r2", "episode_acc", "episode_r2"):
                vals = [r[student]["probes"].get(factor, {}).get(key, float("nan"))
                        for r in all_results]
                vals = [v for v in vals if not np.isnan(v)]
                agg[student]["probes"][factor][key] = {
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std":  float(np.std(vals))  if vals else float("nan"),
                }
    return agg


def _print_aggregate(agg):
    factors = list(PROBE_FACTORS.keys())
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS  (mean ± std across seeds)")
    print(f"{'='*70}")
    for rep in ("frame", "episode"):
        print(f"\n  [{rep} representation — probe accuracy on test set]")
        print(f"  {'Factor':<14}  {'Teacher':>14}  {'IID-match':>14}  {'IID-MHN':>14}  {'Seq':>14}  {'Winner':>10}")
        print("  " + "─" * 82)
        for f in factors:
            def _fmt(name):
                d = agg[name]["probes"].get(f, {}).get(f"{rep}_acc", {})
                m, s = d.get("mean", float("nan")), d.get("std", float("nan"))
                return f"{m:.3f}±{s:.3f}"
            t   = _fmt("teacher")
            im  = _fmt("iid_matched")
            mhn = _fmt("iid_mhn")
            s   = _fmt("sequential")
            im_m  = agg["iid_matched"]["probes"].get(f, {}).get(f"{rep}_acc", {}).get("mean", 0)
            seq_m = agg["sequential"]["probes"].get(f, {}).get(f"{rep}_acc", {}).get("mean", 0)
            winner = "SEQ ✓" if seq_m > im_m else "IID-match"
            print(f"  {f:<14}  {t:>14}  {im:>14}  {mhn:>14}  {s:>14}  {winner:>10}")

    print(f"\n  Reconstruction MSE (test images):")
    for name in ("teacher", "iid_matched", "iid_mhn", "sequential"):
        d = agg[name]["recon_mse"]
        print(f"    {name:<14}: {d['mean']:.5f} ± {d['std']:.5f}")


if __name__ == "__main__":
    main()
