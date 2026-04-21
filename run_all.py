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
    agg = {"exp1": {}, "exp2": {}, "exp3": {}, "exp4": {}}

    # ── Exp 1 ─────────────────────────────────────────────────────────────
    for cond in ("sequential", "shuffled"):
        agg["exp1"][cond] = {}
        for metric in ("pixel_mse", "pixel_mse_std",
                        "latent_mse", "latent_mse_std"):
            agg["exp1"][cond][metric] = _agg_scalar(
                all_results, "exp1", cond, metric)
    for key in ("seq_wins_pixel", "seq_wins_latent", "n_seqs"):
        agg["exp1"][key] = _agg_scalar(all_results, "exp1", key)

    # ── Exp 2 ─────────────────────────────────────────────────────────────
    probe_factors = [f for f in ("shape", "object_hue", "floor_hue",
                                  "wall_hue", "scale")]
    for cond in ("sequential", "shuffled"):
        agg["exp2"][cond] = {}
        all_epoch_keys = ([f"epoch_{e}" for e in CHECKPOINT_EPOCHS] + ["final"])
        for ep_key in all_epoch_keys:
            ep_results = []
            for r in all_results:
                if ep_key in r["exp2"][cond]:
                    accs = [r["exp2"][cond][ep_key][f]["acc"]
                            for f in probe_factors
                            if f in r["exp2"][cond][ep_key]]
                    if accs:
                        ep_results.append(np.mean(accs))
            if ep_results:
                agg["exp2"][cond][ep_key] = {
                    "mean_acc": float(np.mean(ep_results)),
                    "std_acc":  float(np.std(ep_results)),
                }

    # ── Exp 3 ─────────────────────────────────────────────────────────────
    for cond in ("sequential", "shuffled"):
        agg["exp3"][cond] = {}
        for metric in ("smoothness_mean", "smoothness_std",
                        "validity_mean", "validity_std"):
            agg["exp3"][cond][metric] = _agg_scalar(
                all_results, "exp3", cond, metric)

    # ── Exp 4 ─────────────────────────────────────────────────────────────
    agg["exp4"]["canonical_mad"] = {"mean": 0.0, "std": 0.0}
    agg["exp4"]["atypical_mad"]  = _agg_scalar(
        all_results, "exp4", "atypical_mad")
    for cond in ("sequential", "shuffled"):
        agg["exp4"][cond] = {}
        for metric in ("recon_mad", "recon_mad_std",
                        "schema_pull", "wins_vs_atyp"):
            agg["exp4"][cond][metric] = _agg_scalar(
                all_results, "exp4", cond, metric)
    agg["exp4"]["seq_vs_shuffled"] = _agg_scalar(
        all_results, "exp4", "seq_vs_shuffled")
    agg["exp4"]["seq_wins_vs_shuf"] = _agg_scalar(
        all_results, "exp4", "seq_wins_vs_shuf")

    return agg


def _print_aggregate(agg):
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS  (mean ± std across seeds)")
    print(f"{'='*70}")

    e1 = agg["exp1"]
    print(f"\n  Exp 1 — Partial-cue recall (gap frames 5–9 masked)")
    for cond in ("sequential", "shuffled"):
        print(f"    [{cond}]  "
              f"pixel MSE={e1[cond]['pixel_mse']['mean']:.5f}±"
              f"{e1[cond]['pixel_mse']['std']:.5f}  "
              f"latent MSE={e1[cond]['latent_mse']['mean']:.5f}±"
              f"{e1[cond]['latent_mse']['std']:.5f}")
    win_p = e1['seq_wins_pixel']['mean'] / e1['n_seqs']['mean'] * 100
    win_l = e1['seq_wins_latent']['mean'] / e1['n_seqs']['mean'] * 100
    print(f"    Seq wins: pixel={win_p:.1f}%  latent={win_l:.1f}%")

    e2 = agg["exp2"]
    print(f"\n  Exp 2 — Semanticisation over consolidation")
    for cond in ("sequential", "shuffled"):
        print(f"    [{cond}]")
        all_keys = ([f"epoch_{e}" for e in CHECKPOINT_EPOCHS] + ["final"])
        for ep_key in all_keys:
            if ep_key in e2[cond]:
                d = e2[cond][ep_key]
                print(f"      {ep_key}: "
                      f"mean acc={d['mean_acc']:.3f}±{d['std_acc']:.3f}")

    e3 = agg["exp3"]
    print(f"\n  Exp 3 — Imagination (interpolation smoothness)")
    for cond in ("sequential", "shuffled"):
        print(f"    [{cond}]  "
              f"smoothness={e3[cond]['smoothness_mean']['mean']:.3f}±"
              f"{e3[cond]['smoothness_mean']['std']:.3f}  "
              f"validity={e3[cond]['validity_mean']['mean']:.3f}±"
              f"{e3[cond]['validity_mean']['std']:.3f}")

    e4 = agg["exp4"]
    print(f"\n  Exp 4 — Schema distortion")
    print(f"    Canonical MAD: 0.000")
    print(f"    Atypical MAD:  {e4['atypical_mad']['mean']:.3f}±"
          f"{e4['atypical_mad']['std']:.3f}")
    for cond in ("sequential", "shuffled"):
        pull = e4[cond]['schema_pull']['mean']
        print(f"    [{cond}]  "
              f"recon MAD={e4[cond]['recon_mad']['mean']:.3f}±"
              f"{e4[cond]['recon_mad']['std']:.3f}  "
              f"schema pull={pull:+.3f}  "
              f"({'toward canonical ✓' if pull < 0 else 'away from canonical'})")
    adv = e4['seq_vs_shuffled']['mean']
    print(f"    Seq vs shuffled: {adv:+.3f}  "
          f"({'SEQ MORE CANONICAL ✓' if adv < 0 else 'SHUFFLED more canonical'})")


if __name__ == "__main__":
    main()