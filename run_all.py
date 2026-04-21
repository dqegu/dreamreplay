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


def _agg_scalar(all_results, *keys):
    """Extract a scalar from nested keys across seeds, return mean±std dict."""
    vals = []
    for r in all_results:
        v = r
        for k in keys:
            v = v[k]
        vals.append(float(v))
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def _aggregate(all_results):
    """Mean ± std across seeds for every metric."""
    agg = {"factor_probes": {}, "exp1a": {}, "exp1b": {}, "exp2": {}, "exp3": {}}
    students = ["teacher", "iid_matched", "iid_mhn", "sequential"]
    factors  = list(PROBE_FACTORS.keys())

    # ── Factor probes ─────────────────────────────────────────────────────────
    for student in students:
        agg["factor_probes"][student] = {"recon_mse": {}, "probes": {}}

        agg["factor_probes"][student]["recon_mse"] = _agg_scalar(
            all_results, "factor_probes", student, "recon_mse")

        for factor in factors:
            agg["factor_probes"][student]["probes"][factor] = {}
            for key in ("frame_acc", "frame_r2", "episode_acc", "episode_r2"):
                vals = [r["factor_probes"][student]["probes"]
                        .get(factor, {}).get(key, float("nan"))
                        for r in all_results]
                vals = [v for v in vals if not np.isnan(v)]
                agg["factor_probes"][student]["probes"][factor][key] = {
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std":  float(np.std(vals))  if vals else float("nan"),
                }

    # ── Exp 1A: probe-based completion ───────────────────────────────────────
    agg["exp1a"] = {"sequential": {}, "iid_matched": {}}
    for student in ("sequential", "iid_matched"):
        for key in ("latent_mse", "latent_mse_std"):
            agg["exp1a"][student][key] = _agg_scalar(
                all_results, "exp1a_probe", student, key)
    agg["exp1a"]["seq_vs_iid"] = _agg_scalar(
        all_results, "exp1a_probe", "seq_vs_iid")

    # ── Exp 1B: native generative completion ──────────────────────────────────
    agg["exp1b"] = {}
    for key in ("seq_mse", "seq_mse_std", "iid_mse", "iid_mse_std",
                "seq_latent_mse", "seq_latent_std",
                "iid_latent_mse", "iid_latent_std"):
        agg["exp1b"][key] = _agg_scalar(all_results, "exp1b_generative", key)
    for pct_key, raw_key in (("seq_wins_pixel_pct", "seq_wins_pixel"),
                              ("seq_wins_latent_pct", "seq_wins_latent")):
        agg["exp1b"][pct_key] = {
            "mean": float(np.mean([r["exp1b_generative"][raw_key] /
                                    r["exp1b_generative"]["n_seqs"]
                                    for r in all_results])),
            "std":  float(np.std([r["exp1b_generative"][raw_key] /
                                   r["exp1b_generative"]["n_seqs"]
                                   for r in all_results])),
        }

    # ── Exp 2: schema distortion ──────────────────────────────────────────────
    for key in ("canonical_mad", "atypical_mad", "seq_mad", "iid_mad",
                "seq_schema_pull", "iid_schema_pull", "seq_vs_iid"):
        agg["exp2"][key] = _agg_scalar(all_results, "exp2_schema", key)
    agg["exp2"]["seq_wins_pct"] = {
        "mean": float(np.mean([r["exp2_schema"]["seq_wins"] /
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
    main()r["exp2_schema"]["n_seqs"]
                                for r in all_results])),
        "std":  float(np.std([r["exp2_schema"]["seq_wins"] /
                               r["exp2_schema"]["n_seqs"]
                               for r in all_results])),
    }

    # ── Exp 3: temporal gist ──────────────────────────────────────────────────
    for student in students:
        agg["exp3"][student] = {}
        for key in ("phase_acc", "phase_r2"):
            agg["exp3"][student][key] = _agg_scalar(
                all_results, "exp3_gist", student, key)

    return agg


def _print_aggregate(agg):
    factors = list(PROBE_FACTORS.keys())
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS  (mean ± std across seeds)")
    print(f"{'='*70}")

    # ── Factor probes ─────────────────────────────────────────────────────────
    for rep in ("frame", "episode"):
        print(f"\n  [{rep} representation — probe accuracy on test set]")
        print(f"  {'Factor':<14}  {'Teacher':>14}  {'IID-match':>14}  "
              f"{'IID-MHN':>14}  {'Seq':>14}  {'Winner':>10}")
        print("  " + "─" * 84)
        for f in factors:
            def _fmt(name):
                d = (agg["factor_probes"][name]["probes"]
                     .get(f, {}).get(f"{rep}_acc", {}))
                m = d.get("mean", float("nan"))
                s = d.get("std",  float("nan"))
                return f"{m:.3f}±{s:.3f}"
            t   = _fmt("teacher")
            im  = _fmt("iid_matched")
            mhn = _fmt("iid_mhn")
            sq  = _fmt("sequential")
            im_m  = (agg["factor_probes"]["iid_matched"]["probes"]
                     .get(f, {}).get(f"{rep}_acc", {}).get("mean", 0))
            seq_m = (agg["factor_probes"]["sequential"]["probes"]
                     .get(f, {}).get(f"{rep}_acc", {}).get("mean", 0))
            winner = "SEQ ✓" if seq_m > im_m else "IID-match"
            print(f"  {f:<14}  {t:>14}  {im:>14}  {mhn:>14}  {sq:>14}  {winner:>10}")

    print(f"\n  Reconstruction MSE:")
    for name in ("teacher", "iid_matched", "iid_mhn", "sequential"):
        d = agg["factor_probes"][name]["recon_mse"]
        print(f"    {name:<14}: {d['mean']:.5f} ± {d['std']:.5f}")

    # ── Consolidation experiments ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONSOLIDATION EXPERIMENTS  (mean ± std across seeds)")
    print(f"{'='*70}")

    e1a = agg["exp1a"]
    e1b = agg["exp1b"]
    print(f"\n  Exp 1A — Probe-based completion (fully fair)")
    print(f"    Sequential latent MSE:  {e1a['sequential']['latent_mse']['mean']:.5f} "
          f"± {e1a['sequential']['latent_mse']['std']:.5f}")
    print(f"    IID-matched latent MSE: {e1a['iid_matched']['latent_mse']['mean']:.5f} "
          f"± {e1a['iid_matched']['latent_mse']['std']:.5f}")
    adv = e1a['seq_vs_iid']['mean']
    print(f"    Seq vs IID: {adv:+.5f} ± {e1a['seq_vs_iid']['std']:.5f} "
          f"({'SEQ ✓' if adv < 0 else 'IID'})")

    print(f"\n  Exp 1B — Native generative completion (behavioural)")
    print(f"    Pixel MSE:  Seq={e1b['seq_mse']['mean']:.5f}  "
          f"IID={e1b['iid_mse']['mean']:.5f}  "
          f"Seq wins {e1b['seq_wins_pixel_pct']['mean']*100:.1f}%")
    print(f"    Latent MSE: Seq={e1b['seq_latent_mse']['mean']:.5f}  "
          f"IID={e1b['iid_latent_mse']['mean']:.5f}  "
          f"Seq wins {e1b['seq_wins_latent_pct']['mean']*100:.1f}%")

    e2 = agg["exp2"]
    print(f"\n  Exp 2 — Schema distortion (MAD from canonical sweep)")
    print(f"    Canonical MAD:     {e2['canonical_mad']['mean']:.3f}")
    print(f"    Atypical input:    {e2['atypical_mad']['mean']:.3f} ± {e2['atypical_mad']['std']:.3f}")
    print(f"    Seq completion:    {e2['seq_mad']['mean']:.3f} ± {e2['seq_mad']['std']:.3f}")
    print(f"    IID completion:    {e2['iid_mad']['mean']:.3f} ± {e2['iid_mad']['std']:.3f}")
    print(f"    Seq vs IID:        {e2['seq_vs_iid']['mean']:+.3f} ± {e2['seq_vs_iid']['std']:.3f}  "
          f"({'SEQ MORE CANONICAL ✓' if e2['seq_vs_iid']['mean'] < 0 else 'IID more canonical'})")
    win_pct2 = e2['seq_wins_pct']['mean'] * 100
    print(f"    Seq wins {win_pct2:.1f}% of sequences on average")

    e3 = agg["exp3"]
    print(f"\n  Exp 3 — Temporal-gist semanticization (phase decoding, chance=0.33)")
    print(f"  {'Student':<14}  {'Phase acc (mean±std)':>22}")
    print("  " + "─" * 40)
    for name in ("teacher", "iid_matched", "iid_mhn", "sequential"):
        d = e3[name]["phase_acc"]
        print(f"  {name:<14}  {d['mean']:.3f} ± {d['std']:.3f}")
    seq_acc  = e3["sequential"]["phase_acc"]["mean"]
    iid_acc  = e3["iid_matched"]["phase_acc"]["mean"]
    print(f"    {'SEQ ✓' if seq_acc > iid_acc else 'IID'} on phase decoding")


if __name__ == "__main__":
    main()