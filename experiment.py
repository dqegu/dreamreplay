"""
Experiment: Does sequential replay produce better general representations?
══════════════════════════════════════════════════════════════════════════

Hypothesis
──────────
A predictive (next-frame) objective forces the encoder to disentangle what
changes across time (orientation) from what stays constant (shape, colour,
scale).  An IID reconstruction objective has no such pressure.  Therefore
Student B (sequential) should produce more structured latent representations
even on static semantic factors it was never explicitly trained to decode.

This maps directly onto Spens & Burgess (2024): their claim is that
sequential hippocampal replay consolidates memories more effectively than
random replay.  Here we test whether the cortical representation after
consolidation (the student latent space) carries more information about
the original event's semantic content.

Design
──────
• Teacher VAE        — trained directly on Shapes3D frames (upper bound)
• IID-matched student — trains on the SAME decoded frames as sequential,
                        pooled and shuffled (primary fair comparison):
                        same pixels, different order, reconstruction objective
• IID-MHN student    — trains on MHN-retrieved frames (Spens & Burgess model);
                        different frame distribution — secondary reference only
• Sequential student  — trains on K/V-Hopfield chains, ordered pairs,
                         prediction objective

The primary comparison is IID-matched vs Sequential: identical pixels,
only ordering and objective differ.  IID-MHN is shown alongside for
reference to the original Spens & Burgess setup.

Evaluation (linear probes on held-out test groups)
──────────────────────────────────────────────────
For each held-out test group we extract two representations:

  single-frame   : z_mean of the middle orientation frame (frame 7)
  episode-mean   : mean of z_means across all 15 orientation frames

We fit a logistic regression probe on TRAINING representations and
report accuracy on TEST representations (primary metric — disjoint
groups, no overlap with student training data).

Probe factors:  shape, floor_hue, wall_hue, object_hue, scale, orientation.

Including orientation is deliberate: if sequential training improves
orientation decoding (expected — it's what sequences vary along) AND
also improves static factor decoding (non-tautological), that double
pattern is the core finding.

Additionally we report:
  • Reconstruction MSE   (test images → encode → decode → pixel MSE)
  • Disentanglement score (R² of probe fit = how linearly separable each
                           factor is in the latent space)

Results saved to OUT_DIR/exp_seed{seed}/
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from config import OUT_DIR, SEQ_LENGTH, PROBE_FACTORS


MIDDLE_FRAME = SEQ_LENGTH // 2   # frame 7 of each 15-frame episode


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(encoder, imgs, sequences):
    """
    Returns
        frame_feats   (N, latent_dim)  z_mean of the middle frame per group
        episode_feats (N, latent_dim)  mean z_mean over all 15 frames per group
    """
    frame_feats, episode_feats = [], []
    for seq in sequences:
        clip    = imgs[seq]                                           # (15, 64, 64, 3)
        z_means = encoder.predict(clip, batch_size=64, verbose=0)[0] # (15, latent_dim)
        frame_feats.append(z_means[MIDDLE_FRAME])
        episode_feats.append(z_means.mean(axis=0))
    return np.stack(frame_feats), np.stack(episode_feats)


def extract_frame_level_features(encoder, imgs, sequences):
    """
    Extract z_mean for every individual frame alongside its orientation index.
    Used for the orientation probe only — orientation varies within each group
    so it needs a frame-level label, unlike the static factors (shape, colour
    etc.) which are constant across all 15 frames in a group.

    Returns
        all_z    (N × SEQ_LENGTH, latent_dim)
        all_ori  (N × SEQ_LENGTH,)  orientation indices 0..14
    """
    all_z, all_ori = [], []
    for seq in sequences:
        clip    = imgs[seq]
        z_means = encoder.predict(clip, batch_size=64, verbose=0)[0]  # (15, d)
        for frame_pos, z in enumerate(z_means):
            all_z.append(z)
            all_ori.append(frame_pos)   # orientation label = position in sequence
    return np.stack(all_z), np.array(all_ori, dtype=int)


# ── Probe ─────────────────────────────────────────────────────────────────────

def _probe(X_tr, y_tr, X_te, y_te):
    """Fit logistic regression probe; return (accuracy, r2)."""
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                           multi_class="auto", n_jobs=-1)
    )
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))

    # R² of a Ridge regression as a proxy for linear disentanglement
    reg = make_pipeline(StandardScaler(), Ridge())
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    r2 = float(r2_score(y_te, y_pred))

    return acc, r2


# ── Reconstruction MSE ────────────────────────────────────────────────────────

def reconstruction_mse(encoder, decoder, imgs, sequences, n=200):
    """Sample n test frames, encode and decode, return pixel MSE."""
    rng   = np.random.default_rng(0)
    idxs  = rng.choice(len(sequences), size=min(n, len(sequences)), replace=False)
    batch = np.concatenate([imgs[sequences[i]] for i in idxs])  # (n*15, 64, 64, 3)
    z_mean, _, _ = encoder.predict(batch, batch_size=128, verbose=0)
    recon        = decoder.predict(z_mean, batch_size=128, verbose=0)
    return float(np.mean((batch - recon) ** 2))


# ── Main evaluation ───────────────────────────────────────────────────────────

def run(models: dict,
        train_imgs, train_seqs, train_labels: dict,
        test_imgs,  test_seqs,  test_labels:  dict,
        seed: int):
    """
    models : dict returned by training.run_training_pipeline
    """
    out_dir = os.path.join(OUT_DIR, f"exp_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    students = {
        "teacher":     (models["teacher_enc"],  models["teacher_dec"]),
        "iid_matched": (models["iid_enc"],       models["iid_dec"]),
        "iid_mhn":     (models["iid_mhn_enc"],   models["iid_mhn_dec"]),
        "sequential":  (models["seq_enc"],       models["seq_dec"]),
    }

    results = {}

    for student_name, (enc, dec) in students.items():
        print(f"\n  Evaluating: {student_name}")

        # ── Reconstruction MSE ─────────────────────────────────────────────────
        recon_mse = reconstruction_mse(enc, dec, test_imgs, test_seqs)
        print(f"    Reconstruction MSE = {recon_mse:.5f}")

        # ── Extract features ───────────────────────────────────────────────────
        tr_frame, tr_ep = extract_features(enc, train_imgs, train_seqs)
        te_frame, te_ep = extract_features(enc, test_imgs,  test_seqs)

        # Frame-level features for the orientation probe (one entry per frame,
        # not per group — orientation varies within the group).
        tr_fl_z, tr_fl_ori = extract_frame_level_features(enc, train_imgs, train_seqs)
        te_fl_z, te_fl_ori = extract_frame_level_features(enc, test_imgs,  test_seqs)

        probe_results = {}
        for factor in PROBE_FACTORS:
            # ── orientation: frame-level probe ────────────────────────────────
            if factor == "orientation":
                acc_f, r2_f = _probe(tr_fl_z, tr_fl_ori, te_fl_z, te_fl_ori)
                # episode_acc for orientation isn't meaningful (all orientations
                # present in every episode), but we fill it to keep the schema
                # uniform; readers should focus on frame_acc here.
                probe_results[factor] = {
                    "frame_acc": acc_f, "frame_r2": r2_f,
                    "episode_acc": float("nan"), "episode_r2": float("nan"),
                }
                print(f"    {'orientation':<12}  frame acc={acc_f:.3f}  "
                      f"(frame-level probe; episode N/A)")
                continue

            # ── static factors: group-level probe ─────────────────────────────
            if factor not in train_labels:
                continue
            y_tr = train_labels[factor]
            y_te = test_labels[factor]

            acc_f, r2_f = _probe(tr_frame, y_tr, te_frame, y_te)
            acc_e, r2_e = _probe(tr_ep,    y_tr, te_ep,    y_te)
            probe_results[factor] = {
                "frame_acc":   acc_f, "frame_r2":   r2_f,
                "episode_acc": acc_e, "episode_r2": r2_e,
            }
            print(f"    {factor:<12}  frame acc={acc_f:.3f}  episode acc={acc_e:.3f}")

        results[student_name] = {
            "recon_mse": recon_mse,
            "probes":    probe_results,
        }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    _print_summary(results)
    _save_plots(results, out_dir)
    return results


def _print_summary(results):
    factors = list(PROBE_FACTORS.keys())
    print(f"\n{'─'*80}")
    print("PRIMARY METRIC: test-set probe accuracy (disjoint groups from training)")
    print(f"{'─'*80}")
    for rep in ("frame", "episode"):
        print(f"\n  [{rep} representation]")
        print(f"  {'Factor':<14}  {'Teacher':>9}  {'IID-match':>10}  {'IID-MHN':>9}  {'Seq':>7}  {'Winner':>8}")
        print("  " + "─" * 68)
        for f in factors:
            # orientation has no episode representation (all 15 present per group)
            if f == "orientation" and rep == "episode":
                continue
            def _g(name):
                return results[name]["probes"].get(f, {}).get(f"{rep}_acc", float("nan"))
            t, im, mhn, s = _g("teacher"), _g("iid_matched"), _g("iid_mhn"), _g("sequential")
            winner = "SEQ ✓" if s > im else "IID-match"
            note = " [frame-level]" if f == "orientation" else ""
            print(f"  {f:<14}  {t:9.3f}  {im:10.3f}  {mhn:9.3f}  {s:7.3f}  {winner:>8}{note}")
    print(f"\n  Reconstruction MSE (test images):")
    for name in ("teacher", "iid_matched", "iid_mhn", "sequential"):
        print(f"    {name:<14}: {results[name]['recon_mse']:.5f}")


_CONDITIONS = [
    ("teacher",     "seagreen",  "Teacher (upper bound)"),
    ("iid_matched", "steelblue", "IID-matched (primary)"),
    ("iid_mhn",     "mediumpurple", "IID-MHN (Spens & Burgess)"),
    ("sequential",  "coral",     "Sequential (ours)"),
]


def _save_plots(results, out_dir):
    factors = [f for f in PROBE_FACTORS if f in results["iid_matched"]["probes"]]
    x       = np.arange(len(factors))
    n_cond  = len(_CONDITIONS)
    width   = 0.8 / n_cond
    offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * width

    for rep in ("frame", "episode"):
        # For episode rep, orientation probe is not meaningful — exclude it
        rep_factors = [f for f in factors if not (f == "orientation" and rep == "episode")]
        x_rep = np.arange(len(rep_factors))

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        for ax_idx, metric_key, ylabel in [
            (0, f"{rep}_acc", "Probe accuracy (test set)"),
            (1, f"{rep}_r2",  "R² (Ridge, test set)"),
        ]:
            ax = axes[ax_idx]
            for (name, color, label), offset in zip(_CONDITIONS, offsets):
                vals = [results[name]["probes"].get(f, {}).get(metric_key, 0)
                        for f in rep_factors]
                ax.bar(x_rep + offset, vals, width, label=label, color=color, alpha=0.85)

            ax.set_xticks(x_rep)
            xlabels = [f + " *" if f == "orientation" else f for f in rep_factors]
            ax.set_xticklabels(xlabels, rotation=20, ha="right")
            ax.set_ylabel(ylabel); ax.legend(fontsize=7)
            if metric_key.endswith("_acc"):
                ax.set_ylim(0, 1.05)
                for f in rep_factors:
                    ax.axhline(1.0 / PROBE_FACTORS[f], ls=":", c="gray", lw=0.7, alpha=0.4)
            else:
                ax.axhline(0, ls="--", c="gray", lw=0.8)

        ori_note = "  (* orientation = frame-level probe)" if rep == "frame" else ""
        plt.suptitle(
            f"Representation quality: {rep} encoding  "
            f"(primary comparison: IID-matched vs Sequential)\n"
            f"Probed on held-out test groups — disjoint from all student training data{ori_note}",
            fontsize=9,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"probes_{rep}.png"), dpi=150)
        plt.close()

    # Summary: mean accuracy across static factors only (exclude orientation)
    static_factors = [f for f in factors if f != "orientation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_i, rep in enumerate(("frame", "episode")):
        ax   = axes[ax_i]
        names  = [c[0] for c in _CONDITIONS]
        colors = [c[1] for c in _CONDITIONS]
        labels = [c[2] for c in _CONDITIONS]
        mean_accs = [
            np.mean([results[n]["probes"].get(f, {}).get(f"{rep}_acc", 0)
                     for f in static_factors])
            for n in names
        ]
        bars = ax.bar(range(len(names)), mean_accs, color=colors, alpha=0.85)
        for bar, v in zip(bars, mean_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean accuracy (static factors only)")
        ax.set_title(f"{rep} representation")
        ax.axhline(0.25, ls="--", c="gray", lw=0.8, label="chance (shape)")

    plt.suptitle(
        "Does sequential replay produce better representations of static semantics?\n"
        "(orientation excluded — it is what sequences vary along)",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_static_factors.png"), dpi=150)
    plt.close()
