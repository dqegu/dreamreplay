"""
experiment.py
─────────────────────────────────────────────────────────────────────────────
Single experiment: partial-cue episode recall.

Claim
─────
Sequential replay produces better memory consolidation than itemwise
replay, as measured by the ability to reconstruct a complete episode
from a partial cue.

Design
──────
Three students, identical sequence VAE architecture and training objective.
Only their replay corpus differs:

  sequential  — episodes replayed in temporal order (K/V heteroassociative)
  shuffled    — same episodes, frames randomly permuted within each episode
  iid         — sequences of SEQ_LENGTH independently MHN-retrieved frames
                (no temporal relationship between frames)

All three students are trained on the same number of replay sequences
for the same number of epochs.  The only variable is the structure of
the replay data they receive.

Evaluation
──────────
Each held-out test episode is corrupted by masking frames gap_start–gap_end.
Each student encodes the corrupted episode through its full pipeline
(teacher encoder → seq encoder → seq decoder → teacher decoder) and
produces a full reconstruction.  Quality is measured on the masked frames
only, in both pixel space and teacher latent space.

Teacher latent MSE is the primary metric because it is objective-neutral:
it measures how close the reconstruction is to the ground truth in a shared
representation space, independent of each student's decoder sharpness.

Pixel MSE is reported as a secondary metric.

Analogous to Spens & Burgess Fig. 1d (reconstruction from partial input).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUT_DIR, SEQ_LENGTH, LATENT_DIM


# ═══════════════════════════════════════════════════════════════════════════
# Core helper
# ═══════════════════════════════════════════════════════════════════════════

def _reconstruct_episode(seq_enc, seq_dec, teacher_enc, teacher_dec,
                          frames, mask_indices):
    """
    Reconstruct a full episode through one student's pipeline.

    1. Zero out the masked frames (partial cue).
    2. Encode each frame through the frozen teacher encoder → frame latents.
    3. Pass frame latent sequence through the student seq encoder → z_seq.
    4. Decode z_seq through the student seq decoder → predicted teacher latents.
    5. Decode each predicted latent through frozen teacher decoder → frames.

    Returns: recon_frames (SEQ_LENGTH, 64, 64, 3)
    """
    inp = frames.copy()
    inp[mask_indices] = 0.0

    f_lat     = teacher_enc.predict(inp, batch_size=SEQ_LENGTH, verbose=0)[0]
    f_lat_exp = f_lat[np.newaxis]
    z_mean, _, _ = seq_enc.predict(f_lat_exp, verbose=0)
    pred_latents  = seq_dec.predict(z_mean, verbose=0)[0]
    recon = teacher_dec.predict(
        pred_latents, batch_size=SEQ_LENGTH, verbose=0)
    return recon


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def exp_partial_cue_recall(models,
                            test_imgs, test_seqs,
                            gap_start=5, gap_end=10,
                            n_seqs=200, rng_seed=0):
    """
    Partial-cue episode recall.  Analogous to Spens & Burgess Fig. 1d.

    For each held-out test episode:
      - Mask frames gap_start..gap_end-1 (set to zero)
      - Each student reconstructs the full episode from the partial cue
      - Evaluate on masked frames only

    Primary metric:  teacher latent MSE (objective-neutral)
    Secondary metric: pixel MSE
    """
    rng      = np.random.default_rng(rng_seed)
    idxs     = rng.choice(len(test_seqs),
                           size=min(n_seqs, len(test_seqs)), replace=False)
    mask_idx = list(range(gap_start, gap_end))
    gap_len  = gap_end - gap_start

    teacher_enc = models["teacher_enc"]
    teacher_dec = models["teacher_dec"]

    students = {
        "sequential": (models["seq_enc"],  models["seq_dec"]),
        "shuffled":   (models["shuf_enc"], models["shuf_dec"]),
        "iid":        (models["iid_enc"],  models["iid_dec"]),
    }

    per_seq = {name: {"pixel": [], "latent": []} for name in students}

    for i in idxs:
        frames     = test_imgs[test_seqs[i]]
        gap_frames = frames[gap_start:gap_end]
        gt_z = teacher_enc.predict(
            gap_frames, batch_size=gap_len, verbose=0)[0]

        for name, (enc, dec) in students.items():
            recon     = _reconstruct_episode(
                enc, dec, teacher_enc, teacher_dec, frames, mask_idx)
            recon_gap = recon[gap_start:gap_end]

            per_seq[name]["pixel"].append(
                float(np.mean((gap_frames - recon_gap) ** 2)))

            recon_z = teacher_enc.predict(
                recon_gap, batch_size=gap_len, verbose=0)[0]
            per_seq[name]["latent"].append(
                float(np.mean((gt_z - recon_z) ** 2)))

    out = {}
    for name in students:
        p = np.array(per_seq[name]["pixel"])
        l = np.array(per_seq[name]["latent"])
        out[name] = {
            "pixel_mse":      float(np.mean(p)),
            "pixel_mse_std":  float(np.std(p)),
            "latent_mse":     float(np.mean(l)),
            "latent_mse_std": float(np.std(l)),
        }

    seq_l  = np.array(per_seq["sequential"]["latent"])
    shuf_l = np.array(per_seq["shuffled"]["latent"])
    iid_l  = np.array(per_seq["iid"]["latent"])

    out["seq_wins_vs_shuffled"] = int(np.sum(seq_l < shuf_l))
    out["seq_wins_vs_iid"]      = int(np.sum(seq_l < iid_l))
    out["shuf_wins_vs_iid"]     = int(np.sum(shuf_l < iid_l))
    out["n_seqs"]               = len(idxs)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Run and save
# ═══════════════════════════════════════════════════════════════════════════

def run(models,
        train_imgs, train_seqs, train_labels,
        test_imgs,  test_seqs,  test_labels,
        seed: int):

    out_dir = os.path.join(OUT_DIR, f"exp_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n── Partial-cue episode recall ──")
    results = exp_partial_cue_recall(models, test_imgs, test_seqs)

    print(f"\n  {'Condition':<14}  {'Pixel MSE':>14}  {'Latent MSE':>14}")
    print("  " + "─" * 48)
    for name in ("sequential", "shuffled", "iid"):
        r = results[name]
        print(f"  {name:<14}  "
              f"{r['pixel_mse']:>8.5f}±{r['pixel_mse_std']:.5f}  "
              f"{r['latent_mse']:>8.5f}±{r['latent_mse_std']:.5f}")

    n = results["n_seqs"]
    print(f"\n  Win rates on latent MSE (lower = better consolidation):")
    print(f"    Seq vs Shuffled: {results['seq_wins_vs_shuffled']}/{n}")
    print(f"    Seq vs IID:      {results['seq_wins_vs_iid']}/{n}")
    print(f"    Shuf vs IID:     {results['shuf_wins_vs_iid']}/{n}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    _save_plot(results, out_dir)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════

def _save_plot(results, out_dir):
    conditions = ["sequential", "shuffled", "iid"]
    labels     = ["Sequential\n(ordered replay)",
                  "Shuffled\n(random order)",
                  "IID\n(independent frames)"]
    colors     = ["coral", "steelblue", "mediumpurple"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax_idx, metric, ylabel, note in [
        (0, "latent_mse", "Teacher latent MSE (lower = better)",
         "Primary — objective-neutral shared space"),
        (1, "pixel_mse",  "Pixel MSE (lower = better)",
         "Secondary metric"),
    ]:
        ax   = axes[ax_idx]
        vals = [results[n][metric]           for n in conditions]
        errs = [results[n][f"{metric}_std"]  for n in conditions]
        x    = np.arange(len(conditions))
        bars = ax.bar(x, vals, yerr=errs, color=colors,
                       alpha=0.85, capsize=6, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(vals) * 0.02,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Partial-cue recall — gap frames 5–9\n{note}")
        ax.set_ylim(0, max(vals) * 1.35)

    plt.suptitle(
        "Effect of replay structure on memory consolidation\n"
        "All three students: identical sequence VAE — only replay differs",
        fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "partial_cue_recall.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved to {path}")