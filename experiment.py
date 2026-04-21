"""
experiment.py
─────────────────────────────────────────────────────────────────────────────
Four experiments translating the main consolidation signatures from
Spens & Burgess (2024) into the sequential domain.

Exp 1 — Partial-cue recall
  Mask frames 5–9 of a held-out sequence. Each student reconstructs the
  full sequence from the corrupted input. Measure reconstruction quality
  on the masked frames in both pixel and teacher-latent space.
  Analogous to Spens & Burgess Fig. 1d.

Exp 2 — Semanticisation over consolidation
  Probe the sequence latent z_seq for episode-level semantic variables
  (shape, object hue, trajectory direction) at multiple training
  checkpoints (epochs 5, 10, 20, 30). Tests whether semantic structure
  becomes increasingly decodable as replay consolidation progresses.
  Analogous to Spens & Burgess Fig. 3a.

Exp 3 — Imagination and relational inference
  Interpolate between pairs of episode latents and decode intermediate
  sequences. Measure temporal smoothness and semantic consistency of
  generated trajectories. Tests whether the latent space supports
  structured generalisation beyond seen episodes.
  Analogous to Spens & Burgess Fig. 3b–d.

Exp 4 — Schema distortion
  Feed each student a full atypical sequence (locally scrambled frame
  order). Reconstruct through the student's own sequence decoder.
  Recover the orientation trajectory of the reconstruction via
  teacher nearest-neighbour lookup. Measure MAD from the canonical
  monotonic sweep. True schema distortion because the student's own
  generative process must produce the regularisation.
  Analogous to Spens & Burgess Fig. 4.

All experiments compare:
  Primary:   sequential seq-VAE  vs  shuffled seq-VAE
             (same architecture, same pixels — isolates temporal order)
  Secondary: sequential seq-VAE  vs  IID frame-VAE
             (compares to original Spens & Burgess baseline)
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
from tensorflow import keras

from config import (
    OUT_DIR, SEQ_LENGTH, LATENT_DIM, SEQ_LATENT_DIM,
    PROBE_FACTORS, CHECKPOINT_EPOCHS, student_paths,
)


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _encode_sequences_to_z(seq_enc, teacher_enc,
                             imgs, seqs, batch=64):
    """
    Encode each episode (15 frames) to a single z_seq vector.

    Pipeline:
      frames (15, 64,64,3)
        → frozen teacher encoder → frame latents (15, LATENT_DIM)
        → seq encoder            → z_seq (SEQ_LATENT_DIM,)

    Returns z_means: (N, SEQ_LATENT_DIM)
    """
    z_list = []
    for seq in seqs:
        frames    = imgs[seq]                               # (15, 64,64,3)
        f_lat     = teacher_enc.predict(
            frames, batch_size=SEQ_LENGTH, verbose=0)[0]   # (15, LATENT_DIM)
        f_lat_exp = f_lat[np.newaxis]                      # (1, 15, LATENT_DIM)
        z_mean, _, _ = seq_enc.predict(f_lat_exp, verbose=0)
        z_list.append(z_mean[0])
    return np.stack(z_list)                                # (N, SEQ_LATENT_DIM)


def _reconstruct_sequence(seq_enc, seq_dec, teacher_enc, teacher_dec,
                            frames, mask_indices=None):
    """
    Reconstruct a full sequence through the student's own pipeline.

    If mask_indices is given, those frame positions are zeroed before
    encoding (partial-cue experiment).

    Returns recon_frames: (SEQ_LENGTH, 64, 64, 3)
    """
    inp = frames.copy()
    if mask_indices is not None:
        inp[mask_indices] = 0.0

    # Encode frames through frozen teacher
    f_lat = teacher_enc.predict(
        inp, batch_size=SEQ_LENGTH, verbose=0)[0]           # (15, LATENT_Dim)
    f_lat_exp = f_lat[np.newaxis]                           # (1, 15, LATENT_DIM)

    # Encode to z_seq
    z_mean, _, _ = seq_enc.predict(f_lat_exp, verbose=0)   # (1, SEQ_LATENT_DIM)

    # Decode to predicted teacher latents
    pred_latents = seq_dec.predict(z_mean, verbose=0)[0]   # (15, LATENT_DIM)

    # Decode each predicted latent to a frame
    recon = teacher_dec.predict(
        pred_latents, batch_size=SEQ_LENGTH, verbose=0)    # (15, 64, 64, 3)
    return recon


def _orientation_from_frames(recon_frames, teacher_enc, canonical_latents):
    """
    Recover the orientation index of each reconstructed frame via
    nearest-neighbour lookup in the teacher's latent space.

    canonical_latents: (SEQ_LENGTH, LATENT_DIM) teacher latents for the
                       15 canonical orientations of this group.
    Returns: int array of shape (SEQ_LENGTH,) with orientation indices 0–14.
    """
    z = teacher_enc.predict(
        recon_frames, batch_size=SEQ_LENGTH, verbose=0)[0]  # (15, LATENT_DIM)
    traj = []
    for zi in z:
        dists = np.linalg.norm(canonical_latents - zi, axis=1)
        traj.append(int(np.argmin(dists)))
    return np.array(traj, dtype=int)


def _mad_from_canonical(trajectory):
    """Mean absolute deviation of an orientation trajectory from [0,1,...,14]."""
    return float(np.mean(np.abs(np.array(trajectory) -
                                 np.arange(SEQ_LENGTH))))


def _probe(X_tr, y_tr, X_te, y_te):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    reg = make_pipeline(StandardScaler(), Ridge())
    reg.fit(X_tr, y_tr)
    r2  = float(r2_score(y_te, reg.predict(X_te)))
    return acc, r2


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1 — Partial-cue recall
# ═══════════════════════════════════════════════════════════════════════════

def exp1_partial_cue_recall(models,
                              test_imgs, test_seqs,
                              gap_start=5, gap_end=10,
                              n_seqs=200, rng_seed=0):
    """
    Partial-cue recall.  Analogous to Spens & Burgess Fig. 1d.

    Each 15-frame test sequence is presented with frames gap_start–gap_end-1
    masked (set to zero).  Each student encodes the corrupted sequence through
    its full pipeline (frame embedding → seq encoder → seq decoder → teacher
    decoder) and produces a full sequence reconstruction.

    We evaluate quality only on the MASKED frames, using two metrics:
      Pixel MSE     — direct pixel comparison (shared scale for all students)
      Latent MSE    — MSE in teacher latent space (removes decoder-sharpness
                      differences between students)

    Conditions compared:
      sequential  — seq-VAE trained on ordered replay
      shuffled    — seq-VAE trained on shuffled replay (same arch, same pixels)
      iid         — original Spens & Burgess frame-level VAE (secondary)
    """
    rng         = np.random.default_rng(rng_seed)
    idxs        = rng.choice(len(test_seqs),
                              size=min(n_seqs, len(test_seqs)),
                              replace=False)
    mask_idx    = list(range(gap_start, gap_end))
    gap_len     = gap_end - gap_start
    teacher_enc = models["teacher_enc"]
    teacher_dec = models["teacher_dec"]

    students = {
        "sequential": (models["seq_enc"],  models["seq_dec"]),
        "shuffled":   (models["shuf_enc"], models["shuf_dec"]),
    }

    results = {name: {"pixel_errors": [], "latent_errors": []}
               for name in students}

    for i in idxs:
        frames     = test_imgs[test_seqs[i]]             # (15, 64,64,3)
        gap_frames = frames[gap_start:gap_end]            # ground truth gap

        # Teacher latents for ground-truth gap (shared reference)
        gt_z = teacher_enc.predict(
            gap_frames, batch_size=gap_len, verbose=0)[0]  # (gap_len, LATENT_DIM)

        for name, (enc, dec) in students.items():
            recon = _reconstruct_sequence(
                enc, dec, teacher_enc, teacher_dec,
                frames, mask_indices=mask_idx)              # (15, 64,64,3)
            recon_gap = recon[gap_start:gap_end]

            # Pixel MSE on gap frames
            pixel_err = float(np.mean((gap_frames - recon_gap) ** 2))
            results[name]["pixel_errors"].append(pixel_err)

            # Latent MSE: encode recon gap through teacher
            recon_z = teacher_enc.predict(
                recon_gap, batch_size=gap_len, verbose=0)[0]
            lat_err = float(np.mean((gt_z - recon_z) ** 2))
            results[name]["latent_errors"].append(lat_err)

    out = {}
    for name in students:
        pe = np.array(results[name]["pixel_errors"])
        le = np.array(results[name]["latent_errors"])
        out[name] = {
            "pixel_mse":     float(np.mean(pe)),
            "pixel_mse_std": float(np.std(pe)),
            "latent_mse":    float(np.mean(le)),
            "latent_mse_std":float(np.std(le)),
        }

    # Win counts
    seq_p = np.array(results["sequential"]["pixel_errors"])
    shuf_p = np.array(results["shuffled"]["pixel_errors"])
    seq_l = np.array(results["sequential"]["latent_errors"])
    shuf_l = np.array(results["shuffled"]["latent_errors"])
    out["seq_wins_pixel"]  = int(np.sum(seq_p < shuf_p))
    out["seq_wins_latent"] = int(np.sum(seq_l < shuf_l))
    out["n_seqs"]          = len(idxs)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2 — Semanticisation over consolidation
# ═══════════════════════════════════════════════════════════════════════════

def exp2_semanticisation(models, seed,
                          train_imgs, train_seqs, train_labels,
                          test_imgs,  test_seqs,  test_labels,
                          n_train=800, n_test=200):
    """
    Semanticisation.  Analogous to Spens & Burgess Fig. 3a.

    Probes whether the sequence latent z_seq encodes episode-level semantic
    structure, and whether this improves across replay epochs (consolidation).

    For each checkpoint epoch and for the final model:
      1. Encode all training/test sequences to z_seq vectors.
      2. Fit logistic regression probes for each semantic variable
         (shape, object_hue, floor_hue, wall_hue, scale) on training z_seq.
      3. Evaluate on test z_seq.

    Conditions:
      sequential  — seq-VAE checkpoints and final model
      shuffled    — shuffled seq-VAE checkpoints and final model

    The expected finding (if sequential replay consolidates semantic
    structure): probe accuracy increases across epochs for sequential
    and does so faster or further than shuffled.
    """
    paths     = student_paths(seed)
    ckpt_dir  = paths["seq_ckpt_dir"]
    shuf_dir  = ckpt_dir.replace("seq_vae", "shuf_vae")
    teacher_enc = models["teacher_enc"]

    # Probing factors (episode-level — one label per episode)
    probe_factors = [f for f in ("shape", "object_hue", "floor_hue",
                                  "wall_hue", "scale")
                     if f in train_labels]

    def _probe_at(seq_enc, seq_dec, label):
        z_tr = _encode_sequences_to_z(
            seq_enc, teacher_enc, train_imgs, train_seqs[:n_train])
        z_te = _encode_sequences_to_z(
            seq_enc, teacher_enc, test_imgs,  test_seqs[:n_test])

        factor_results = {}
        for factor in probe_factors:
            y_tr = train_labels[factor][:n_train]
            y_te = test_labels[factor][:n_test]
            acc, r2 = _probe(z_tr, y_tr, z_te, y_te)
            factor_results[factor] = {"acc": acc, "r2": r2}
            print(f"      [{label}] {factor}: acc={acc:.3f}  r2={r2:.3f}")
        return factor_results

    results = {"sequential": {}, "shuffled": {}}

    # Probe at each checkpoint epoch
    for epoch in CHECKPOINT_EPOCHS:
        print(f"  Epoch {epoch}:")
        for cond, ckpt_d in (("sequential", ckpt_dir),
                              ("shuffled",   shuf_dir)):
            enc_path = os.path.join(ckpt_d, f"enc_epoch{epoch}.keras")
            dec_path = os.path.join(ckpt_d, f"dec_epoch{epoch}.keras")
            if not os.path.exists(enc_path):
                print(f"    [{cond}] checkpoint epoch {epoch} not found — skipping.")
                continue
            enc = keras.models.load_model(enc_path, compile=False)
            dec = keras.models.load_model(dec_path, compile=False)
            label = f"{cond} epoch {epoch}"
            results[cond][f"epoch_{epoch}"] = _probe_at(enc, dec, label)

    # Probe final models
    print(f"  Final model:")
    for cond, (enc, dec) in (("sequential", (models["seq_enc"], models["seq_dec"])),
                               ("shuffled",   (models["shuf_enc"], models["shuf_dec"]))):
        results[cond]["final"] = _probe_at(enc, dec, f"{cond} final")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3 — Imagination and relational inference
# ═══════════════════════════════════════════════════════════════════════════

def exp3_imagination(models,
                      test_imgs, test_seqs,
                      n_pairs=100, n_interp_steps=5,
                      rng_seed=2):
    """
    Imagination via latent interpolation.
    Analogous to Spens & Burgess Fig. 3b–d.

    For random pairs of held-out episodes:
      1. Encode each episode to z_seq_A and z_seq_B.
      2. Linearly interpolate: z_alpha = (1-alpha)*z_A + alpha*z_B
         for alpha in {0, 0.25, 0.5, 0.75, 1.0}.
      3. Decode each z_alpha to a full sequence of frames.
      4. Recover the orientation trajectory of each decoded sequence.
      5. Measure:
           temporal_smoothness: how monotonically ordered is the trajectory?
           semantic_consistency: is each decoded frame closest to a valid
                                 orientation (i.e. does the decoded sequence
                                 lie on the data manifold)?

    A well-structured latent space (consistent with Spens & Burgess's
    imagination argument) should produce smooth, valid intermediate sequences.

    Conditions: sequential and shuffled seq-VAE.
    """
    rng         = np.random.default_rng(rng_seed)
    n_seqs      = len(test_seqs)
    teacher_enc = models["teacher_enc"]
    teacher_dec = models["teacher_dec"]
    alphas      = np.linspace(0, 1, n_interp_steps)

    students = {
        "sequential": (models["seq_enc"], models["seq_dec"]),
        "shuffled":   (models["shuf_enc"], models["shuf_dec"]),
    }

    results = {}

    for name, (seq_enc, seq_dec) in students.items():
        smoothness_scores = []
        validity_scores   = []

        # Sample random pairs
        pair_idxs = [(rng.integers(n_seqs), rng.integers(n_seqs))
                     for _ in range(n_pairs)]

        for ia, ib in pair_idxs:
            frames_a = test_imgs[test_seqs[ia]]
            frames_b = test_imgs[test_seqs[ib]]

            # Encode both episodes
            fa_lat = teacher_enc.predict(
                frames_a, batch_size=SEQ_LENGTH, verbose=0)[0][np.newaxis]
            fb_lat = teacher_enc.predict(
                frames_b, batch_size=SEQ_LENGTH, verbose=0)[0][np.newaxis]

            z_a = seq_enc.predict(fa_lat, verbose=0)[0][0]  # (SEQ_LATENT_DIM,)
            z_b = seq_enc.predict(fb_lat, verbose=0)[0][0]

            # Canonical teacher latents for episode A (for orientation lookup)
            canonical_z = teacher_enc.predict(
                frames_a, batch_size=SEQ_LENGTH, verbose=0)[0]

            ep_smooth, ep_valid = [], []

            for alpha in alphas:
                z_interp = ((1 - alpha) * z_a +
                             alpha      * z_b)[np.newaxis]   # (1, SEQ_LATENT_DIM)

                # Decode
                pred_latents = seq_dec.predict(
                    z_interp, verbose=0)[0]                  # (15, LATENT_DIM)
                recon = teacher_dec.predict(
                    pred_latents, batch_size=SEQ_LENGTH, verbose=0)

                # Recover orientation trajectory
                traj = _orientation_from_frames(
                    recon, teacher_enc, canonical_z)

                # Temporal smoothness: fraction of steps where orientation
                # changes by at most 2 (tolerant measure of local order)
                diffs = np.abs(np.diff(traj))
                smooth = float(np.mean(diffs <= 2))
                ep_smooth.append(smooth)

                # Validity: all recovered orientations are in [0, 14]
                valid = float(np.all((traj >= 0) & (traj < SEQ_LENGTH)))
                ep_valid.append(valid)

            smoothness_scores.append(np.mean(ep_smooth))
            validity_scores.append(np.mean(ep_valid))

        results[name] = {
            "smoothness_mean": float(np.mean(smoothness_scores)),
            "smoothness_std":  float(np.std(smoothness_scores)),
            "validity_mean":   float(np.mean(validity_scores)),
            "validity_std":    float(np.std(validity_scores)),
        }
        print(f"    [{name}] smoothness={results[name]['smoothness_mean']:.3f}  "
              f"validity={results[name]['validity_mean']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 4 — Schema distortion
# ═══════════════════════════════════════════════════════════════════════════

def _make_atypical_sequence(frames, rng, n_perturb=3):
    """
    Create an atypical full sequence by swapping n_perturb pairs of adjacent
    frames anywhere in the sequence, breaking the smooth monotonic sweep.

    Returns:
      perturbed_frames: (SEQ_LENGTH, 64, 64, 3)
      ori_indices:      (SEQ_LENGTH,) int — the new orientation order
    """
    perturbed   = frames.copy()
    ori_indices = list(range(SEQ_LENGTH))
    positions   = rng.choice(np.arange(SEQ_LENGTH - 1),
                              size=min(n_perturb, SEQ_LENGTH - 1),
                              replace=False)
    for pos in positions:
        perturbed[[pos, pos + 1]] = perturbed[[pos + 1, pos]]
        ori_indices[pos], ori_indices[pos + 1] = (ori_indices[pos + 1],
                                                   ori_indices[pos])
    return perturbed, np.array(ori_indices)


def exp4_schema_distortion(models,
                            test_imgs, test_seqs,
                            n_seqs=200, n_perturb=3,
                            rng_seed=1):
    """
    Schema distortion.  Analogous to Spens & Burgess Fig. 4.

    For each held-out canonical sequence:
      1. Create an atypical version by swapping n_perturb adjacent frame pairs.
      2. Feed the FULL atypical sequence through each student's pipeline
         (teacher encoder → seq encoder → seq decoder → teacher decoder).
      3. Recover the orientation trajectory of the reconstructed sequence
         via nearest-neighbour lookup in the teacher latent space.
      4. Measure MAD from the canonical sweep [0, 1, ..., 14].

    Key quantities:
      canonical_mad   = 0.0  (reference)
      atypical_mad    = MAD of the perturbed input (how distorted was the input)
      recon_mad       = MAD of the student's reconstruction

    Schema distortion is present when recon_mad < atypical_mad:
      the student's reconstruction has been pulled toward the canonical schema.

    This is a TRUE schema distortion test because:
      - the student sees the full atypical sequence
      - the student's OWN generative process (seq decoder) produces the output
      - any schema pull must come from the learned latent space, not an
        external decoder

    Conditions: sequential vs shuffled (primary), sequential vs IID (secondary).
    """
    rng         = np.random.default_rng(rng_seed)
    idxs        = rng.choice(len(test_seqs),
                              size=min(n_seqs, len(test_seqs)),
                              replace=False)
    teacher_enc = models["teacher_enc"]
    teacher_dec = models["teacher_dec"]

    students = {
        "sequential": (models["seq_enc"], models["seq_dec"]),
        "shuffled":   (models["shuf_enc"], models["shuf_dec"]),
    }

    atyp_mads   = []
    recon_mads  = {name: [] for name in students}

    for i in idxs:
        true_frames = test_imgs[test_seqs[i]]

        # Teacher latents for canonical frames (orientation lookup reference)
        canonical_z = teacher_enc.predict(
            true_frames, batch_size=SEQ_LENGTH, verbose=0)[0]

        # Create atypical sequence
        atyp_frames, atyp_ori = _make_atypical_sequence(
            true_frames, rng, n_perturb)
        atyp_mads.append(_mad_from_canonical(atyp_ori))

        for name, (seq_enc, seq_dec) in students.items():
            # Reconstruct through student's full pipeline
            recon = _reconstruct_sequence(
                seq_enc, seq_dec, teacher_enc, teacher_dec, atyp_frames)

            # Recover orientation trajectory
            traj = _orientation_from_frames(recon, teacher_enc, canonical_z)
            recon_mads[name].append(_mad_from_canonical(traj))

    atyp_arr = np.array(atyp_mads)
    out = {
        "canonical_mad":    0.0,
        "atypical_mad":     float(np.mean(atyp_arr)),
        "atypical_mad_std": float(np.std(atyp_arr)),
    }

    for name in students:
        r_arr = np.array(recon_mads[name])
        out[name] = {
            "recon_mad":     float(np.mean(r_arr)),
            "recon_mad_std": float(np.std(r_arr)),
            "schema_pull":   float(np.mean(r_arr) - np.mean(atyp_arr)),
            "wins_vs_atyp":  int(np.sum(r_arr < atyp_arr)),
        }

    seq_r  = np.array(recon_mads["sequential"])
    shuf_r = np.array(recon_mads["shuffled"])
    out["seq_vs_shuffled"] = float(np.mean(seq_r) - np.mean(shuf_r))
    out["seq_wins_vs_shuf"] = int(np.sum(seq_r < shuf_r))
    out["n_seqs"]           = len(idxs)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main run function
# ═══════════════════════════════════════════════════════════════════════════

def run(models,
        train_imgs, train_seqs, train_labels,
        test_imgs,  test_seqs,  test_labels,
        seed: int):

    out_dir = os.path.join(OUT_DIR, f"exp_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    # ── Exp 1 ─────────────────────────────────────────────────────────────
    print("\n── Exp 1: Partial-cue recall ──")
    exp1 = exp1_partial_cue_recall(models, test_imgs, test_seqs)
    all_results["exp1"] = exp1
    for name in ("sequential", "shuffled"):
        print(f"  [{name}]  pixel MSE={exp1[name]['pixel_mse']:.5f}  "
              f"latent MSE={exp1[name]['latent_mse']:.5f}")
    print(f"  Seq wins (pixel):  {exp1['seq_wins_pixel']}/{exp1['n_seqs']}")
    print(f"  Seq wins (latent): {exp1['seq_wins_latent']}/{exp1['n_seqs']}")

    # ── Exp 2 ─────────────────────────────────────────────────────────────
    print("\n── Exp 2: Semanticisation over consolidation ──")
    exp2 = exp2_semanticisation(
        models, seed,
        train_imgs, train_seqs, train_labels,
        test_imgs,  test_seqs,  test_labels)
    all_results["exp2"] = exp2

    # ── Exp 3 ─────────────────────────────────────────────────────────────
    print("\n── Exp 3: Imagination via latent interpolation ──")
    exp3 = exp3_imagination(models, test_imgs, test_seqs)
    all_results["exp3"] = exp3
    for name in ("sequential", "shuffled"):
        print(f"  [{name}]  smoothness={exp3[name]['smoothness_mean']:.3f}  "
              f"validity={exp3[name]['validity_mean']:.3f}")

    # ── Exp 4 ─────────────────────────────────────────────────────────────
    print("\n── Exp 4: Schema distortion ──")
    exp4 = exp4_schema_distortion(models, test_imgs, test_seqs)
    all_results["exp4"] = exp4
    print(f"  Canonical MAD:     {exp4['canonical_mad']:.3f}")
    print(f"  Atypical MAD:      {exp4['atypical_mad']:.3f}")
    for name in ("sequential", "shuffled"):
        print(f"  [{name}]  recon MAD={exp4[name]['recon_mad']:.3f}  "
              f"schema pull={exp4[name]['schema_pull']:+.3f}  "
              f"wins vs atypical={exp4[name]['wins_vs_atyp']}/{exp4['n_seqs']}")
    print(f"  Seq vs shuffled: {exp4['seq_vs_shuffled']:+.3f}  "
          f"seq wins {exp4['seq_wins_vs_shuf']}/{exp4['n_seqs']}")

    # ── Save & plot ────────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    _save_plots(exp1, exp2, exp3, exp4, out_dir)
    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

def _save_plots(exp1, exp2, exp3, exp4, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: Partial-cue recall ───────────────────────────────────────
    ax = axes[0, 0]
    names  = ["sequential", "shuffled"]
    colors = ["coral", "steelblue"]
    x = np.arange(2)
    w = 0.35
    p_vals = [exp1[n]["pixel_mse"]  for n in names]
    l_vals = [exp1[n]["latent_mse"] for n in names]
    p_errs = [exp1[n]["pixel_mse_std"]  for n in names]
    l_errs = [exp1[n]["latent_mse_std"] for n in names]
    ax.bar(x - w/2, p_vals, w, yerr=p_errs, label="Pixel MSE",
           color=colors, alpha=0.85, capsize=4)
    ax.bar(x + w/2, l_vals, w, yerr=l_errs, label="Latent MSE",
           color=colors, alpha=0.5, hatch="//", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Sequential", "Shuffled"])
    ax.set_ylabel("MSE on masked frames (lower = better)")
    ax.set_title("Exp 1: Partial-cue recall\n(frames 5–9 masked)")
    ax.legend(fontsize=8)

    # ── Panel 2: Semanticisation over epochs ──────────────────────────────
    ax = axes[0, 1]
    epochs = CHECKPOINT_EPOCHS + ["final"]
    x_epochs = list(range(len(epochs)))
    for name, color in (("sequential", "coral"), ("shuffled", "steelblue")):
        mean_accs = []
        for ep_key in [f"epoch_{e}" for e in CHECKPOINT_EPOCHS] + ["final"]:
            if ep_key in exp2[name]:
                accs = [v["acc"] for v in exp2[name][ep_key].values()]
                mean_accs.append(np.mean(accs))
            else:
                mean_accs.append(float("nan"))
        ax.plot(x_epochs, mean_accs, marker="o", label=name, color=color)
    ax.set_xticks(x_epochs)
    ax.set_xticklabels([str(e) for e in CHECKPOINT_EPOCHS] + ["final"],
                        rotation=15)
    ax.axhline(0.25, ls="--", c="gray", lw=0.8, label="Chance (~0.25)")
    ax.set_ylabel("Mean probe accuracy (episode semantics)")
    ax.set_title("Exp 2: Semanticisation\n(episode-level probe accuracy vs epoch)")
    ax.legend(fontsize=8)

    # ── Panel 3: Imagination smoothness ──────────────────────────────────
    ax = axes[1, 0]
    names  = ["sequential", "shuffled"]
    smooth = [exp3[n]["smoothness_mean"] for n in names]
    s_err  = [exp3[n]["smoothness_std"]  for n in names]
    bars = ax.bar(["Sequential", "Shuffled"], smooth,
                   yerr=s_err, color=["coral", "steelblue"],
                   alpha=0.85, capsize=5, width=0.5)
    for bar, v in zip(bars, smooth):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Temporal smoothness (higher = better)")
    ax.set_title("Exp 3: Imagination\n(interpolated sequence smoothness)")

    # ── Panel 4: Schema distortion ────────────────────────────────────────
    ax = axes[1, 1]
    labels = ["Canonical\n(ref)", "Atypical\ninput",
              "Seq\nrecon", "Shuffled\nrecon"]
    vals   = [exp4["canonical_mad"], exp4["atypical_mad"],
              exp4["sequential"]["recon_mad"],
              exp4["shuffled"]["recon_mad"]]
    errs   = [0, exp4["atypical_mad_std"],
              exp4["sequential"]["recon_mad_std"],
              exp4["shuffled"]["recon_mad_std"]]
    colors4 = ["seagreen", "gray", "coral", "steelblue"]
    bars4 = ax.bar(labels, vals, yerr=errs, color=colors4,
                    alpha=0.85, capsize=5, width=0.6)
    for bar, v in zip(bars4, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MAD from canonical sweep (lower = more canonical)")
    ax.set_title("Exp 4: Schema distortion\n"
                 "(does reconstruction pull toward canonical?)")

    plt.suptitle(
        "Sequential replay consolidation experiments\n"
        "Primary comparison: sequential vs shuffled seq-VAE",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "consolidation_experiments.png"), dpi=150)
    plt.close()
    print(f"  Plot saved to {out_dir}/consolidation_experiments.png")