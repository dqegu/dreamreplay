"""
experiment.py
═════════════════════════════════════════════════════════════════════════════
Claim
─────
Sequential replay improves the consolidation of temporally structured
memories relative to itemwise (IID) replay.

We operationalise "better consolidation" using three signatures drawn
directly from Spens & Burgess (2024):

  Exp 1 — Partial-cue episode completion  (analogous to Fig. 1d)
           Give each student the first N frames of a held-out episode.
           Ask it to fill the missing middle segment (frames 5–9) and
           evaluate pixel MSE on those hidden frames.
           A student that has consolidated sequential structure should
           reconstruct the missing segment better than one that only
           learned to reconstruct individual frames.

  Exp 2 — Schema distortion for sequences  (analogous to Fig. 4)
           Feed each student an atypical (perturbed) orientation sequence.
           Ask it to reconstruct the full sequence from a partial cue.
           A student with a sequential schema should pull the recalled
           sequence toward the canonical monotonic trajectory — i.e.
           show temporal schema-based distortion.

  Exp 3 — Temporal-gist semanticization  (analogous to Fig. 3a)
           Probe whether latent representations encode episode-level gist:
           coarse temporal phase (early / middle / late within the episode).
           A student whose replay consolidated temporal structure should
           expose this gist in its latent space; an IID student has no
           such pressure.  Better phase decoding = richer temporal
           semanticization.

All three experiments run on the already-trained student models; no
retraining is required.

The existing factor-probe and reconstruction-MSE evaluations are retained
as supporting context.

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

from config import OUT_DIR, SEQ_LENGTH, PROBE_FA"""
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
    print(f"  Plot saved to {out_dir}/consolidation_experiments.png")CTORS

MIDDLE_FRAME = SEQ_LENGTH // 2   # frame 7

# ═══════════════════════════════════════════════════════════════════════════
# Existing helpers (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(encoder, imgs, sequences):
    frame_feats, episode_feats = [], []
    for seq in sequences:
        clip    = imgs[seq]
        z_means = encoder.predict(clip, batch_size=64, verbose=0)[0]
        frame_feats.append(z_means[MIDDLE_FRAME])
        episode_feats.append(z_means.mean(axis=0))
    return np.stack(frame_feats), np.stack(episode_feats)


def extract_frame_level_features(encoder, imgs, sequences):
    all_z, all_ori = [], []
    for seq in sequences:
        clip    = imgs[seq]
        z_means = encoder.predict(clip, batch_size=64, verbose=0)[0]
        for frame_pos, z in enumerate(z_means):
            all_z.append(z)
            all_ori.append(frame_pos)
    return np.stack(all_z), np.array(all_ori, dtype=int)


def _probe(X_tr, y_tr, X_te, y_te):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    )
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    reg = make_pipeline(StandardScaler(), Ridge())
    reg.fit(X_tr, y_tr)
    r2 = float(r2_score(y_te, reg.predict(X_te)))
    return acc, r2


def reconstruction_mse(encoder, decoder, imgs, sequences, n=200):
    rng   = np.random.default_rng(0)
    idxs  = rng.choice(len(sequences), size=min(n, len(sequences)), replace=False)
    batch = np.concatenate([imgs[sequences[i]] for i in idxs])
    z_mean, _, _ = encoder.predict(batch, batch_size=128, verbose=0)
    recon        = decoder.predict(z_mean, batch_size=128, verbose=0)
    return float(np.mean((batch - recon) ** 2))


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1A — Probe-based episode completion (fully fair)
# ═══════════════════════════════════════════════════════════════════════════

def exp1a_probe_completion(models, train_imgs, train_seqs,
                            test_imgs, test_seqs,
                            prefix_len=5, gap_start=5, gap_end=10,
                            n_train=800, n_test=200, rng_seed=0):
    """
    Probe-based gap-filling task  (Exp 1A — fully fair comparison).

    Both students are evaluated through an IDENTICAL downstream predictor
    fitted on top of their frozen representations.  Neither student's own
    decoder or transition model is used.  This removes both decoder bias
    (sequential has a blurrier decoder) and mechanism bias (sequential has
    a transition MLP, IID does not).

    Protocol
    ────────
    For each student encoder independently:

      1. Represent each sequence's prefix (frames 0–prefix_len-1) as the
         CONCATENATION of its per-frame latents → shape (prefix_len × latent_dim,).

      2. Represent the gap target as the CONCATENATION of the TEACHER's
         latents for frames gap_start–gap_end-1 → shape (gap_len × latent_dim,).
         The teacher's latent space is used as a shared, objective-neutral
         reference; neither student's encoder enters the target.

      3. Fit a Ridge regression probe mapping prefix representation → gap
         target on TRAINING sequences.

      4. Evaluate MSE of the probe's predictions on held-out TEST sequences.

    Interpretation
    ──────────────
    A lower test MSE means the learned prefix representation contains more
    information about the upcoming gap — i.e. the encoder has consolidated
    more sequential structure regardless of its decoding ability.

    This is the fairest possible Exp 1: same input, same predictor, same
    target, same metric for both students.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    rng       = np.random.default_rng(rng_seed)
    gap_len   = gap_end - gap_start
    latent_dim = 32  # matches LATENT_DIM in config

    teacher_enc = models["teacher_enc"]
    encoders = {
        "sequential":  models["seq_enc"],
        "iid_matched": models["iid_enc"],
    }

    # ── Pre-compute teacher latents for gap frames (shared target) ─────────
    def _teacher_gap_target(imgs, seqs, max_n=None):
        seqs_ = seqs[:max_n] if max_n else seqs
        targets = []
        for seq in seqs_:
            frames    = imgs[seq]
            gap_f     = frames[gap_start:gap_end]           # (gap_len, 64,64,3)
            t_z       = teacher_enc.predict(
                gap_f, batch_size=gap_len, verbose=0)[0]    # (gap_len, latent_dim)
            targets.append(t_z.flatten())                   # (gap_len*latent_dim,)
        return np.stack(targets)

    print("    Building teacher gap targets (train)...")
    Y_train = _teacher_gap_target(train_imgs, train_seqs, n_train)
    print("    Building teacher gap targets (test)...")
    Y_test  = _teacher_gap_target(test_imgs,  test_seqs,  n_test)

    results = {}

    for name, enc in encoders.items():
        # ── Build prefix representations ────────────────────────────────────
        def _prefix_repr(imgs, seqs, max_n=None):
            seqs_ = seqs[:max_n] if max_n else seqs
            reprs = []
            for seq in seqs_:
                frames  = imgs[seq]
                prefix  = frames[:prefix_len]               # (prefix_len,64,64,3)
                z_pre   = enc.predict(
                    prefix, batch_size=prefix_len, verbose=0)[0]  # (prefix_len, d)
                reprs.append(z_pre.flatten())               # (prefix_len*latent_dim,)
            return np.stack(reprs)

        print(f"    [{name}] encoding prefix representations...")
        X_train = _prefix_repr(train_imgs, train_seqs, n_train)
        X_test  = _prefix_repr(test_imgs,  test_seqs,  n_test)

        # ── Fit identical Ridge probe ────────────────────────────────────────
        probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        probe.fit(X_train, Y_train)
        Y_pred = probe.predict(X_test)

        # Per-sequence latent MSE
        seq_mses = np.mean((Y_test - Y_pred) ** 2, axis=1)  # (n_test,)
        results[name] = {
            "latent_mse":     float(np.mean(seq_mses)),
            "latent_mse_std": float(np.std(seq_mses)),
        }
        print(f"    [{name}] probe latent MSE = {results[name]['latent_mse']:.5f} "
              f"± {results[name]['latent_mse_std']:.5f}")

    results["seq_wins"] = int(
        results["sequential"]["latent_mse"] < results["iid_matched"]["latent_mse"])
    results["seq_vs_iid"] = float(
        results["sequential"]["latent_mse"] - results["iid_matched"]["latent_mse"])
    results["n_test"] = n_test

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1B — Native generative completion (behavioural, kept for insight)
# ═══════════════════════════════════════════════════════════════════════════

def exp1b_generative_completion(models, test_imgs, test_seqs,
                                 prefix_len=5, gap_start=5, gap_end=10,
                                 n_seqs=200, rng_seed=0):
    """
    Native generative gap-filling  (Exp 1B — behavioural comparison).

    Each student uses its own native mechanism to complete the gap.
    This is NOT a fair architectural comparison — the sequential student
    has a transition MLP while the IID student does not — but it reveals
    what each model actually *does* when asked to complete a sequence.
    Results should be interpreted as a characterisation of each model's
    generative behaviour, not as a controlled test of consolidation.

    Sequential : encode last prefix frame → roll transition MLP forward
                 → decode each step.
    IID-matched : encode last prefix frame → decode → tile across gap.

    Two metrics:
      Pixel MSE  — raw pixel comparison (biased toward IID's sharper decoder).
      Latent MSE — both outputs re-encoded through the teacher; removes
                   decoder sharpness bias but not mechanism bias.
    """
    rng  = np.random.default_rng(rng_seed)
    idxs = rng.choice(len(test_seqs), size=min(n_seqs, len(test_seqs)),
                      replace=False)
    gap_len = gap_end - gap_start

    teacher_enc = models["teacher_enc"]
    seq_enc     = models["seq_enc"]
    seq_dec     = models["seq_dec"]
    seq_trans   = models["seq_trans"]
    iid_enc     = models["iid_enc"]
    iid_dec     = models["iid_dec"]

    seq_pixel_errors,  iid_pixel_errors  = [], []
    seq_latent_errors, iid_latent_errors = [], []

    for i in idxs:
        frames     = test_imgs[test_seqs[i]]
        gap_frames = frames[gap_start:gap_end]               # (gap_len,64,64,3)

        # Teacher latents for ground-truth gap (shared reference)
        gt_z = teacher_enc.predict(
            gap_frames, batch_size=gap_len, verbose=0)[0]   # (gap_len, latent_dim)

        # ── Sequential rollout ───────────────────────────────────────────────
        last_prefix = frames[prefix_len - 1:prefix_len]
        z = seq_enc.predict(last_prefix, verbose=0)[0]
        predicted_gap, pred_z_seq = [], []
        for _ in range(gap_len):
            z          = seq_trans.predict(z, verbose=0)
            frame_pred = seq_dec.predict(z, verbose=0)
            predicted_gap.append(frame_pred[0])
            tz = teacher_enc.predict(frame_pred, verbose=0)[0]
            pred_z_seq.append(tz[0])
        predicted_gap = np.stack(predicted_gap)
        pred_z_seq    = np.stack(pred_z_seq)

        seq_pixel_errors.append(float(np.mean((gap_frames - predicted_gap) ** 2)))
        seq_latent_errors.append(float(np.mean((gt_z - pred_z_seq) ** 2)))

        # ── IID tiled reconstruction ─────────────────────────────────────────
        z_iid     = iid_enc.predict(last_prefix, verbose=0)[0]
        recon     = iid_dec.predict(z_iid, verbose=0)
        iid_tiled = np.repeat(recon, gap_len, axis=0)

        iid_pixel_errors.append(float(np.mean((gap_frames - iid_tiled) ** 2)))
        iid_z = teacher_enc.predict(
            iid_tiled, batch_size=gap_len, verbose=0)[0]
        iid_latent_errors.append(float(np.mean((gt_z - iid_z) ** 2)))

    return {
        "seq_mse":         float(np.mean(seq_pixel_errors)),
        "seq_mse_std":     float(np.std(seq_pixel_errors)),
        "iid_mse":         float(np.mean(iid_pixel_errors)),
        "iid_mse_std":     float(np.std(iid_pixel_errors)),
        "seq_wins_pixel":  int(np.sum(np.array(seq_pixel_errors) <
                                      np.array(iid_pixel_errors))),
        "seq_latent_mse":  float(np.mean(seq_latent_errors)),
        "seq_latent_std":  float(np.std(seq_latent_errors)),
        "iid_latent_mse":  float(np.mean(iid_latent_errors)),
        "iid_latent_std":  float(np.std(iid_latent_errors)),
        "seq_wins_latent": int(np.sum(np.array(seq_latent_errors) <
                                      np.array(iid_latent_errors))),
        "n_seqs":          len(seq_pixel_errors),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2 — Schema distortion for sequences
# ═══════════════════════════════════════════════════════════════════════════

def _make_atypical_sequence(frames, rng, n_perturb=5):
    """
    Create an atypical orientation sequence by randomly swapping n_perturb
    frame pairs drawn from anywhere in the FULL sequence, including the
    prefix cue region.  Students therefore receive a genuinely disordered
    cue — atypicality is not confined to the hidden segment.

    Non-adjacent pairs are sampled so the permutation is stronger than
    adjacent swaps.

    Returns the perturbed frame array (same shape) and the permuted
    orientation indices for the full sequence.
    """
    perturbed   = frames.copy()
    ori_indices = np.arange(SEQ_LENGTH)

    n_actual = min(n_perturb, SEQ_LENGTH // 2)
    all_pos  = np.arange(SEQ_LENGTH)
    rng.shuffle(all_pos)
    pairs = all_pos[: n_actual * 2].reshape(n_actual, 2)

    for a, b in pairs:
        perturbed[[a, b]]       = perturbed[[b, a]]
        ori_indices[a], ori_indices[b] = int(ori_indices[b]), int(ori_indices[a])

    return perturbed, ori_indices


def _orientation_from_frame(frame, teacher_enc, all_ori_z, all_ori_indices):
    """
    Estimate the orientation index of a decoded frame by finding the nearest
    neighbour in the teacher's latent space among the 15 canonical orientation
    frames for that group.

    all_ori_z      : (15, latent_dim) teacher latents for the canonical sweep
    all_ori_indices: array [0,1,...,14] corresponding orientation indices

    Returns the estimated orientation index (int 0–14).
    """
    z = teacher_enc.predict(frame[np.newaxis], verbose=0)[0][0]  # (latent_dim,)
    dists = np.linalg.norm(all_ori_z - z, axis=1)               # (15,)
    return int(all_ori_indices[np.argmin(dists)])


def _mad_from_canonical(ori_trajectory):
    """
    Mean Absolute Deviation of an orientation trajectory from the
    canonical sweep [0,1,2,...,14].

    Lower = more canonical / more schema-consistent.
    """
    canonical = np.arange(SEQ_LENGTH)
    return float(np.mean(np.abs(np.array(ori_trajectory) - canonical)))


def exp2_schema_distortion(models, test_imgs, test_seqs,
                            prefix_len=5, n_seqs=200,
                            n_perturb=3, rng_seed=1):
    """
    Schema distortion test — directly measuring orientation trajectories.

    For each held-out sequence:
      1. Build canonical orientation latents from the teacher (ground truth).
      2. Create an atypical version by swapping n_perturb adjacent frame pairs.
      3. Give each student the first prefix_len frames of the atypical sequence.
      4. Sequential student: roll transition MLP forward to complete the episode.
         IID student: tile the last prefix frame across the remaining positions.
      5. For each completed sequence, recover the orientation index of each
         frame via nearest-neighbour lookup in the teacher's latent space.
      6. Compute Mean Absolute Deviation (MAD) from the canonical sweep [0..14].

    Lower MAD = recalled sequence closer to canonical = stronger schema distortion.

    Key comparisons (all on the same orientation-index scale):
      canonical_mad  = 0 by definition (canonical sweep vs itself)
      atypical_mad   = MAD of the perturbed input trajectory (baseline distortion)
      seq_mad        = MAD of sequential student's completion
      iid_mad        = MAD of IID student's completion

    If seq_mad < iid_mad AND seq_mad < atypical_mad, the sequential student
    has pulled recall toward the canonical schema — temporal schema distortion.
    """
    rng  = np.random.default_rng(rng_seed)
    idxs = rng.choice(len(test_seqs), size=min(n_seqs, len(test_seqs)),
                      replace=False)
    gap_len = SEQ_LENGTH - prefix_len

    teacher_enc = models["teacher_enc"]
    seq_enc     = models["seq_enc"]
    seq_dec     = models["seq_dec"]
    seq_trans   = models["seq_trans"]
    iid_enc     = models["iid_enc"]
    iid_dec     = models["iid_dec"]

    # Pre-build the canonical orientation-index lookup for each test group.
    # For group i, all_ori_z[i] = teacher latents of its 15 canonical frames.
    all_ori_z = []
    for i in idxs:
        frames = test_imgs[test_seqs[i]]
        z_group = teacher_enc.predict(frames, batch_size=15, verbose=0)[0]
        all_ori_z.append(z_group)   # (15, latent_dim)

    canonical_indices = np.arange(SEQ_LENGTH)

    atyp_mads = []
    seq_mads  = []
    iid_mads  = []

    for k, i in enumerate(idxs):
        true_frames = test_imgs[test_seqs[i]]
        ori_z_group = all_ori_z[k]          # (15, latent_dim) teacher latents

        # ── Build atypical sequence ───────────────────────────────────────────
        atyp_frames, atyp_ori = _make_atypical_sequence(true_frames, rng,
                                                          n_perturb, prefix_len)
        # MAD of atypical input from canonical
        atyp_mads.append(_mad_from_canonical(atyp_ori))

        last_prefix = atyp_frames[prefix_len - 1:prefix_len]  # (1, 64, 64, 3)

        # ── Sequential completion ─────────────────────────────────────────────
        z = seq_enc.predict(last_prefix, verbose=0)[0]         # (1, latent_dim)
        seq_ori_traj = list(atyp_ori[:prefix_len])
        for _ in range(gap_len):
            z = seq_trans.predict(z, verbose=0)
            frame_pred = seq_dec.predict(z, verbose=0)          # (1, 64, 64, 3)
            ori_est = _orientation_from_frame(
                frame_pred[0], teacher_enc,
                all_ori_z=ori_z_group,
                all_ori_indices=canonical_indices)
            seq_ori_traj.append(ori_est)
        seq_mads.append(_mad_from_canonical(seq_ori_traj))

        # ── IID completion (tile last prefix frame) ───────────────────────────
        z_iid  = iid_enc.predict(last_prefix, verbose=0)[0]
        recon  = iid_dec.predict(z_iid, verbose=0)             # (1, 64, 64, 3)
        iid_ori_traj = list(atyp_ori[:prefix_len])
        for _ in range(gap_len):
            ori_est = _orientation_from_frame(
                recon[0], teacher_enc,
                all_ori_z=ori_z_group,
                all_ori_indices=canonical_indices)
            iid_ori_traj.append(ori_est)
        iid_mads.append(_mad_from_canonical(iid_ori_traj))

    return {
        # Canonical MAD is 0 by construction — included for reference only.
        "canonical_mad":   0.0,
        "atypical_mad":    float(np.mean(atyp_mads)),
        "atypical_mad_std": float(np.std(atyp_mads)),
        "seq_mad":         float(np.mean(seq_mads)),
        "seq_mad_std":     float(np.std(seq_mads)),
        "iid_mad":         float(np.mean(iid_mads)),
        "iid_mad_std":     float(np.std(iid_mads)),
        # Negative = seq MORE canonical than atypical input (pulled toward schema)
        "seq_schema_pull": float(np.mean(seq_mads)  - np.mean(atyp_mads)),
        "iid_schema_pull": float(np.mean(iid_mads)  - np.mean(atyp_mads)),
        # Negative = seq more canonical than IID
        "seq_vs_iid":      float(np.mean(seq_mads)  - np.mean(iid_mads)),
        "seq_wins":        int(np.sum(np.array(seq_mads) < np.array(iid_mads))),
        "n_seqs":          len(seq_mads),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3 — Temporal-gist semanticization
# ═══════════════════════════════════════════════════════════════════════════

def exp3_temporal_gist(models, test_imgs, test_seqs,
                        train_imgs, train_seqs,
                        n_seqs_probe=800, rng_seed=2):
    """
    Temporal-gist semanticization  (analogous to Spens & Burgess Fig. 3a).

    Probes whether consolidated latent representations encode episode-level
    temporal gist — specifically, coarse phase within the episode.

    Task: Phase decoding
      Label each frame as early (positions 0–4), middle (5–9), or late (10–14).
      Train a logistic regression probe on per-frame latents from training
      sequences. Evaluate on held-out test sequences.

      A student whose replay consolidated temporal structure should expose
      phase information in its latent space, because its training objective
      required tracking where in the sequence each frame belongs.
      An IID student has no such pressure — individual frames are presented
      in random order with no temporal context.

    This is the sequential analogue of Spens Fig. 3a, where semantic decoding
    accuracy from latent variables increases as the generative model consolidates.
    Here, the "semantic fact" being decoded is the frame's temporal position
    within the episode rather than its visual category.

    Chance level = 1/3 ≈ 0.33.
    All students evaluated with an identical probe on the same test data.
    """
    students = {
        "teacher":     models["teacher_enc"],
        "iid_matched": models["iid_enc"],
        "iid_mhn":     models["iid_mhn_enc"],
        "sequential":  models["seq_enc"],
    }

    max_tr = n_seqs_probe
    max_te = n_seqs_probe // 4

    def _build_phase_data(encoder, imgs, seqs, max_seqs=None):
        seqs_ = seqs[:max_seqs] if max_seqs else seqs
        all_z, all_phase = [], []
        for seq in seqs_:
            frames = imgs[seq]
            z = encoder.predict(frames, batch_size=15, verbose=0)[0]  # (15, d)
            for pos, zi in enumerate(z):
                all_z.append(zi)
                # 3-class coarse phase label
                phase = 0 if pos < 5 else (1 if pos < 10 else 2)
                all_phase.append(phase)
        return np.stack(all_z), np.array(all_phase, dtype=int)

    results = {}
    for name, enc in students.items():
        Xtr, ytr = _build_phase_data(enc, train_imgs, train_seqs, max_tr)
        Xte, yte = _build_phase_data(enc, test_imgs,  test_seqs,  max_te)
        phase_acc, phase_r2 = _probe(Xtr, ytr, Xte, yte)
        results[name] = {"phase_acc": phase_acc, "phase_r2": phase_r2}
        print(f"    {name:<14}  phase acc={phase_acc:.3f}  r2={phase_r2:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════════

def run(models: dict,
        train_imgs, train_seqs, train_labels: dict,
        test_imgs,  test_seqs,  test_labels:  dict,
        seed: int):

    out_dir = os.path.join(OUT_DIR, f"exp_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    # ── Existing: factor probes + reconstruction MSE ─────────────────────────
    print("\n── Factor probes (existing evaluation) ──")
    student_pairs = {
        "teacher":     (models["teacher_enc"],  models["teacher_dec"]),
        "iid_matched": (models["iid_enc"],       models["iid_dec"]),
        "iid_mhn":     (models["iid_mhn_enc"],   models["iid_mhn_dec"]),
        "sequential":  (models["seq_enc"],       models["seq_dec"]),
    }

    probe_results = {}
    for student_name, (enc, dec) in student_pairs.items():
        print(f"\n  Evaluating: {student_name}")
        recon_mse = reconstruction_mse(enc, dec, test_imgs, test_seqs)
        print(f"    Reconstruction MSE = {recon_mse:.5f}")

        tr_frame, tr_ep = extract_features(enc, train_imgs, train_seqs)
        te_frame, te_ep = extract_features(enc, test_imgs,  test_seqs)
        tr_fl_z, tr_fl_ori = extract_frame_level_features(enc, train_imgs, train_seqs)
        te_fl_z, te_fl_ori = extract_frame_level_features(enc, test_imgs,  test_seqs)

        pres = {}
        for factor in PROBE_FACTORS:
            if factor == "orientation":
                acc_f, r2_f = _probe(tr_fl_z, tr_fl_ori, te_fl_z, te_fl_ori)
                pres[factor] = {"frame_acc": acc_f, "frame_r2": r2_f,
                                "episode_acc": float("nan"), "episode_r2": float("nan")}
                print(f"    {'orientation':<12}  frame acc={acc_f:.3f}  (frame-level)")
                continue
            if factor not in train_labels:
                continue
            acc_f, r2_f = _probe(tr_frame, train_labels[factor], te_frame, test_labels[factor])
            acc_e, r2_e = _probe(tr_ep,    train_labels[factor], te_ep,    test_labels[factor])
            pres[factor] = {"frame_acc": acc_f, "frame_r2": r2_f,
                            "episode_acc": acc_e, "episode_r2": r2_e}
            print(f"    {factor:<12}  frame acc={acc_f:.3f}  episode acc={acc_e:.3f}")

        probe_results[student_name] = {"recon_mse": recon_mse, "probes": pres}

    all_results["factor_probes"] = probe_results

    # ── Experiment 1: Partial-cue episode completion ──────────────────────────
    # ── Experiment 1A: Probe-based episode completion (fully fair) ───────────
    print("\n── Exp 1A: Probe-based episode completion (fully fair) ──")
    exp1a = exp1a_probe_completion(models,
                                    train_imgs, train_seqs,
                                    test_imgs,  test_seqs)
    all_results["exp1a_probe"] = exp1a
    winner1a = "SEQ ✓" if exp1a["seq_vs_iid"] < 0 else "IID"
    print(f"  Sequential latent MSE:  {exp1a['sequential']['latent_mse']:.5f} "
          f"± {exp1a['sequential']['latent_mse_std']:.5f}")
    print(f"  IID-matched latent MSE: {exp1a['iid_matched']['latent_mse']:.5f} "
          f"± {exp1a['iid_matched']['latent_mse_std']:.5f}")
    print(f"  {winner1a} (advantage = {exp1a['seq_vs_iid']:+.5f})")

    # ── Experiment 1B: Native generative completion (behavioural) ────────────
    print("\n── Exp 1B: Native generative completion (behavioural) ──")
    exp1b = exp1b_generative_completion(models, test_imgs, test_seqs)
    all_results["exp1b_generative"] = exp1b
    print(f"  Pixel MSE  — Seq: {exp1b['seq_mse']:.5f} ± {exp1b['seq_mse_std']:.5f} "
          f"| IID: {exp1b['iid_mse']:.5f} ± {exp1b['iid_mse_std']:.5f} "
          f"| Seq wins {exp1b['seq_wins_pixel']}/{exp1b['n_seqs']}")
    print(f"  Latent MSE — Seq: {exp1b['seq_latent_mse']:.5f} ± {exp1b['seq_latent_std']:.5f} "
          f"| IID: {exp1b['iid_latent_mse']:.5f} ± {exp1b['iid_latent_std']:.5f} "
          f"| Seq wins {exp1b['seq_wins_latent']}/{exp1b['n_seqs']}")

    # ── Experiment 2: Schema distortion for sequences ─────────────────────────
    print("\n── Exp 2: Schema distortion for sequences ──")
    exp2 = exp2_schema_distortion(models, test_imgs, test_seqs)
    all_results["exp2_schema"] = exp2
    print(f"  Canonical MAD (reference):    {exp2['canonical_mad']:.3f}")
    print(f"  Atypical input MAD:           {exp2['atypical_mad']:.3f} "
          f"± {exp2['atypical_mad_std']:.3f}")
    print(f"  Seq completion MAD:           {exp2['seq_mad']:.3f} "
          f"± {exp2['seq_mad_std']:.3f}")
    print(f"  IID completion MAD:           {exp2['iid_mad']:.3f} "
          f"± {exp2['iid_mad_std']:.3f}")
    print(f"  Seq schema pull (vs atypical): {exp2['seq_schema_pull']:+.3f}")
    print(f"  IID schema pull (vs atypical): {exp2['iid_schema_pull']:+.3f}")
    print(f"  Seq vs IID MAD advantage:      {exp2['seq_vs_iid']:+.3f} "
          f"(seq wins {exp2['seq_wins']}/{exp2['n_seqs']} sequences)")

    # ── Experiment 3: Temporal-gist semanticization ───────────────────────────
    print("\n── Exp 3: Temporal-gist semanticization ──")
    exp3 = exp3_temporal_gist(models, test_imgs, test_seqs,
                               train_imgs, train_seqs)
    all_results["exp3_gist"] = exp3

    # ── Save all results ──────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    _print_summary(probe_results)
    _print_consolidation_summary(exp1a, exp1b, exp2, exp3)
    _save_plots(probe_results, out_dir)
    _save_consolidation_plots(exp1a, exp1b, exp2, exp3, out_dir)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════════════

def _print_summary(probe_results):
    factors = list(PROBE_FACTORS.keys())
    print(f"\n{'─'*80}")
    print("EXISTING EVALUATION: factor probe accuracy (test set)")
    print(f"{'─'*80}")
    for rep in ("frame", "episode"):
        print(f"\n  [{rep} representation]")
        print(f"  {'Factor':<14}  {'Teacher':>9}  {'IID-match':>10}  "
              f"{'IID-MHN':>9}  {'Seq':>7}  {'Winner':>8}")
        print("  " + "─" * 68)
        for f in factors:
            if f == "orientation" and rep == "episode":
                continue
            def _g(name):
                return probe_results[name]["probes"].get(
                    f, {}).get(f"{rep}_acc", float("nan"))
            t, im, mhn, s = (_g("teacher"), _g("iid_matched"),
                             _g("iid_mhn"), _g("sequential"))
            winner = "SEQ ✓" if s > im else "IID-match"
            note   = " [frame-level]" if f == "orientation" else ""
            print(f"  {f:<14}  {t:9.3f}  {im:10.3f}  {mhn:9.3f}  "
                  f"{s:7.3f}  {winner:>8}{note}")
    print(f"\n  Reconstruction MSE:")
    for name in ("teacher", "iid_matched", "iid_mhn", "sequential"):
        print(f"    {name:<14}: {probe_results[name]['recon_mse']:.5f}")


def _print_consolidation_summary(exp1a, exp1b, exp2, exp3):
    print(f"\n{'═'*80}")
    print("CONSOLIDATION EXPERIMENTS: Sequential vs IID-matched")
    print(f"{'═'*80}")

    print(f"\n  Exp 1A — Probe-based completion (fully fair, identical ridge probe)")
    winner = "SEQ ✓" if exp1a["seq_vs_iid"] < 0 else "IID"
    print(f"    Sequential latent MSE:  {exp1a['sequential']['latent_mse']:.5f} "
          f"± {exp1a['sequential']['latent_mse_std']:.5f}")
    print(f"    IID-matched latent MSE: {exp1a['iid_matched']['latent_mse']:.5f} "
          f"± {exp1a['iid_matched']['latent_mse_std']:.5f}")
    print(f"    {winner} (advantage = {exp1a['seq_vs_iid']:+.5f})")

    print(f"\n  Exp 1B — Native generative completion (behavioural, not fair comparison)")
    print(f"    Pixel MSE:  Seq={exp1b['seq_mse']:.5f}  IID={exp1b['iid_mse']:.5f}  "
          f"Seq wins {exp1b['seq_wins_pixel']}/{exp1b['n_seqs']}")
    print(f"    Latent MSE: Seq={exp1b['seq_latent_mse']:.5f}  "
          f"IID={exp1b['iid_latent_mse']:.5f}  "
          f"Seq wins {exp1b['seq_wins_latent']}/{exp1b['n_seqs']}")

    print(f"\n  Exp 2 — Schema distortion (MAD from canonical orientation sweep)")
    print(f"    Canonical MAD (reference):    {exp2['canonical_mad']:.3f}")
    print(f"    Atypical input MAD:           {exp2['atypical_mad']:.3f} "
          f"± {exp2['atypical_mad_std']:.3f}")
    print(f"    Seq completion MAD:           {exp2['seq_mad']:.3f} "
          f"± {exp2['seq_mad_std']:.3f}")
    print(f"    IID completion MAD:           {exp2['iid_mad']:.3f} "
          f"± {exp2['iid_mad_std']:.3f}")
    win2_vs_iid   = exp2["seq_vs_iid"] < 0
    win2_vs_atyp  = exp2["seq_schema_pull"] < 0
    print(f"    Seq schema pull (vs atypical): {exp2['seq_schema_pull']:+.3f} "
          f"({'toward canonical ✓' if win2_vs_atyp else 'away from canonical'})")
    print(f"    IID schema pull (vs atypical): {exp2['iid_schema_pull']:+.3f}")
    print(f"    Seq vs IID MAD advantage:      {exp2['seq_vs_iid']:+.3f} "
          f"({'SEQ MORE CANONICAL ✓' if win2_vs_iid else 'IID more canonical'})")
    print(f"    Seq wins on {exp2['seq_wins']}/{exp2['n_seqs']} sequences")

    print(f"\n  Exp 3 — Temporal-gist semanticization (phase decoding, chance=0.33)")
    print(f"  {'Student':<14}  {'Phase acc':>10}  {'Phase R²':>10}")
    print("  " + "─" * 38)
    for name in ("teacher", "iid_matched", "iid_mhn", "sequential"):
        r = exp3[name]
        print(f"  {name:<14}  {r['phase_acc']:10.3f}  {r['phase_r2']:10.3f}")
    seq_wins = exp3["sequential"]["phase_acc"] > exp3["iid_matched"]["phase_acc"]
    print(f"    Phase decoding: {'SEQ ✓' if seq_wins else 'IID'}")


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

_CONDITIONS = [
    ("teacher",     "seagreen",     "Teacher (upper bound)"),
    ("iid_matched", "steelblue",    "IID-matched (primary)"),
    ("iid_mhn",     "mediumpurple", "IID-MHN (Spens & Burgess)"),
    ("sequential",  "coral",        "Sequential (ours)"),
]


def _save_plots(probe_results, out_dir):
    """Existing factor-probe bar charts."""
    factors = [f for f in PROBE_FACTORS
               if f in probe_results["iid_matched"]["probes"]]
    x      = np.arange(len(factors))
    n_cond = len(_CONDITIONS)
    width  = 0.8 / n_cond
    offsets = np.linspace(-(n_cond-1)/2, (n_cond-1)/2, n_cond) * width

    for rep in ("frame", "episode"):
        rep_factors = [f for f in factors
                       if not (f == "orientation" and rep == "episode")]
        x_rep = np.arange(len(rep_factors))
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        for ax_idx, metric_key, ylabel in [
            (0, f"{rep}_acc", "Probe accuracy (test set)"),
            (1, f"{rep}_r2",  "R² (Ridge, test set)"),
        ]:
            ax = axes[ax_idx]
            for (name, color, label), offset in zip(_CONDITIONS, offsets):
                vals = [probe_results[name]["probes"].get(
                            f, {}).get(metric_key, 0) for f in rep_factors]
                ax.bar(x_rep + offset, vals, width, label=label,
                       color=color, alpha=0.85)
            ax.set_xticks(x_rep)
            ax.set_xticklabels(
                [f + " *" if f == "orientation" else f for f in rep_factors],
                rotation=20, ha="right")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            if metric_key.endswith("_acc"):
                ax.set_ylim(0, 1.05)
                for f in rep_factors:
                    ax.axhline(1.0 / PROBE_FACTORS[f], ls=":", c="gray",
                               lw=0.7, alpha=0.4)
            else:
                ax.axhline(0, ls="--", c="gray", lw=0.8)

        plt.suptitle(f"Factor probe accuracy — {rep} representation\n"
                     "Test set (disjoint groups)", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"probes_{rep}.png"), dpi=150)
        plt.close()


def _save_consolidation_plots(exp1a, exp1b, exp2, exp3, out_dir):
    """Four-panel figure: 1A probe, 1B generative, schema distortion, phase gist."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # ── Panel 1: Exp 1A probe-based completion ────────────────────────────────
    ax = axes[0]
    vals   = [exp1a["iid_matched"]["latent_mse"],
              exp1a["sequential"]["latent_mse"]]
    errs   = [exp1a["iid_matched"]["latent_mse_std"],
              exp1a["sequential"]["latent_mse_std"]]
    colors_1a = ["steelblue", "coral"]
    bars = ax.bar(["IID-matched", "Sequential"], vals,
                  color=colors_1a, alpha=0.85, width=0.5)
    ax.errorbar(["IID-matched", "Sequential"], vals, yerr=errs,
                fmt="none", color="black", capsize=5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Probe latent MSE (lower = better)")
    ax.set_title("Exp 1A: Probe-based completion\n"
                 "(identical ridge probe, teacher latent target — fully fair)")

    # ── Panel 2: Exp 1B native generative ────────────────────────────────────
    ax = axes[1]
    x_pos = np.array([0, 1, 3, 4])
    vals2  = [exp1b["iid_mse"], exp1b["seq_mse"],
              exp1b["iid_latent_mse"], exp1b["seq_latent_mse"]]
    errs2  = [exp1b["iid_mse_std"], exp1b["seq_mse_std"],
              exp1b["iid_latent_std"], exp1b["seq_latent_std"]]
    colors_1b = ["steelblue", "coral", "steelblue", "coral"]
    bars2 = ax.bar(x_pos, vals2, color=colors_1b, alpha=0.85, width=0.7)
    ax.errorbar(x_pos, vals2, yerr=errs2, fmt="none", color="black", capsize=5)
    for bar, v in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vals2)*0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["IID\n(pixel)", "Seq\n(pixel)",
                         "IID\n(latent)", "Seq\n(latent)"], fontsize=8)
    ax.set_ylabel("MSE (lower = better)")
    ax.set_title("Exp 1B: Native generative completion\n"
                 "(behavioural — architecturally asymmetric)")

    # ── Panel 3: Schema distortion — MAD from canonical ──────────────────────
    ax = axes[2]
    labels = ["Canonical\n(ref)", "Atypical\ninput",
              "IID\ncompletion", "Seq\ncompletion"]
    vals   = [exp2["canonical_mad"], exp2["atypical_mad"],
              exp2["iid_mad"],       exp2["seq_mad"]]
    colors = ["seagreen", "gray", "steelblue", "coral"]
    errs   = [0, exp2["atypical_mad_std"],
              exp2["iid_mad_std"], exp2["seq_mad_std"]]
    bars2  = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    ax.errorbar(labels, vals, yerr=errs, fmt="none", color="black", capsize=5)
    for bar, v in zip(bars2, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MAD from canonical sweep (lower = more canonical)")
    ax.set_title("Exp 2: Schema distortion\n"
                 "(does recall pull toward canonical orientation trajectory?)")

    # ── Panel 4: Temporal gist — phase accuracy only ─────────────────────────
    ax     = axes[3]
    names  = ["teacher", "iid_matched", "iid_mhn", "sequential"]
    labels3 = ["Teacher", "IID-match", "IID-MHN", "Sequential"]
    colors3 = ["seagreen", "steelblue", "mediumpurple", "coral"]
    phase_accs = [exp3[n]["phase_acc"] for n in names]
    bars3 = ax.bar(labels3, phase_accs, color=colors3, alpha=0.85, width=0.5)
    for bar, v in zip(bars3, phase_accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(1/3, ls="--", c="gray", lw=1.0, label="Chance (0.33)")
    ax.set_ylabel("Phase decoding accuracy (test set)")
    ax.set_title("Exp 3: Temporal-gist semanticization\n"
                 "(early / middle / late phase decoded from frame latents)")
    ax.legend(fontsize=8)

    plt.suptitle(
        "Consolidation experiments: Sequential vs IID replay\n"
        "Claim: sequential replay improves consolidation of temporally "
        "structured memories",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "consolidation_experiments.png"), dpi=150)
    plt.close()
    print(f"  Consolidation plot saved.")