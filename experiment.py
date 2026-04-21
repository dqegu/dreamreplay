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

from config import OUT_DIR, SEQ_LENGTH, PROBE_FACTORS

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

def _make_atypical_sequence(frames, rng, n_perturb=3, prefix_len=5):
    """
    Create an atypical orientation sequence by randomly swapping n_perturb
    pairs of adjacent frames within the HIDDEN segment only (positions
    >= prefix_len). The visible prefix is left intact so both students
    receive a clean, canonical cue — any atypicality is purely in the
    part they must reconstruct.

    Returns the perturbed frame array (same shape) and the permuted
    orientation indices for the full sequence.
    """
    perturbed   = frames.copy()
    ori_indices = list(range(SEQ_LENGTH))

    # Only swap within the hidden suffix so the prefix cue is always clean.
    hidden_positions = np.arange(prefix_len, SEQ_LENGTH - 1)
    n_actual = min(n_perturb, len(hidden_positions))

    if n_actual > 0:
        swap_positions = rng.choice(hidden_positions, size=n_actual,
                                    replace=False)
        for pos in swap_positions:
            perturbed[[pos, pos + 1]] = perturbed[[pos + 1, pos]]
            ori_indices[pos], ori_indices[pos + 1] = (ori_indices[pos + 1],
                                                       ori_indices[pos])
    return perturbed, np.array(ori_indices)


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