"""
Training routines for teacher VAE and all three student models.

Pipeline order (important for fairness)
────────────────────────────────────────
1. Teacher VAE — trained on real training images.
2. K/V memory  — built from teacher latents of training sequences.
3. Sequential replay generated (frames_t, frames_t1).
4. IID-matched student — trains on pooled+shuffled frames from step 3.
   Same pixels as sequential, different order. Primary comparison.
5. IID-MHN student — trains on MHN-retrieved frames. Replicates Spens &
   Burgess exactly; kept as secondary reference only.
6. Sequential student — trains on ordered (frames_t, frames_t1) pairs.
"""

import os
import numpy as np
import tensorflow as tf"""
training.py
─────────────────────────────────────────────────────────────────────────────
Training pipeline.

Order:
  1. Teacher VAE          — trained on real frames, frozen thereafter
  2. K/V heteroassociative memory  — built from teacher latents
  3. Sequential replay sequences generated from K/V memory
  4. Sequence VAE student (sequential)  — trained on ordered replay sequences
     Checkpoints saved at epochs 5, 10, 20, 30 for Exp 2 (semanticisation)
  5. Sequence VAE student (shuffled ablation)  — same arch, shuffled sequences
  6. IID frame VAE student  — original Spens & Burgess baseline

Baselines:
  Primary comparison:   sequential seq-VAE  vs  shuffled seq-VAE
    (isolates the effect of temporal order, same architecture)
  Secondary comparison: sequential seq-VAE  vs  IID frame VAE
    (compares to the original Spens & Burgess model)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    LATENT_DIM, SEQ_LATENT_DIM, BATCH_SIZE,
    TEACHER_EPOCHS, STUDENT_EPOCHS, STUDENT_KL_WEIGHT,
    CHECKPOINT_EPOCHS, SEQ_LENGTH,
    TEACHER_ENC_PATH, TEACHER_DEC_PATH, K_PATH, V_PATH,
    student_paths,
)
from models import (
    build_frame_encoder, build_frame_decoder, VAETrainer,
    build_seq_encoder, build_seq_decoder, SeqVAETrainer,
)
from replay import build_kv, build_replay_sequences, build_iid_replay, kv_retrieve


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ═══════════════════════════════════════════════════════════════════════════
# Teacher VAE
# ═══════════════════════════════════════════════════════════════════════════

def train_or_load_teacher(train_imgs: np.ndarray, force: bool = False):
    if (os.path.exists(TEACHER_ENC_PATH) and
            os.path.exists(TEACHER_DEC_PATH) and not force):
        print("  Loading saved teacher.")
        enc = keras.models.load_model(TEACHER_ENC_PATH, compile=False)
        dec = keras.models.load_model(TEACHER_DEC_PATH, compile=False)
        return enc, dec

    print("  Training teacher VAE...")
    enc = build_frame_encoder(LATENT_DIM)
    dec = build_frame_decoder(LATENT_DIM)
    trainer = VAETrainer(enc, dec, kl_weight=STUDENT_KL_WEIGHT)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3))
    trainer.fit(train_imgs, epochs=TEACHER_EPOCHS,
                batch_size=BATCH_SIZE, verbose=2)
    enc.save(TEACHER_ENC_PATH)
    dec.save(TEACHER_DEC_PATH)
    print("  Teacher saved.")
    return enc, dec


# ═══════════════════════════════════════════════════════════════════════════
# K/V memory
# ═══════════════════════════════════════════════════════════════════════════

def build_or_load_kv(train_imgs, train_seqs, teacher_enc,
                      force: bool = False):
    if os.path.exists(K_PATH) and os.path.exists(V_PATH) and not force:
        print("  Loading saved K/V matrices.")
        return np.load(K_PATH), np.load(V_PATH)

    print("  Building K/V heteroassociative memory...")
    K, V = build_kv(train_imgs, train_seqs, teacher_enc)
    np.save(K_PATH, K)
    np.save(V_PATH, V)
    print(f"  K/V built: {K.shape[0]:,} transitions.")
    return K, V


# ═══════════════════════════════════════════════════════════════════════════
# Sequence dataset helper
# ═══════════════════════════════════════════════════════════════════════════

def _build_seq_dataset(replay_sequences: np.ndarray,
                        shuffle: bool,
                        rng: np.random.Generator) -> tf.data.Dataset:
    """
    replay_sequences: (N_chains, SEQ_LENGTH, 64, 64, 3)
    Returns a tf.data.Dataset yielding (seq,) batches of shape
    (batch, SEQ_LENGTH, 64, 64, 3).

    If shuffle=True, shuffle the FRAME ORDER within each sequence
    (this is the shuffled ablation — same sequences, broken temporal order).
    If shuffle=False, preserve temporal order (sequential student).
    """
    seqs = replay_sequences.copy()
    if shuffle:
        for i in range(len(seqs)):
            perm = rng.permutation(SEQ_LENGTH)
            seqs[i] = seqs[i][perm]

    ds = (tf.data.Dataset
          .from_tensor_slices(seqs)
          .shuffle(buffer_size=2048)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))
    return ds


# ═══════════════════════════════════════════════════════════════════════════
# Sequence VAE training with checkpoints
# ═══════════════════════════════════════════════════════════════════════════

def _train_seq_vae(replay_sequences: np.ndarray,
                    teacher_enc, teacher_dec,
                    enc_path: str, dec_path: str,
                    ckpt_dir: str,
                    seed: int,
                    shuffle_frames: bool,
                    label: str):
    """
    Core training loop for the sequence VAE.

    Saves:
      - final encoder/decoder to enc_path / dec_path
      - intermediate checkpoints at CHECKPOINT_EPOCHS to ckpt_dir
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    seq_enc = build_seq_encoder(LATENT_DIM, SEQ_LATENT_DIM, SEQ_LENGTH)
    seq_dec = build_seq_decoder(LATENT_DIM, SEQ_LATENT_DIM, SEQ_LENGTH)
    trainer = SeqVAETrainer(seq_enc, seq_dec, teacher_enc, teacher_dec,
                             kl_weight=STUDENT_KL_WEIGHT)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3))

    os.makedirs(ckpt_dir, exist_ok=True)

    # Train epoch by epoch so we can save checkpoints
    epochs_done = 0
    for target_epoch in sorted(set(CHECKPOINT_EPOCHS)):
        epochs_to_run = target_epoch - epochs_done
        if epochs_to_run <= 0:
            continue

        ds = _build_seq_dataset(replay_sequences, shuffle_frames, rng)
        trainer.fit(ds, epochs=epochs_to_run, verbose=2)
        epochs_done = target_epoch

        # Save checkpoint
        ckpt_enc = os.path.join(ckpt_dir, f"enc_epoch{target_epoch}.keras")
        ckpt_dec = os.path.join(ckpt_dir, f"dec_epoch{target_epoch}.keras")
        seq_enc.save(ckpt_enc)
        seq_dec.save(ckpt_dec)
        print(f"  [{label}] checkpoint saved at epoch {target_epoch}.")

    # Save final models
    seq_enc.save(enc_path)
    seq_dec.save(dec_path)
    print(f"  [{label}] final model saved (seed={seed}).")
    return seq_enc, seq_dec


def train_or_load_seq_vae(replay_sequences: np.ndarray,
                           teacher_enc, teacher_dec,
                           seed: int,
                           shuffle_frames: bool = False,
                           force: bool = False):
    """
    Train or load the sequence VAE student.

    shuffle_frames=False → sequential student (ordered replay)
    shuffle_frames=True  → shuffled ablation (same sequences, broken order)
    """
    paths   = student_paths(seed)
    label   = "shuffled" if shuffle_frames else "sequential"
    enc_key = "shuf_vae_enc" if shuffle_frames else "seq_vae_enc"
    dec_key = "shuf_vae_dec" if shuffle_frames else "seq_vae_dec"
    enc_path  = paths[enc_key]
    dec_path  = paths[dec_key]
    ckpt_dir  = paths["seq_ckpt_dir"] if not shuffle_frames else \
                paths["seq_ckpt_dir"].replace("seq_vae", "shuf_vae")

    if os.path.exists(enc_path) and os.path.exists(dec_path) and not force:
        print(f"  Loading saved {label} seq-VAE (seed={seed}).")
        enc = keras.models.load_model(enc_path, compile=False)
        dec = keras.models.load_model(dec_path, compile=False)
        return enc, dec

    print(f"  Training {label} seq-VAE (seed={seed})...")
    return _train_seq_vae(
        replay_sequences, teacher_enc, teacher_dec,
        enc_path, dec_path, ckpt_dir, seed, shuffle_frames, label)


# ═══════════════════════════════════════════════════════════════════════════
# IID frame VAE (original Spens & Burgess baseline)
# ═══════════════════════════════════════════════════════════════════════════

def train_or_load_iid_vae(train_imgs, teacher_enc, teacher_dec,
                           seed: int, force: bool = False):
    """
    Train original Spens & Burgess frame-level IID VAE on MHN-replayed frames.
    """
    paths    = student_paths(seed)
    enc_path = paths["iid_enc"]
    dec_path = paths["iid_dec"]

    if os.path.exists(enc_path) and os.path.exists(dec_path) and not force:
        print(f"  Loading saved IID frame-VAE (seed={seed}).")
        enc = keras.models.load_model(enc_path, compile=False)
        dec = keras.models.load_model(dec_path, compile=False)
        return enc, dec

    set_seed(seed)
    print(f"  Generating MHN replay for IID student (seed={seed})...")
    replay_imgs = build_iid_replay(train_imgs, teacher_enc, teacher_dec)

    enc = build_frame_encoder(LATENT_DIM)
    dec = build_frame_decoder(LATENT_DIM)
    trainer = VAETrainer(enc, dec, kl_weight=STUDENT_KL_WEIGHT)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3))
    trainer.fit(replay_imgs, epochs=STUDENT_EPOCHS,
                batch_size=BATCH_SIZE, verbose=2)
    enc.save(enc_path)
    dec.save(dec_path)
    print(f"  IID frame-VAE saved (seed={seed}).")
    return enc, dec


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_training_pipeline(train_imgs, train_seqs, seed: int,
                           force_teacher: bool = False,
                           force_students: bool = False):
    print(f"\n{'='*60}")
    print(f"Training pipeline  [seed={seed}]")
    print(f"{'='*60}")

    # 1. Teacher
    print("\n[1/5] Teacher VAE")
    teacher_enc, teacher_dec = train_or_load_teacher(
        train_imgs, force=force_teacher)

    # 2. K/V memory
    print("\n[2/5] K/V heteroassociative memory")
    K, V = build_or_load_kv(train_imgs, train_seqs, teacher_enc,
                              force=force_teacher)

    # 3. Generate replay sequences: (N_chains, SEQ_LENGTH, 64, 64, 3)
    print(f"\n[3/5] Sequential replay chains (seed={seed})")
    set_seed(seed)
    replay_sequences = build_sequential_replay_sequences(
        train_imgs, K, V, teacher_enc, teacher_dec)
    print(f"  Generated {len(replay_sequences)} replay sequences "
          f"of length {SEQ_LENGTH}.")

    # 4. Sequential seq-VAE
    print(f"\n[4/5] Sequential seq-VAE (seed={seed})")
    seq_enc, seq_dec = train_or_load_seq_vae(
        replay_sequences, teacher_enc, teacher_dec,
        seed=seed, shuffle_frames=False, force=force_students)

    # 5. Shuffled seq-VAE ablation
    print(f"\n[5/5] Shuffled seq-VAE ablation (seed={seed})")
    shuf_enc, shuf_dec = train_or_load_seq_vae(
        replay_sequences, teacher_enc, teacher_dec,
        seed=seed, shuffle_frames=True, force=force_students)

    # 6. IID frame-VAE (Spens & Burgess baseline)
    print(f"\n[6/5] IID frame-VAE baseline (seed={seed})")
    iid_enc, iid_dec = train_or_load_iid_vae(
        train_imgs, teacher_enc, teacher_dec,
        seed=seed, force=force_students)

    return {
        "teacher_enc": teacher_enc, "teacher_dec": teacher_dec,
        "K": K, "V": V,
        "replay_sequences": replay_sequences,
        # Sequential seq-VAE (primary condition)
        "seq_enc": seq_enc, "seq_dec": seq_dec,
        # Shuffled seq-VAE (ablation — same arch, broken order)
        "shuf_enc": shuf_enc, "shuf_dec": shuf_dec,
        # IID frame-VAE (Spens & Burgess baseline)
        "iid_enc": iid_enc, "iid_dec": iid_dec,
        # Paths for loading checkpoints in Exp 2
        "seq_ckpt_dir":  student_paths(seed)["seq_ckpt_dir"],
        "shuf_ckpt_dir": student_paths(seed)["seq_ckpt_dir"].replace(
            "seq_vae", "shuf_vae"),
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build full sequences from K/V chains
# ═══════════════════════════════════════════════════════════════════════════

def build_sequential_replay_sequences(train_imgs, K, V,
                                       teacher_enc, teacher_dec):
    """
    Build replay sequences as full (N_chains, SEQ_LENGTH, 64, 64, 3) arrays
    rather than flat (frames_t, frames_t1) pairs.

    Each chain is a sequence of SEQ_LENGTH decoded frames generated by
    iterative K/V retrieval starting from a random training sequence.
    """
    from config import N_CHAINS, CHAIN_LENGTH, BETA_KV, TOPK
    from replay import kv_retrieve, build_replay_sequences
    return build_replay_sequences(train_imgs, K, V, teacher_enc, teacher_dec)
from tensorflow import keras

from config import (
    LATENT_DIM, BATCH_SIZE, TEACHER_EPOCHS, STUDENT_EPOCHS,
    TEACHER_ENC_PATH, TEACHER_DEC_PATH, K_PATH, V_PATH,
    student_paths,
)
from models import build_encoder, build_decoder, build_transition, VAETrainer, SeqTrainer
from replay import (build_iid_replay_matched, build_iid_replay,
                    build_kv, build_sequential_replay)


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ── Teacher ───────────────────────────────────────────────────────────────────

def train_or_load_teacher(train_imgs: np.ndarray, force: bool = False):
    if os.path.exists(TEACHER_ENC_PATH) and os.path.exists(TEACHER_DEC_PATH) and not force:
        print("  Loading saved teacher.")
        enc = keras.models.load_model(TEACHER_ENC_PATH, compile=False)
        dec = keras.models.load_model(TEACHER_DEC_PATH, compile=False)
        return enc, dec

    print("  Training teacher VAE...")
    enc = build_encoder(LATENT_DIM)
    dec = build_decoder(LATENT_DIM)
    trainer = VAETrainer(enc, dec)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3))
    trainer.fit(train_imgs, epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    enc.save(TEACHER_ENC_PATH)
    dec.save(TEACHER_DEC_PATH)
    return enc, dec


# ── K/V memory ────────────────────────────────────────────────────────────────

def build_or_load_kv(train_imgs, train_sequences, teacher_enc, force: bool = False):
    if os.path.exists(K_PATH) and os.path.exists(V_PATH) and not force:
        print("  Loading saved K/V matrices.")
        return np.load(K_PATH), np.load(V_PATH)

    print("  Building K/V heteroassociative memory...")
    K, V = build_kv(train_imgs, train_sequences, teacher_enc)
    np.save(K_PATH, K)
    np.save(V_PATH, V)
    print(f"  K/V built: {K.shape[0]} transitions.")
    return K, V


# ── IID-matched student (primary fair comparison) ─────────────────────────────

def train_or_load_iid_student(replay_imgs: np.ndarray, seed: int,
                               force: bool = False):
    """
    Train IID student on matched replay (same decoded frames as sequential,
    pooled and shuffled).  Reconstruction (VAE) objective.
    """
    paths  = student_paths(seed)
    ep, dp = paths["iid_enc"], paths["iid_dec"]

    if os.path.exists(ep) and os.path.exists(dp) and not force:
        print(f"  Loading saved IID-matched student (seed={seed}).")
        return keras.models.load_model(ep, compile=False), keras.models.load_model(dp, compile=False)

    set_seed(seed)
    enc = build_encoder(LATENT_DIM)
    dec = build_decoder(LATENT_DIM)
    trainer = VAETrainer(enc, dec)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3))
    trainer.fit(replay_imgs, epochs=STUDENT_EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    enc.save(ep); dec.save(dp)
    print(f"  IID-matched student saved (seed={seed}).")
    return enc, dec


# ── IID-MHN student (Spens & Burgess reference) ───────────────────────────────

def train_or_load_iid_mhn_student(train_imgs, teacher_enc, teacher_dec,
                                   seed: int, force: bool = False):
    """
    Train IID student on MHN-retrieved frames.  Replicates Spens & Burgess
    exactly.  Kept as a secondary reference condition — note that this student
    sees different frames (MHN retrievals) than the sequential student.
    """
    paths  = student_paths(seed)
    ep, dp = paths["iid_mhn_enc"], paths["iid_mhn_dec"]

    if os.path.exists(ep) and os.path.exists(dp) and not force:
        print(f"  Loading saved IID-MHN student (seed={seed}).")
        return keras.models.load_model(ep, compile=False), keras.models.load_model(dp, compile=False)

    set_seed(seed)
    print(f"  Generating MHN replay (seed={seed})...")
    replay_imgs = build_iid_replay(train_imgs, teacher_enc, teacher_dec)

    enc = build_encoder(LATENT_DIM)
    dec = build_decoder(LATENT_DIM)
    trainer = VAETrainer(enc, dec)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3))
    trainer.fit(replay_imgs, epochs=STUDENT_EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    enc.save(ep); dec.save(dp)
    print(f"  IID-MHN student saved (seed={seed}).")
    return enc, dec


# ── Sequential student (ours) ────────────────────────────────────────────────

def train_or_load_seq_student(frames_t: np.ndarray, frames_t1: np.ndarray,
                               seed: int, force: bool = False):
    """
    Train sequential student on ordered (frame_t, frame_{t+1}) pairs.
    Prediction objective.
    """
    paths      = student_paths(seed)
    ep, dp, tp = paths["seq_enc"], paths["seq_dec"], paths["seq_trans"]

    if os.path.exists(ep) and os.path.exists(dp) and os.path.exists(tp) and not force:
        print(f"  Loading saved sequential student (seed={seed}).")
        return (keras.models.load_model(ep, compile=False),
                keras.models.load_model(dp, compile=False),
                keras.models.load_model(tp, compile=False))

    set_seed(seed)
    enc   = build_encoder(LATENT_DIM)
    dec   = build_decoder(LATENT_DIM)
    trans = build_transition(LATENT_DIM)
    trainer = SeqTrainer(enc, dec, trans)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3))
    ds = tf.data.Dataset.from_tensor_slices((frames_t, frames_t1)).shuffle(8192).batch(BATCH_SIZE)
    trainer.fit(ds, epochs=STUDENT_EPOCHS, verbose=2)
    enc.save(ep); dec.save(dp); trans.save(tp)
    print(f"  Sequential student saved (seed={seed}).")
    return enc, dec, trans


# ── Full pipeline for one seed ────────────────────────────────────────────────

def run_training_pipeline(train_imgs, train_sequences, seed: int,
                           force_teacher=False, force_students=False):
    print(f"\n{'='*60}")
    print(f"Training pipeline  [seed={seed}]")
    print(f"{'='*60}")

    print("\n[1/5] Teacher VAE")
    teacher_enc, teacher_dec = train_or_load_teacher(train_imgs, force=force_teacher)

    print("\n[2/5] K/V heteroassociative memory")
    K, V = build_or_load_kv(train_imgs, train_sequences, teacher_enc,
                             force=force_teacher)

    # Generate sequential replay once — reused by both IID-matched and seq students.
    print(f"\n[3/5] Sequential replay chains (seed={seed})")
    set_seed(seed)
    frames_t, frames_t1 = build_sequential_replay(
        train_imgs, K, V, teacher_enc, teacher_dec)
    print(f"  Generated {len(frames_t)} consecutive pairs.")

    # IID-matched: pool and shuffle the exact same decoded frames.
    iid_matched_imgs = build_iid_replay_matched(frames_t, frames_t1)
    print(f"  IID-matched corpus: {len(iid_matched_imgs)} frames "
          f"(same pixels as sequential, shuffled).")

    print(f"\n[4/5] IID students (seed={seed})")
    iid_enc, iid_dec = train_or_load_iid_student(
        iid_matched_imgs, seed=seed, force=force_students)

    iid_mhn_enc, iid_mhn_dec = train_or_load_iid_mhn_student(
        train_imgs, teacher_enc, teacher_dec, seed=seed, force=force_students)

    print(f"\n[5/5] Sequential student (seed={seed})")
    seq_enc, seq_dec, seq_trans = train_or_load_seq_student(
        frames_t, frames_t1, seed=seed, force=force_students)

    return {
        "teacher_enc": teacher_enc, "teacher_dec": teacher_dec,
        "K": K, "V": V,
        "iid_enc":     iid_enc,     "iid_dec":     iid_dec,
        "iid_mhn_enc": iid_mhn_enc, "iid_mhn_dec": iid_mhn_dec,
        "seq_enc":     seq_enc,     "seq_dec":     seq_dec,
        "seq_trans":   seq_trans,
    }
