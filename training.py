"""
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
# IID sequence VAE (Spens & Burgess baseline — same architecture as others)
# ═══════════════════════════════════════════════════════════════════════════

def _build_iid_seq_corpus(train_imgs: np.ndarray,
                           teacher_enc, teacher_dec,
                           n_sequences: int,
                           rng: np.random.Generator) -> np.ndarray:
    """
    Build IID replay corpus as (N, SEQ_LENGTH, 64, 64, 3) sequences.

    Each "sequence" is constructed by sampling SEQ_LENGTH frames
    independently at random from the MHN (with noise), then assembling
    them into a sequence array.  There is no temporal relationship
    between frames within a sequence — this is itemwise replay
    repackaged into episode-shaped arrays so all three students can
    use the identical sequence VAE architecture and objective.

    This mirrors the Spens & Burgess replay mechanism (random noise →
    MHN → retrieved frame) but produces sequences of the same shape
    as the other students' replay data.
    """
    from replay import LatentMHN

    print("    [IID] Encoding training frames for MHN...")
    z_means = teacher_enc.predict(
        train_imgs, batch_size=256, verbose=0)[0]

    print("    [IID] Fitting latent MHN...")
    mhn = LatentMHN()
    mhn.fit(z_means)

    total_frames = n_sequences * SEQ_LENGTH
    print(f"    [IID] Retrieving {total_frames} independent frames...")
    idxs    = rng.choice(len(z_means), size=total_frames, replace=True)
    queries = (z_means[idxs] +
               rng.standard_normal((total_frames, z_means.shape[1])) * 0.3)
    retrieved = mhn.retrieve(queries.astype("float32"))
    frames    = teacher_dec.predict(
        retrieved, batch_size=256, verbose=0).astype("float32")

    # Reshape into (N, SEQ_LENGTH, 64, 64, 3) — no temporal ordering
    sequences = frames.reshape(n_sequences, SEQ_LENGTH, 64, 64, 3)
    return sequences


def train_or_load_iid_seq_vae(replay_sequences: np.ndarray,
                               teacher_enc, teacher_dec,
                               seed: int,
                               force: bool = False):
    """
    Train the IID sequence VAE student.

    Same architecture and objective as the sequential and shuffled students.
    Replay corpus: SEQ_LENGTH independently MHN-retrieved frames assembled
    into sequences — no temporal structure between frames.

    This is the fairest possible Spens & Burgess baseline: the only
    difference from the sequential student is that its replay sequences
    contain no temporal ordering information.
    """
    paths    = student_paths(seed)
    enc_path = paths["iid_enc"]
    dec_path = paths["iid_dec"]
    ckpt_dir = paths["seq_ckpt_dir"].replace("seq_vae", "iid_vae")

    if os.path.exists(enc_path) and os.path.exists(dec_path) and not force:
        print(f"  Loading saved IID seq-VAE (seed={seed}).")
        enc = keras.models.load_model(enc_path, compile=False)
        dec = keras.models.load_model(dec_path, compile=False)
        return enc, dec

    set_seed(seed)
    rng = np.random.default_rng(seed)

    print(f"  Building IID replay corpus from shared replay frames (seed={seed})...")
    iid_sequences = _build_iid_seq_corpus_from_replay(replay_sequences, rng)
    print(f"  Generated {len(iid_sequences)} IID replay sequences.")

    print(f"  Training IID seq-VAE (seed={seed})...")
    return _train_seq_vae(
        iid_sequences, teacher_enc, teacher_dec,
        enc_path, dec_path, ckpt_dir,
        seed=seed, shuffle_frames=False, label="iid")


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

    # 6. IID seq-VAE (Spens & Burgess baseline — same architecture)
    print(f"\n[6/6] IID seq-VAE baseline (seed={seed})")
    iid_enc, iid_dec = train_or_load_iid_seq_vae(
        replay_sequences, teacher_enc, teacher_dec,
        seed=seed, force=force_students)

    return {
        "teacher_enc": teacher_enc, "teacher_dec": teacher_dec,
        "K": K, "V": V,
        "replay_sequences": replay_sequences,
        # Sequential seq-VAE (ordered replay)
        "seq_enc": seq_enc, "seq_dec": seq_dec,
        # Shuffled seq-VAE (same pixels, random order)
        "shuf_enc": shuf_enc, "shuf_dec": shuf_dec,
        # IID seq-VAE (independent MHN frames, no temporal structure)
        "iid_enc": iid_enc, "iid_dec": iid_dec,
        # Checkpoint dirs for Exp 2
        "seq_ckpt_dir":  student_paths(seed)["seq_ckpt_dir"],
        "shuf_ckpt_dir": student_paths(seed)["seq_ckpt_dir"].replace(
            "seq_vae", "shuf_vae"),
        "iid_ckpt_dir":  student_paths(seed)["seq_ckpt_dir"].replace(
            "seq_vae", "iid_vae"),
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


def _build_iid_seq_corpus_from_replay(
        replay_sequences: np.ndarray,
        rng: np.random.Generator) -> np.ndarray:
    """
    Build IID replay corpus from the same replay frames used by the
    sequential and shuffled students.

    Input:
        replay_sequences: (N, SEQ_LENGTH, 64, 64, 3)

    Output:
        iid_sequences: (N, SEQ_LENGTH, 64, 64, 3)

    Each IID sequence is made by sampling frames independently from the
    flattened replay corpus. This removes episode/order structure while
    keeping the same replay frame distribution.
    """
    n_sequences = replay_sequences.shape[0]

    flat_frames = replay_sequences.reshape(
        -1,
        replay_sequences.shape[2],
        replay_sequences.shape[3],
        replay_sequences.shape[4],
    )

    total_frames = n_sequences * SEQ_LENGTH
    idxs = rng.choice(len(flat_frames), size=total_frames, replace=True)

    iid_sequences = flat_frames[idxs].reshape(
        n_sequences,
        SEQ_LENGTH,
        replay_sequences.shape[2],
        replay_sequences.shape[3],
        replay_sequences.shape[4],
    ).astype("float32")

    return iid_sequences