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
import tensorflow as tf
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
