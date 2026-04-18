"""
Replay generation for both conditions.

IID (Spens & Burgess baseline)
────────────────────────────────────────────────────────────────────────────────
  1. Encode all training frames → latent means Z  (N × latent_dim)
  2. Store Z in a latent-space ContinuousHopfield network (as per Spens & Burgess
     but operating in latent space to keep it tractable for 64×64 images)
  3. Query with Z + Gaussian noise → retrieve clean latents
  4. Decode retrieved latents → IID replay images
  5. Student trains on these images (reconstruction objective, no temporal order)

Sequential (ours)
────────────────────────────────────────────────────────────────────────────────
  1. Encode training frames → z_means
  2. Build K/V heteroassociative Hopfield: K[i] = z_t, V[i] = z_{t+1}
     for every consecutive pair in every training sequence
  3. Generate chains: start from random training frame, repeatedly query
     K/V Hopfield to advance one step
  4. Decode chain → consecutive (frame_t, frame_{t+1}) pairs
  5. Student trains on these ordered pairs (prediction objective)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generative-memory-main"))

import numpy as np
from scipy.special import softmax as scipy_softmax

from config import (
    BETA_MHN, MHN_NOISE_STD, N_IID_REPLAY,
    BETA_KV, TOPK, N_CHAINS, CHAIN_LENGTH,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _l2_norm(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


# ── IID replay: latent-space ContinuousHopfield ───────────────────────────────

class LatentMHN:
    """
    ContinuousHopfield operating in latent space.

    Implements the same update rule as Spens & Burgess
    (generative-memory-main/hopfield_models.py ContinuousHopfield)
    but on latent vectors instead of raw pixels.

    Update rule (one step):
        state_new = patterns @ softmax(beta * patterns^T @ state / sqrt(d))
    """
    def __init__(self, beta: float = BETA_MHN):
        self.beta = beta
        self.patterns = None   # (d, M)  column-wise
        self.d = None

    def fit(self, Z: np.ndarray):
        """Z : (M, d) latent means from teacher encoder."""
        self.patterns = Z.T.astype("float32")   # (d, M)
        self.d = Z.shape[1]

    def retrieve(self, query: np.ndarray, max_iter: int = 20) -> np.ndarray:
        """
        query : (d,) or (B, d)
        Returns retrieved pattern(s) of same shape.
        """
        batch = query.ndim == 2
        if not batch:
            query = query[np.newaxis]            # (1, d)

        state = query.copy().astype("float32")   # (B, d)
        sqrt_d = np.sqrt(self.d)

        for _ in range(max_iter):
            scores = (state @ self.patterns) / sqrt_d         # (B, M)
            # numerically stable softmax over memories
            scores -= scores.max(axis=1, keepdims=True)
            w = np.exp(self.beta * scores)
            w /= w.sum(axis=1, keepdims=True)                 # (B, M)
            state_new = w @ self.patterns.T                   # (B, d)
            if np.abs(state_new - state).max() < 1e-4:
                break
            state = state_new

        return state if batch else state[0]


def build_iid_replay_matched(frames_t: np.ndarray,
                             frames_t1: np.ndarray) -> np.ndarray:
    """
    IID replay matched to sequential: exactly the same decoded frames that
    the sequential student sees, pooled together and randomly shuffled.

    This is the primary fair comparison — both students see identical pixels;
    only the ordering and the training objective differ.

    frames_t, frames_t1 : (N, 64, 64, 3) consecutive-pair arrays from
                          build_sequential_replay()
    Returns               (2N, 64, 64, 3) shuffled frame array.
    """
    all_frames = np.concatenate([frames_t, frames_t1], axis=0)
    idx        = np.random.permutation(len(all_frames))
    return all_frames[idx].astype("float32")


def build_iid_replay(train_imgs: np.ndarray, teacher_enc, teacher_dec,
                     n_samples: int = N_IID_REPLAY) -> np.ndarray:
    """
    Encode training frames, store in latent MHN, query with noisy latents,
    decode retrieved latents → return IID replay image array (n_samples, 64, 64, 3).
    """
    print("    Encoding training frames for MHN...")
    z_means = teacher_enc.predict(train_imgs, batch_size=256, verbose=0)[0]  # (N, d)

    print("    Fitting latent MHN...")
    mhn = LatentMHN(beta=BETA_MHN)
    mhn.fit(z_means)

    print(f"    Generating {n_samples} IID replay frames via MHN retrieval...")
    idxs   = np.random.choice(len(z_means), size=n_samples, replace=True)
    queries = z_means[idxs] + np.random.normal(0, MHN_NOISE_STD, (n_samples, z_means.shape[1]))
    retrieved = mhn.retrieve(queries.astype("float32"), max_iter=20)  # (n_samples, d)

    replay_imgs = teacher_dec.predict(retrieved, batch_size=256, verbose=0)  # (n_samples, 64, 64, 3)
    return replay_imgs.astype("float32")


# ── Sequential replay: K/V heteroassociative Hopfield ────────────────────────

def build_kv(train_imgs: np.ndarray, train_sequences: list, teacher_enc):
    """
    Encode all train frames, then build K/V matrices from consecutive pairs
    within each sequence.

    Returns K (M, d), V (M, d)  where K[i] = z_t, V[i] = z_{t+1}.
    """
    z_means = teacher_enc.predict(train_imgs, batch_size=256, verbose=0)[0]  # (N, d)
    keys, vals = [], []
    for seq in train_sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            keys.append(z_means[a])
            vals.append(z_means[b])
    K = np.stack(keys).astype("float32")
    V = np.stack(vals).astype("float32")
    return K, V


def _hopfield_step(z_query: np.ndarray, K_norm: np.ndarray, V: np.ndarray,
                   beta: float = BETA_KV, topk: int = TOPK) -> np.ndarray:
    """One heteroassociative retrieval step: z_t → z_{t+1}."""
    zq = z_query / (np.linalg.norm(z_query) + 1e-8)
    scores = K_norm @ zq                                    # (M,)
    if topk < len(scores):
        idx = np.argpartition(scores, -topk)[-topk:]
    else:
        idx = np.arange(len(scores))
    sub = beta * scores[idx]
    sub -= sub.max()
    w = np.exp(sub); w /= w.sum()
    return (w @ V[idx]).astype("float32")


def build_sequential_replay(train_imgs: np.ndarray, K: np.ndarray, V: np.ndarray,
                             teacher_enc, teacher_dec,
                             n_chains: int = N_CHAINS,
                             chain_length: int = CHAIN_LENGTH):
    """
    Generate sequential (frame_t, frame_{t+1}) training pairs from Hopfield chains.
    Returns frames_t, frames_t1 each (n_pairs, 64, 64, 3).
    """
    K_norm = _l2_norm(K)

    latents_t, latents_t1 = [], []
    start_idxs = np.random.choice(len(train_imgs), size=n_chains, replace=True)

    print(f"    Generating {n_chains} chains of length {chain_length}...")
    for si in start_idxs:
        z = teacher_enc.predict(train_imgs[si:si+1], verbose=0)[0][0]
        chain = [z]
        for _ in range(chain_length - 1):
            z = _hopfield_step(z, K_norm, V)
            chain.append(z)
        for t in range(len(chain) - 1):
            latents_t.append(chain[t])
            latents_t1.append(chain[t + 1])

    lt  = np.array(latents_t,  dtype="float32")
    lt1 = np.array(latents_t1, dtype="float32")

    print(f"    Decoding {len(lt)} sequential pairs...")
    frames_t  = teacher_dec.predict(lt,  batch_size=256, verbose=0).astype("float32")
    frames_t1 = teacher_dec.predict(lt1, batch_size=256, verbose=0).astype("float32")
    return frames_t, frames_t1
