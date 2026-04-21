"""
replay.py
─────────────────────────────────────────────────────────────────────────────
Replay generation.

IID replay (Spens & Burgess baseline)
  Encodes training frames, stores in a latent-space ContinuousHopfield
  network, queries with noisy versions, decodes retrieved latents.
  Student trains on these frames with a reconstruction objective.

Sequential replay (ours)
  Builds a K/V heteroassociative Hopfield memory storing (z_t, z_{t+1})
  pairs from all consecutive frames in training sequences.
  Generates chains of length SEQ_LENGTH by iterative K/V retrieval.
  Returns full sequences of shape (N_chains, SEQ_LENGTH, 64, 64, 3) for
  training the sequence VAE.
"""

import numpy as np
from config import (
    BETA_MHN, MHN_NOISE_STD, N_IID_REPLAY,
    BETA_KV, TOPK, N_CHAINS, CHAIN_LENGTH, SEQ_LENGTH,
)


# ═══════════════════════════════════════════════════════════════════════════
# IID replay: latent-space ContinuousHopfield
# ═══════════════════════════════════════════════════════════════════════════

class LatentMHN:
    """ContinuousHopfield operating in teacher latent space."""

    def __init__(self, beta: float = BETA_MHN):
        self.beta     = beta
        self.patterns = None

    def fit(self, Z: np.ndarray):
        self.patterns = Z.T.astype("float32")   # (d, M)
        self.d        = Z.shape[1]

    def retrieve(self, query: np.ndarray, max_iter: int = 20) -> np.ndarray:
        batch = query.ndim == 2
        if not batch:
            query = query[np.newaxis]
        state  = query.copy().astype("float32")
        sqrt_d = np.sqrt(self.d)
        for _ in range(max_iter):
            scores   = (state @ self.patterns) / sqrt_d
            scores  -= scores.max(axis=1, keepdims=True)
            w        = np.exp(self.beta * scores)
            w       /= w.sum(axis=1, keepdims=True)
            state_new = w @ self.patterns.T
            if np.abs(state_new - state).max() < 1e-4:
                break
            state = state_new
        return state if batch else state[0]


def build_iid_replay(train_imgs: np.ndarray, teacher_enc, teacher_dec,
                      n_samples: int = N_IID_REPLAY) -> np.ndarray:
    """Generate IID replay frames via latent MHN retrieval."""
    print("    Encoding training frames for MHN...")
    z_means = teacher_enc.predict(
        train_imgs, batch_size=256, verbose=0)[0]

    print("    Fitting latent MHN...")
    mhn = LatentMHN(beta=BETA_MHN)
    mhn.fit(z_means)

    print(f"    Generating {n_samples} IID replay frames...")
    idxs    = np.random.choice(len(z_means), size=n_samples, replace=True)
    queries = (z_means[idxs] +
               np.random.normal(0, MHN_NOISE_STD,
                                (n_samples, z_means.shape[1])))
    retrieved   = mhn.retrieve(queries.astype("float32"))
    replay_imgs = teacher_dec.predict(
        retrieved, batch_size=256, verbose=0)
    return replay_imgs.astype("float32")


# ═══════════════════════════════════════════════════════════════════════════
# K/V heteroassociative Hopfield
# ═══════════════════════════════════════════════════════════════════════════

def build_kv(train_imgs: np.ndarray, train_sequences: list,
              teacher_enc):
    """
    Build K/V matrices from consecutive frame pairs in training sequences.
    K[i] = z_t,  V[i] = z_{t+1}  for every consecutive pair.
    """
    z_means = teacher_enc.predict(
        train_imgs, batch_size=256, verbose=0)[0]
    keys, vals = [], []
    for seq in train_sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            keys.append(z_means[a])
            vals.append(z_means[b])
    K = np.stack(keys).astype("float32")
    V = np.stack(vals).astype("float32")
    return K, V


def _l2_norm(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def kv_retrieve(z_query: np.ndarray, K: np.ndarray, V: np.ndarray,
                beta: float = BETA_KV, topk: int = TOPK) -> np.ndarray:
    """
    One K/V retrieval step: z_t → z_{t+1}.
    z_query: (1, latent_dim) or (latent_dim,)
    Returns: (1, latent_dim)
    """
    squeeze = z_query.ndim == 1
    if squeeze:
        z_query = z_query[np.newaxis]

    K_norm = _l2_norm(K)
    zq     = z_query[0] / (np.linalg.norm(z_query[0]) + 1e-8)
    scores = K_norm @ zq                         # (M,)

    if topk < len(scores):
        idx = np.argpartition(scores, -topk)[-topk:]
    else:
        idx = np.arange(len(scores))

    sub  = beta * scores[idx]
    sub -= sub.max()
    w    = np.exp(sub)
    w   /= w.sum()
    z_next = (w @ V[idx]).astype("float32")[np.newaxis]   # (1, latent_dim)
    return z_next


def build_replay_sequences(train_imgs: np.ndarray,
                            K: np.ndarray, V: np.ndarray,
                            teacher_enc, teacher_dec,
                            n_chains: int = N_CHAINS,
                            chain_length: int = CHAIN_LENGTH
                            ) -> np.ndarray:
    """
    Generate full replay sequences for training the sequence VAE.

    Each chain starts from a random training frame and advances
    chain_length steps via K/V retrieval. All latents in the chain
    are decoded in one batch, giving a sequence of frames.

    Returns:
        sequences: (n_chains, chain_length, 64, 64, 3)  float32
    """
    print(f"    Encoding all training frames...")
    all_z = teacher_enc.predict(
        train_imgs, batch_size=256, verbose=0)[0]   # (N, latent_dim)

    rng        = np.random.default_rng(0)
    start_idxs = rng.integers(0, len(train_imgs), size=n_chains)
    sequences  = []

    print(f"    Generating {n_chains} chains of length {chain_length}...")
    for si in start_idxs:
        z          = all_z[si:si + 1]           # (1, latent_dim)
        chain_z    = [z[0]]
        for _ in range(chain_length - 1):
            z = kv_retrieve(z, K, V)
            chain_z.append(z[0])

        chain_z_arr = np.stack(chain_z)          # (chain_length, latent_dim)
        chain_frames = teacher_dec.predict(
            chain_z_arr, batch_size=chain_length, verbose=0)
        sequences.append(chain_frames.astype("float32"))

    return np.stack(sequences)   # (n_chains, chain_length, 64, 64, 3)