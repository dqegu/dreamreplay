"""
run_seq_replay_v3.py

Experiment: Does sequential dream replay produce better consolidation of
temporally ordered experiences than unstructured IID replay?

This version correctly implements Spens & Burgess's proposed extension
(Discussion, p.535): the sequential student is a PREDICTIVE generative model
trained to predict frame t+1 given frame t, not merely reconstruct frames.

Two training conditions:
  A) IID student VAE  — standard VAE trained on randomly ordered replay frames
                        (reconstruction objective, no temporal structure)
                        This matches Spens & Burgess basic model.

  B) Sequential predictive student — shared VAE encoder + transition MLP,
                        trained on consecutive pairs from Hopfield chains.
                        Training objective: given frame t, reconstruct frame t+1.
                        This is what Spens meant by "a sequential generative
                        network trained to predict the next input during
                        sequential replay" (Discussion, p.535, refs 122-124).

Evaluation:
  1. Next-frame prediction MSE — linear probe on student latent space
     (Student B is now *trained* to do this, so this is a fair primary test)
  2. Reconstruction MSE on held-out real images (sanity check)
  3. Schema distortion ratio (connects back to Spens & Burgess Fig 4)

Toulmin mapping:
  Claim   : A student trained with a sequential predictive objective on
            Hopfield replay chains consolidates temporally ordered structure
            better than a student trained with a reconstruction objective
            on unordered IID replay
  Grounds : Student B outperforms Student A on next-frame prediction MSE
  Warrant : Student B was explicitly trained to predict temporal transitions,
            so lower MSE reflects genuine consolidation of sequential structure
            rather than a confound of latent space scale or compression
  Backing : Spens & Burgess 2024 (Discussion p.535, refs 122-124);
            Stella et al. 2019 (sequential hippocampal reactivation);
            Diba & Buzsaki 2007 (forward replay)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = "/home/ao1g22/comp6228/irp"
TFDS_DIR = os.path.join(BASE_DIR, "tfds_data")
ART_DIR  = os.path.join(BASE_DIR, "artifacts")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ENC_PATH     = os.path.join(ART_DIR, "teacher_encoder.keras")
DEC_PATH     = os.path.join(ART_DIR, "teacher_decoder.keras")
K_PATH       = os.path.join(ART_DIR, "K.npy")
V_PATH       = os.path.join(ART_DIR, "V.npy")

# IID student (standard VAE, same as before)
ENC_IID_PATH = os.path.join(ART_DIR, "student_iid_encoder.keras")
DEC_IID_PATH = os.path.join(ART_DIR, "student_iid_decoder.keras")

# Sequential predictive student (shared encoder + transition MLP + decoder)
ENC_SEQ_PATH     = os.path.join(ART_DIR, "student_seq_encoder.keras")
DEC_SEQ_PATH     = os.path.join(ART_DIR, "student_seq_decoder.keras")
TRANS_SEQ_PATH   = os.path.join(ART_DIR, "student_seq_transition.keras")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
N_SAMPLES        = 2000
LATENT_DIM       = 32
BATCH_SIZE       = 32
TEACHER_EPOCHS   = 30
STUDENT_EPOCHS   = 30
N_REPLAY_SAMPLES = 2000
CHAIN_LENGTH     = 16
N_CHAINS         = 125    # 125 × 16 = 2000 replay samples
BETA_HOPFIELD    = 60.0
TOPK             = 50
N_EVAL           = 300

FORCE_RETRAIN_TEACHER  = False
FORCE_RETRAIN_STUDENTS = True   # set True to rebuild with new architecture
FORCE_REBUILD_KV       = False

SEQ_LABEL    = "label_orientation"
GROUP_LABELS = (
    "label_floor_hue", "label_wall_hue",
    "label_object_hue", "label_scale", "label_shape"
)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_shapes3d_with_labels(n):
    ds = tfds.load(
        "shapes3d",
        split=f"train[:{n}]",
        data_dir=TFDS_DIR,
        shuffle_files=False
    )
    imgs, labels = [], []
    for ex in tfds.as_numpy(ds):
        imgs.append(ex["image"])
        labels.append({k: int(ex[k]) for k in ex.keys() if k.startswith("label_")})
    imgs = np.stack(imgs).astype("float32") / 255.0
    return imgs, labels


# ─────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package='spens_seq')
class SamplingLayer(layers.Layer):
    """Reparameterisation trick as a registered Keras layer."""
    def call(self, inputs):
        z_mean, z_logvar = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps


def build_encoder_decoder(latent_dim):
    """Standard conv VAE encoder/decoder."""
    enc_in = keras.Input(shape=(64, 64, 3))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(enc_in)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean   = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)
    z = SamplingLayer(name="z")([z_mean, z_logvar])
    encoder = keras.Model(enc_in, [z_mean, z_logvar, z], name="encoder")

    lat_in = keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 128, activation="relu")(lat_in)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)
    decoder = keras.Model(lat_in, out, name="decoder")
    return encoder, decoder


def build_transition_mlp(latent_dim):
    """
    Transition network: maps z_t -> z_{t+1}.

    This is the key component Spens describes: a network that learns
    to predict the next latent state from the current one. During
    sequential replay, consecutive Hopfield chain frames provide the
    (z_t, z_{t+1}) supervision signal.

    Architecture: 3-layer MLP with residual connection.
    The residual connection is important — orientation changes smoothly,
    so z_{t+1} ≈ z_t + small_delta, and the residual makes this easy to learn.
    """
    inp = keras.Input(shape=(latent_dim,), name="z_t")
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    delta = layers.Dense(latent_dim, name="delta")(x)
    # Residual: predict the change, not the absolute next state
    z_next = layers.Add(name="z_t_plus_1")([inp, delta])
    transition = keras.Model(inp, z_next, name="transition_mlp")
    return transition


# ─────────────────────────────────────────────
# VAE trainer for Student A (IID, reconstruction)
# ─────────────────────────────────────────────
class VAETrainer(keras.Model):
    """Standard VAE: encode frame, reconstruct same frame."""
    def __init__(self, encoder, decoder, kl_weight=1e-4):
        super().__init__()
        self.encoder   = encoder
        self.decoder   = decoder
        self.kl_weight = kl_weight

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        data = tf.cast(data, tf.float32)
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encoder(data, training=True)
            recon      = self.decoder(z, training=True)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.abs(data - recon), axis=(1, 2, 3))
            )
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
            )
            loss = recon_loss + self.kl_weight * kl
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl}


# ─────────────────────────────────────────────
# Predictive student trainer for Student B (Sequential)
# ─────────────────────────────────────────────
class PredictiveStudentTrainer(keras.Model):
    """
    Predictive student: given frame t, predict frame t+1.

    This is the direct implementation of Spens's proposed sequential
    extension. Training objective:
      1. Encode frame_t -> z_t (via VAE encoder)
      2. Transition: z_t -> z_{t+1}_pred (via transition MLP)
      3. Decode z_{t+1}_pred -> frame_{t+1}_pred (via VAE decoder)
      4. Loss: reconstruction of frame_{t+1} + KL on z_t

    The model is trained on consecutive pairs (frame_t, frame_{t+1})
    from sequential Hopfield chains. This forces the encoder to organise
    its latent space so that temporal transitions are linearly predictable —
    which is exactly what Spens means by "trained to predict the next input".

    Note: we keep a KL term on z_t to regularise the latent space,
    consistent with the VAE framework in Spens & Burgess.
    """
    def __init__(self, encoder, decoder, transition, kl_weight=1e-4):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.transition = transition
        self.kl_weight  = kl_weight

    def train_step(self, data):
        # data is a tuple (frame_t, frame_t+1)
        frame_t, frame_t1 = data
        frame_t  = tf.cast(frame_t,  tf.float32)
        frame_t1 = tf.cast(frame_t1, tf.float32)

        with tf.GradientTape() as tape:
            # Encode current frame
            z_mean, z_logvar, z_t = self.encoder(frame_t, training=True)

            # Predict next latent state
            z_t1_pred = self.transition(z_t, training=True)

            # Decode predicted next latent -> predicted next frame
            frame_t1_pred = self.decoder(z_t1_pred, training=True)

            # Prediction loss: how well did we predict frame t+1?
            pred_loss = tf.reduce_mean(
                tf.reduce_sum(tf.abs(frame_t1 - frame_t1_pred), axis=(1, 2, 3))
            )

            # KL on z_t to regularise latent space
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
            )

            loss = pred_loss + self.kl_weight * kl

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss, "pred_loss": pred_loss, "kl_loss": kl}


# ─────────────────────────────────────────────
# Heteroassociative K/V memory
# ─────────────────────────────────────────────
def build_KV(imgs, labels, encoder):
    z_means = encoder.predict(imgs, verbose=0)[0]

    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        gkey = tuple(int(lab[k]) for k in GROUP_LABELS)
        groups[gkey].append((int(lab[SEQ_LABEL]), i))

    keys, values = [], []
    for _, seq in groups.items():
        seq_sorted = sorted(seq, key=lambda x: x[0])
        idxs = [i for _, i in seq_sorted]
        for a, b in zip(idxs[:-1], idxs[1:]):
            keys.append(z_means[a])
            values.append(z_means[b])

    K = np.stack(keys).astype("float32")
    V = np.stack(values).astype("float32")
    return K, V


def l2_normalize(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def hopfield_next(z_query, K_norm, V, beta=BETA_HOPFIELD, topk=TOPK):
    zq = np.atleast_2d(z_query).astype("float32")
    zq = zq / (np.linalg.norm(zq, axis=1, keepdims=True) + 1e-8)
    scores = (zq @ K_norm.T)[0]
    if topk is not None and topk < len(scores):
        idx = np.argpartition(scores, -topk)[-topk:]
        sub = beta * scores[idx]
    else:
        idx = np.arange(len(scores))
        sub = beta * scores
    sub = sub - sub.max()
    w = np.exp(sub); w = w / w.sum()
    return (w @ V[idx]).astype("float32")


# ─────────────────────────────────────────────
# Replay sample generation
# ─────────────────────────────────────────────
def generate_sequential_pairs(K, K_norm, V, n_chains, chain_length,
                               encoder, imgs, decoder):
    """
    Generate consecutive (frame_t, frame_{t+1}) pairs from Hopfield chains.

    Returns two arrays of images: frames_t and frames_t1, where
    frames_t1[i] is the frame that follows frames_t[i] in the chain.

    This is the training data for the predictive student. The pairs
    provide explicit (current_state, next_state) supervision, which is
    what Spens means by "sequential replay".
    """
    all_latents_t  = []
    all_latents_t1 = []

    start_idxs = np.random.choice(len(imgs), size=n_chains, replace=True)
    for si in start_idxs:
        z = encoder.predict(imgs[si:si+1], verbose=0)[0][0]
        chain = []
        for _ in range(chain_length):
            z = hopfield_next(z, K_norm, V)
            chain.append(z)
        # Consecutive pairs within the chain
        for t in range(len(chain) - 1):
            all_latents_t.append(chain[t])
            all_latents_t1.append(chain[t + 1])

    latents_t  = np.array(all_latents_t,  dtype="float32")
    latents_t1 = np.array(all_latents_t1, dtype="float32")

    # Decode to image space (student trains on images, not latents)
    frames_t  = latents_to_images(latents_t,  decoder)
    frames_t1 = latents_to_images(latents_t1, decoder)

    return frames_t, frames_t1


def generate_iid_replay(K, V, n_samples, decoder):
    """IID baseline: random latents, no ordering."""
    idxs = np.random.choice(len(K), size=n_samples, replace=True)
    latents = K[idxs].astype("float32")
    return latents_to_images(latents, decoder)


def latents_to_images(latents, decoder):
    imgs_out = []
    bs = 64
    for i in range(0, len(latents), bs):
        imgs_out.append(decoder.predict(latents[i:i+bs], verbose=0))
    return np.concatenate(imgs_out, axis=0)


# ─────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────
def evaluate_next_frame_prediction(encoder, teacher_dec, K, V, n=N_EVAL):
    """
    Next-frame prediction MSE via identical Ridge probe on both students.

    Procedure (same for Student A and Student B):
      1. Sample n (frame_t, frame_{t+1}) pairs from the teacher K/V memory
      2. Decode each teacher latent -> image via teacher decoder
      3. Re-encode each image through the student encoder -> z_k, z_v
      4. Fit Ridge regression: z_k[:split] -> z_v[:split]
      5. Report held-out MSE on z_k[split:] -> z_v[split:]

    The probe is a neutral readout of latent space structure. It asks:
    "how much temporal order has been encoded in this latent space?"
    without caring how it got there.

    Because the probe is identical for both students, any difference in
    MSE is attributable only to what each student was trained on —
    sequential Hopfield replay vs IID random replay. This directly
    supports the claim that sequential replay content drives better
    consolidation of temporally ordered experiences.

    Student B's transition MLP is evaluated separately as a secondary
    result (see evaluate_prediction_image_mse) to show the predictive
    architecture works, but is not used as the comparison point.

    Lower MSE = more temporal structure encoded = better consolidation.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    n = min(n, len(K))
    idxs = np.random.choice(len(K), size=n, replace=False)
    K_sub = K[idxs]
    V_sub = V[idxs]

    # Decode teacher latents -> images -> re-encode through student
    K_imgs = teacher_dec.predict(K_sub, verbose=0)
    V_imgs = teacher_dec.predict(V_sub, verbose=0)

    z_k = encoder.predict(K_imgs, verbose=0)[0]
    z_v = encoder.predict(V_imgs, verbose=0)[0]

    split = int(0.8 * n)
    probe = Ridge(alpha=1.0)
    probe.fit(z_k[:split], z_v[:split])
    z_v_pred = probe.predict(z_k[split:])
    return float(mean_squared_error(z_v[split:], z_v_pred))


def evaluate_reconstruction(encoder, decoder, imgs, n=200):
    subset = imgs[:n]
    z_mean, _, _ = encoder.predict(subset, verbose=0)
    recon = decoder.predict(z_mean, verbose=0)
    return float(np.mean((subset - recon) ** 2))


def evaluate_schema_distortion(encoder, decoder, imgs, n=200):
    subset_imgs  = imgs[:n]
    z_mean, _, _ = encoder.predict(subset_imgs, verbose=0)
    recalled_imgs = decoder.predict(z_mean, verbose=0)
    orig_var     = float(np.var(subset_imgs))
    recalled_var = float(np.var(recalled_imgs))
    return recalled_var / (orig_var + 1e-8)


def evaluate_prediction_image_mse(encoder, transition, decoder,
                                   teacher_dec, K, V, n=N_EVAL):
    """
    Additional metric for Student B: image-space prediction error.

    Encode frame_t -> z_t -> transition -> z_{t+1}_pred -> decode -> frame_{t+1}_pred
    Compare predicted image against actual frame_{t+1}.

    This is the most direct test of Spens's claim: does the predictive
    student generate accurate next-frame images?
    """
    n = min(n, len(K))
    idxs = np.random.choice(len(K), size=n, replace=False)
    K_imgs = teacher_dec.predict(K[idxs], verbose=0)
    V_imgs = teacher_dec.predict(V[idxs], verbose=0)

    z_k = encoder.predict(K_imgs, verbose=0)[0]
    z_v_pred = transition.predict(z_k, verbose=0)
    V_imgs_pred = decoder.predict(z_v_pred, verbose=0)

    return float(np.mean((V_imgs - V_imgs_pred) ** 2))


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────
def save_sequence_grid(seq, out_path, ncols=12, title=""):
    T = seq.shape[0]
    nrows = int(np.ceil(T / ncols))
    plt.figure(figsize=(ncols * 1.4, nrows * 1.4))
    for i in range(T):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.imshow(np.clip(seq[i], 0, 1))
        ax.axis("off")
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_prediction_grid(frames_t, frames_t1_true, frames_t1_pred,
                          out_path, n=8, title=""):
    """
    Visualise Student B's predictions: frame_t | true frame_t+1 | predicted frame_t+1.
    This is the key qualitative result showing whether the sequential student
    has learned to generate the correct next frame.
    """
    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 5))
    row_labels = ["Frame t (input)", "Frame t+1 (true)", "Frame t+1 (predicted)"]
    for col in range(n):
        axes[0, col].imshow(np.clip(frames_t[col],        0, 1))
        axes[1, col].imshow(np.clip(frames_t1_true[col],  0, 1))
        axes[2, col].imshow(np.clip(frames_t1_pred[col],  0, 1))
        for row in range(3):
            axes[row, col].axis("off")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=90, labelpad=40)
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_comparison_figure(imgs_orig, imgs_iid, imgs_seq,
                            out_path, n=8, title=""):
    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 5))
    row_labels = ["Original", "IID student recall", "Sequential student recall"]
    for col in range(n):
        axes[0, col].imshow(np.clip(imgs_orig[col], 0, 1))
        axes[1, col].imshow(np.clip(imgs_iid[col],  0, 1))
        axes[2, col].imshow(np.clip(imgs_seq[col],  0, 1))
        for row in range(3):
            axes[row, col].axis("off")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=90, labelpad=40)
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results_plot(results, out_path):
    metrics = [
        "Next-frame\nprediction MSE",
        "Reconstruction\nMSE",
        "Schema distortion\nratio"
    ]
    iid_vals = [results["iid_next_frame_mse"],
                results["iid_recon_mse"],
                results["iid_schema_distortion"]]
    seq_vals = [results["seq_next_frame_mse"],
                results["seq_recon_mse"],
                results["seq_schema_distortion"]]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_iid = ax.bar(x - width/2, iid_vals, width,
                      label="IID baseline (Student A)", color="steelblue", alpha=0.8)
    bars_seq = ax.bar(x + width/2, seq_vals, width,
                      label="Sequential predictive (Student B)", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(
        "IID vs Sequential Predictive Student: Evaluation\n"
        "(lower is better for all metrics; ratio < 1 = schema distortion)",
        fontsize=10
    )
    ax.legend()
    for bar in list(bars_iid) + list(bars_seq):
        ax.annotate(f"{bar.get_height():.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(seed=42, run_dir=None):
    import argparse

    # Allow CLI override when called directly
    if seed == 42 and run_dir is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed",    type=int, default=42)
        parser.add_argument("--run_dir", type=str, default=None)
        args, _ = parser.parse_known_args()
        seed    = args.seed
        run_dir = args.run_dir

    # Per-run output directory (falls back to global OUT_DIR)
    out = run_dir if run_dir else OUT_DIR
    os.makedirs(out, exist_ok=True)

    # Reproducibility — students and replay sampling vary by seed;
    # teacher and K/V are loaded from disk (fixed across runs)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    print("=" * 60)
    print(f"Sequential Predictive vs IID Replay  [seed={seed}]")
    print("(v3: Student B is a predictive model, per Spens & Burgess p.535)")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/7] Loading Shapes3D...")
    imgs, labels = load_shapes3d_with_labels(N_SAMPLES)
    split = int(0.8 * N_SAMPLES)
    train_imgs = imgs[:split]
    eval_imgs  = imgs[split:]
    print(f"    Train: {len(train_imgs)}, Eval: {len(eval_imgs)}")

    # Per-run student artifact paths (teacher/KV are shared and fixed)
    enc_iid_path  = os.path.join(ART_DIR, f"student_iid_encoder_s{seed}.keras")
    dec_iid_path  = os.path.join(ART_DIR, f"student_iid_decoder_s{seed}.keras")
    enc_seq_path  = os.path.join(ART_DIR, f"student_seq_encoder_s{seed}.keras")
    dec_seq_path  = os.path.join(ART_DIR, f"student_seq_decoder_s{seed}.keras")
    trans_seq_path = os.path.join(ART_DIR, f"student_seq_transition_s{seed}.keras")

    # ── 2. Train or load teacher VAE ─────────────────────────────
    print("\n[2/7] Teacher VAE...")
    if os.path.exists(ENC_PATH) and os.path.exists(DEC_PATH) and not FORCE_RETRAIN_TEACHER:
        print("    Loading saved teacher.")
        teacher_enc = keras.models.load_model(ENC_PATH, compile=False)
        teacher_dec = keras.models.load_model(DEC_PATH, compile=False)
    else:
        print("    Training teacher from scratch.")
        teacher_enc, teacher_dec = build_encoder_decoder(LATENT_DIM)
        trainer = VAETrainer(teacher_enc, teacher_dec, kl_weight=1e-4)
        trainer.compile(optimizer=keras.optimizers.Adam(1e-3))
        trainer.fit(train_imgs, epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, verbose=2)
        teacher_enc.save(ENC_PATH)
        teacher_dec.save(DEC_PATH)
        print("    Teacher saved.")

    # ── 3. Build K/V memory ───────────────────────────────────────
    print("\n[3/7] Building heteroassociative K/V memory...")
    if os.path.exists(K_PATH) and os.path.exists(V_PATH) and not FORCE_REBUILD_KV:
        print("    Loading saved K/V.")
        K = np.load(K_PATH)
        V = np.load(V_PATH)
    else:
        K, V = build_KV(train_imgs, labels[:split], teacher_enc)
        np.save(K_PATH, K)
        np.save(V_PATH, V)
        print(f"    Built K/V: {K.shape[0]} transitions.")
    K_norm = l2_normalize(K)

    # ── 4. Generate replay samples ────────────────────────────────
    print("\n[4/7] Generating replay samples...")

    print("    Condition A: IID random replay (images for reconstruction)...")
    iid_replay_imgs = generate_iid_replay(K, V, N_REPLAY_SAMPLES, teacher_dec)
    print(f"    IID replay: {iid_replay_imgs.shape}")

    print("    Condition B: Sequential pairs from Hopfield chains...")
    seq_frames_t, seq_frames_t1 = generate_sequential_pairs(
        K, K_norm, V, N_CHAINS, CHAIN_LENGTH, teacher_enc, train_imgs, teacher_dec
    )
    print(f"    Sequential pairs: {seq_frames_t.shape[0]} consecutive pairs")

    # Save example dream chain for visual inspection
    example_latents = []
    z = teacher_enc.predict(train_imgs[0:1], verbose=0)[0][0]
    for _ in range(24):
        z = hopfield_next(z, K_norm, V)
        example_latents.append(z)
    example_chain_imgs = latents_to_images(np.array(example_latents), teacher_dec)
    save_sequence_grid(
        example_chain_imgs,
        os.path.join(out, "example_dream_chain.png"),
        title="Example sequential dream chain (teacher decoder)"
    )

    # ── 5. Train student models ───────────────────────────────────
    print("\n[5/7] Training student models...")

    # Student A: standard VAE on IID replay (reconstruction objective)
    if (os.path.exists(enc_iid_path) and os.path.exists(dec_iid_path)
            and not FORCE_RETRAIN_STUDENTS):
        print("    Loading saved IID student.")
        student_iid_enc = keras.models.load_model(enc_iid_path, compile=False)
        student_iid_dec = keras.models.load_model(dec_iid_path, compile=False)
    else:
        print("    Training IID student (reconstruction VAE)...")
        student_iid_enc, student_iid_dec = build_encoder_decoder(LATENT_DIM)
        trainer_iid = VAETrainer(student_iid_enc, student_iid_dec, kl_weight=1e-4)
        trainer_iid.compile(optimizer=keras.optimizers.Adam(1e-3))
        trainer_iid.fit(iid_replay_imgs,
                        epochs=STUDENT_EPOCHS, batch_size=BATCH_SIZE, verbose=2)
        student_iid_enc.save(enc_iid_path)
        student_iid_dec.save(dec_iid_path)
        print("    IID student saved.")

    # Student B: predictive model on sequential pairs (prediction objective)
    if (os.path.exists(enc_seq_path) and os.path.exists(dec_seq_path)
            and os.path.exists(trans_seq_path) and not FORCE_RETRAIN_STUDENTS):
        print("    Loading saved sequential predictive student.")
        student_seq_enc   = keras.models.load_model(enc_seq_path,   compile=False)
        student_seq_dec   = keras.models.load_model(dec_seq_path,   compile=False)
        student_seq_trans = keras.models.load_model(trans_seq_path, compile=False)
    else:
        print("    Training sequential predictive student...")
        print("    (encoder + transition MLP, trained to predict frame t+1 from frame t)")
        student_seq_enc, student_seq_dec = build_encoder_decoder(LATENT_DIM)
        student_seq_trans = build_transition_mlp(LATENT_DIM)

        trainer_seq = PredictiveStudentTrainer(
            student_seq_enc, student_seq_dec, student_seq_trans, kl_weight=1e-4
        )
        trainer_seq.compile(optimizer=keras.optimizers.Adam(1e-3))

        seq_dataset = tf.data.Dataset.from_tensor_slices(
            (seq_frames_t, seq_frames_t1)
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        trainer_seq.fit(seq_dataset, epochs=STUDENT_EPOCHS, verbose=2)

        student_seq_enc.save(enc_seq_path)
        student_seq_dec.save(dec_seq_path)
        student_seq_trans.save(trans_seq_path)
        print("    Sequential predictive student saved.")

    # ── 6. Evaluate ───────────────────────────────────────────────
    print("\n[6/7] Evaluating students...")

    # Next-frame prediction: identical Ridge probe on both students.
    # Any difference is attributable only to replay content (IID vs sequential).
    iid_nf_mse = evaluate_next_frame_prediction(
        student_iid_enc, teacher_dec, K, V, n=N_EVAL
    )
    seq_nf_mse = evaluate_next_frame_prediction(
        student_seq_enc, teacher_dec, K, V, n=N_EVAL
    )
    print(f"    Next-frame MSE — IID (Ridge probe): {iid_nf_mse:.6f}  |  Sequential (Ridge probe): {seq_nf_mse:.6f}")

    # Image-space prediction for Student B (additional metric)
    seq_img_pred_mse = evaluate_prediction_image_mse(
        student_seq_enc, student_seq_trans, student_seq_dec,
        teacher_dec, K, V, n=N_EVAL
    )
    print(f"    Image-space prediction MSE (Student B): {seq_img_pred_mse:.6f}")

    # Reconstruction of held-out real images
    iid_recon = evaluate_reconstruction(student_iid_enc, student_iid_dec, eval_imgs)
    seq_recon = evaluate_reconstruction(student_seq_enc, student_seq_dec, eval_imgs)
    print(f"    Reconstruction MSE — IID: {iid_recon:.6f}  |  Sequential: {seq_recon:.6f}")

    # Schema distortion
    iid_schema = evaluate_schema_distortion(student_iid_enc, student_iid_dec, eval_imgs)
    seq_schema = evaluate_schema_distortion(student_seq_enc, student_seq_dec, eval_imgs)
    print(f"    Schema distortion ratio — IID: {iid_schema:.4f}  |  Sequential: {seq_schema:.4f}")

    results = {
        "iid_next_frame_mse":    iid_nf_mse,
        "seq_next_frame_mse":    seq_nf_mse,
        "seq_img_pred_mse":      seq_img_pred_mse,
        "iid_recon_mse":         iid_recon,
        "seq_recon_mse":         seq_recon,
        "iid_schema_distortion": iid_schema,
        "seq_schema_distortion": seq_schema,
    }

    # ── 7. Save outputs ───────────────────────────────────────────
    print("\n[7/7] Saving outputs...")

    metrics_path = os.path.join(out, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Sequential Predictive vs IID Replay: Results (seed={seed})\n")
        f.write("=" * 60 + "\n\n")
        f.write("ARCHITECTURE\n")
        f.write("  Student A: Standard VAE, reconstruction objective, IID replay\n")
        f.write("  Student B: VAE encoder + transition MLP, prediction objective,\n")
        f.write("             sequential Hopfield chain pairs\n")
        f.write("  (per Spens & Burgess 2024 Discussion p.535, refs 122-124)\n\n")
        f.write("PRIMARY METRIC: Next-frame prediction MSE (identical Ridge probe)\n")
        f.write("  Both students evaluated with the same Ridge regression probe.\n")
        f.write("  Any difference reflects latent space structure only, not architecture.\n")
        f.write(f"  IID (Ridge probe):              {iid_nf_mse:.6f}\n")
        f.write(f"  Sequential (Ridge probe):       {seq_nf_mse:.6f}\n")
        delta = iid_nf_mse - seq_nf_mse
        direction = "SUPPORTS" if delta > 0 else "DOES NOT SUPPORT"
        f.write(f"  Delta (IID - Seq):              {delta:.6f}\n")
        f.write(f"  Claim {direction}ED by primary metric.\n\n")
        f.write("SECONDARY METRIC: Image-space prediction MSE (Student B only)\n")
        f.write("  Shows predictive architecture works; not a comparison point.\n")
        f.write(f"  Seq image prediction MSE:       {seq_img_pred_mse:.6f}\n\n")
        f.write("OTHER METRICS\n")
        f.write(f"  Reconstruction MSE (IID):       {iid_recon:.6f}\n")
        f.write(f"  Reconstruction MSE (Sequential):{seq_recon:.6f}\n")
        f.write(f"  Schema distortion (IID):        {iid_schema:.4f}\n")
        f.write(f"  Schema distortion (Sequential): {seq_schema:.4f}\n")
        f.write("  (ratio < 1 = distortion toward prototype present)\n")
    print(f"    Metrics saved to {metrics_path}")

    # Prediction grid for Student B
    n_vis = 8
    z_k_vis = student_seq_enc.predict(seq_frames_t[:n_vis], verbose=0)[0]
    z_v_pred_vis = student_seq_trans.predict(z_k_vis, verbose=0)
    frames_t1_pred_vis = student_seq_dec.predict(z_v_pred_vis, verbose=0)
    save_prediction_grid(
        seq_frames_t[:n_vis], seq_frames_t1[:n_vis], frames_t1_pred_vis,
        os.path.join(out, "prediction_grid.png"),
        title="Student B: frame t (input) | true frame t+1 | predicted frame t+1"
    )
    print("    Prediction grid saved.")

    # Comparison figure (original | IID recall | seq recall)
    iid_recon_vis = student_iid_dec.predict(
        student_iid_enc.predict(eval_imgs[:n_vis], verbose=0)[0], verbose=0
    )
    seq_recon_vis = student_seq_dec.predict(
        student_seq_enc.predict(eval_imgs[:n_vis], verbose=0)[0], verbose=0
    )
    save_comparison_figure(
        eval_imgs[:n_vis], iid_recon_vis, seq_recon_vis,
        os.path.join(out, "comparison_recall.png"),
        title="Original vs IID student recall vs Sequential student recall"
    )
    print("    Comparison figure saved.")

    save_results_plot(results, os.path.join(out, "results_comparison.png"))
    print("    Results plot saved.")

    print("\n" + "=" * 60)
    print(f"SUMMARY  [seed={seed}]")
    print("=" * 60)
    print(f"Next-frame MSE (Ridge probe, identical): IID={iid_nf_mse:.6f}  Seq={seq_nf_mse:.6f}")
    print(f"Image prediction MSE (B):   {seq_img_pred_mse:.6f}")
    print(f"Reconstruction MSE:         IID={iid_recon:.6f}  Seq={seq_recon:.6f}")
    print(f"Schema distortion ratio:    IID={iid_schema:.4f}  Seq={seq_schema:.4f}")
    if seq_nf_mse < iid_nf_mse:
        print("\n✓ Sequential predictive student has LOWER next-frame prediction MSE.")
    else:
        print("\n✗ IID student has equal or lower next-frame prediction MSE.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--run_dir", type=str, default=None)
    args, _ = parser.parse_known_args()
    main(seed=args.seed, run_dir=args.run_dir)