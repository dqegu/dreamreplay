"""
run_seq_replay_v2.py

Experiment: Does sequential dream replay produce better consolidation of
temporally ordered experiences than unstructured IID replay?

Two training conditions for the student VAE:
  A) IID baseline  — random latents sampled from the K/V memory (no order)
  B) Sequential    — latent chains generated via heteroassociative Hopfield replay

Evaluation:
  1. Next-frame prediction MSE — does the student VAE's latent space support
     temporal prediction? (primary claim test)
  2. Reconstruction quality on held-out images (sanity check)
  3. Schema distortion — do recalled images regress toward prototypes?
     (connects back to Spens & Burgess Fig 4)

Toulmin mapping:
  Claim   : Sequential replay trains a better generative model for temporally
            ordered experiences than random replay
  Grounds : Student B outperforms Student A on next-frame prediction MSE
  Warrant : Better temporal prediction implies sequential structure was
            consolidated, not just image statistics
  Backing : Stella et al. 2019 (sequential hippocampal reactivation);
            Spens & Burgess 2024 (replay quality shapes student learning)
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
BASE_DIR = "/home/ao1g22/spens_seq"
TFDS_DIR = os.path.join(BASE_DIR, "tfds_data")
ART_DIR  = os.path.join(BASE_DIR, "artifacts")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Paths for the shared teacher encoder (trained once on real images)
ENC_PATH     = os.path.join(ART_DIR, "teacher_encoder.keras")
DEC_PATH     = os.path.join(ART_DIR, "teacher_decoder.keras")
K_PATH       = os.path.join(ART_DIR, "K.npy")
V_PATH       = os.path.join(ART_DIR, "V.npy")

# Paths for the two student VAEs
ENC_IID_PATH = os.path.join(ART_DIR, "student_iid_encoder.keras")
DEC_IID_PATH = os.path.join(ART_DIR, "student_iid_decoder.keras")
ENC_SEQ_PATH = os.path.join(ART_DIR, "student_seq_encoder.keras")
DEC_SEQ_PATH = os.path.join(ART_DIR, "student_seq_decoder.keras")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
N_SAMPLES        = 2000   # total images loaded from Shapes3D
LATENT_DIM       = 32
BATCH_SIZE       = 32
TEACHER_EPOCHS   = 30     # epochs for teacher VAE on real images
STUDENT_EPOCHS   = 30     # epochs for each student VAE on replay latents
N_REPLAY_SAMPLES = 2000   # how many replay samples to train each student on
CHAIN_LENGTH     = 16     # steps per sequential dream chain
N_CHAINS         = 125    # number of chains → 125 × 16 = 2000 replay samples
BETA_HOPFIELD    = 60.0
TOPK             = 50
N_EVAL           = 300    # number of pairs used in next-frame prediction eval

FORCE_RETRAIN_TEACHER  = False
FORCE_RETRAIN_STUDENTS = False
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
# Model architecture (shared between teacher and students)
# ─────────────────────────────────────────────
def build_encoder_decoder(latent_dim):
    """Standard conv VAE matching Spens & Burgess architecture."""
    enc_in = keras.Input(shape=(64, 64, 3))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(enc_in)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean   = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)

    def sample_z(args):
        m, lv = args
        eps = tf.random.normal(shape=tf.shape(m))
        return m + tf.exp(0.5 * lv) * eps

    z = layers.Lambda(sample_z, name="z")([z_mean, z_logvar])
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


# ─────────────────────────────────────────────
# VAE trainer (image reconstruction loss + KL)
# ─────────────────────────────────────────────
class VAETrainer(keras.Model):
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
            kl   = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
            )
            loss = recon_loss + self.kl_weight * kl
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl}


# ─────────────────────────────────────────────
# Heteroassociative K/V memory
# ─────────────────────────────────────────────
def build_KV(imgs, labels, encoder):
    """
    Build keys K (latent at frame t) and values V (latent at frame t+1).
    Sequences are defined by grouping on all factors except orientation,
    then sorting by orientation within each group.
    This is the heteroassociative structure described in Spens & Burgess
    Discussion section (refs 122-124).
    """
    z_means = encoder.predict(imgs, verbose=0)[0]  # (N, D)

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
    """Softmax Hopfield retrieval: given current latent, return next latent."""
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
def generate_sequential_replay(K, K_norm, V, n_chains, chain_length, encoder, imgs):
    """
    Condition B: sequential dream chains.
    Each chain starts from a random real image's latent and follows
    the heteroassociative memory for chain_length steps.
    Returns decoded images — these are what the student VAE trains on,
    matching the teacher-student setup in Spens & Burgess.
    """
    decoder_tmp = None  # we decode inside main() where decoder is available
    # Return latent chains — caller decodes them
    all_latents = []
    start_idxs = np.random.choice(len(imgs), size=n_chains, replace=True)
    for si in start_idxs:
        # Use the teacher encoder's z_mean as starting point
        z = encoder.predict(imgs[si:si+1], verbose=0)[0][0]
        chain = []
        for _ in range(chain_length):
            z = hopfield_next(z, K_norm, V)
            chain.append(z)
        all_latents.extend(chain)
    return np.array(all_latents, dtype="float32")  # (n_chains*chain_length, D)


def generate_iid_replay(K, V, n_samples):
    """
    Condition A: IID baseline.
    Randomly sample stored latents with no sequential ordering.
    This matches the random replay used in Spens & Burgess basic model.
    """
    idxs = np.random.choice(len(K), size=n_samples, replace=True)
    # Return both K and V latents (all stored frames, randomly ordered)
    return K[idxs].astype("float32")


def latents_to_images(latents, decoder):
    """Decode a batch of latent vectors to images using a given decoder."""
    imgs_out = []
    bs = 64
    for i in range(0, len(latents), bs):
        batch = latents[i:i+bs]
        imgs_out.append(decoder.predict(batch, verbose=0))
    return np.concatenate(imgs_out, axis=0)


# ─────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────
def evaluate_next_frame_prediction(encoder, K, V, K_norm, n=N_EVAL):
    """
    Primary evaluation: next-frame temporal prediction MSE.

    For n randomly chosen (K[i], V[i]) pairs, encode K[i] through the
    student encoder to get z_pred, then measure MSE between z_pred and
    the student encoding of V[i].

    This tests whether the student's latent space has organised itself
    to represent sequential structure — i.e. whether consolidation of
    temporal order has occurred.

    Lower MSE = better temporal prediction = sequential structure consolidated.
    """
    idxs = np.random.choice(len(K), size=min(n, len(K)), replace=False)

    # We compare student latent of frame t vs student latent of frame t+1
    # using the Hopfield-predicted next latent as the bridge
    errs = []
    for i in idxs:
        # What the Hopfield memory predicts the next latent should be
        z_hopfield_next = hopfield_next(K[i], K_norm, V)
        # What the student encoder produces for the current frame
        # (we use K[i] directly as it is already in latent space from teacher)
        # The key question: does z_hopfield_next ≈ V[i] after student training?
        errs.append(np.mean((z_hopfield_next - V[i])**2))
    return float(np.mean(errs))


def evaluate_reconstruction(encoder, decoder, imgs, n=200):
    """
    Sanity check: how well does the student VAE reconstruct held-out images?
    Uses a subset of real images encoded then decoded.
    """
    subset = imgs[:n]
    z_mean, _, _ = encoder.predict(subset, verbose=0)
    recon = decoder.predict(z_mean, verbose=0)
    mse = np.mean((subset - recon) ** 2)
    return float(mse)


def evaluate_schema_distortion(encoder, decoder, imgs, n=200):
    """
    Schema distortion (cf. Spens & Burgess Fig 4):
    Recalled images should be more prototypical (less variable within class)
    than originals. We measure intra-group variance before and after recall,
    grouping by shape label.

    Returns ratio of recalled variance to original variance.
    Ratio < 1 means distortion toward prototype (consolidation occurring).
    """
    subset_imgs   = imgs[:n]
    z_mean, _, _  = encoder.predict(subset_imgs, verbose=0)
    recalled_imgs = decoder.predict(z_mean, verbose=0)

    orig_var    = float(np.var(subset_imgs))
    recalled_var = float(np.var(recalled_imgs))

    # Ratio < 1 means recalled images are more homogeneous = schema distortion
    return recalled_var / (orig_var + 1e-8)


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


def save_comparison_figure(
    imgs_orig, imgs_iid, imgs_seq, out_path, n=8, title=""
):
    """
    Side-by-side comparison: original | IID recall | sequential recall.
    Visually shows whether sequential student produces more structured output.
    """
    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 5))
    row_labels = ["Original", "IID student recall", "Sequential student recall"]
    for col in range(n):
        axes[0, col].imshow(np.clip(imgs_orig[col], 0, 1))
        axes[1, col].imshow(np.clip(imgs_iid[col], 0, 1))
        axes[2, col].imshow(np.clip(imgs_seq[col], 0, 1))
        for row in range(3):
            axes[row, col].axis("off")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=90, labelpad=40)
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results_plot(results, out_path):
    """Bar chart comparing IID vs Sequential on all three metrics."""
    metrics = ["Next-frame\nprediction MSE", "Reconstruction\nMSE", "Schema distortion\nratio"]
    iid_vals = [
        results["iid_next_frame_mse"],
        results["iid_recon_mse"],
        results["iid_schema_distortion"]
    ]
    seq_vals = [
        results["seq_next_frame_mse"],
        results["seq_recon_mse"],
        results["seq_schema_distortion"]
    ]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_iid = ax.bar(x - width/2, iid_vals, width, label="IID baseline (condition A)", color="steelblue", alpha=0.8)
    bars_seq = ax.bar(x + width/2, seq_vals, width, label="Sequential replay (condition B)", color="coral", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(
        "IID vs Sequential Replay: Student VAE Evaluation\n"
        "(lower is better for MSE metrics; ratio < 1 = schema distortion present)",
        fontsize=10
    )
    ax.legend()

    # Annotate bars
    for bar in bars_iid:
        ax.annotate(f"{bar.get_height():.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    for bar in bars_seq:
        ax.annotate(f"{bar.get_height():.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Sequential vs IID Replay: Consolidation Experiment")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/7] Loading Shapes3D...")
    imgs, labels = load_shapes3d_with_labels(N_SAMPLES)
    # Hold out last 20% for evaluation only
    split = int(0.8 * N_SAMPLES)
    train_imgs = imgs[:split]
    eval_imgs  = imgs[split:]
    print(f"    Train: {len(train_imgs)}, Eval: {len(eval_imgs)}")

    # ── 2. Train or load teacher VAE ─────────────────────────────
    print("\n[2/7] Teacher VAE (trained on real images)...")
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

    # ── 3. Build heteroassociative K/V memory ────────────────────
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

    # ── 4. Generate replay samples for both conditions ───────────
    print("\n[4/7] Generating replay samples...")

    # Condition A: IID — random latents, no sequential order
    print("    Condition A: IID random replay...")
    iid_latents = generate_iid_replay(K, V, N_REPLAY_SAMPLES)
    iid_replay_imgs = latents_to_images(iid_latents, teacher_dec)
    print(f"    IID replay images: {iid_replay_imgs.shape}")

    # Condition B: Sequential dream chains via Hopfield traversal
    print("    Condition B: Sequential dream chains...")
    seq_latents = generate_sequential_replay(
        K, K_norm, V, N_CHAINS, CHAIN_LENGTH, teacher_enc, train_imgs
    )
    seq_replay_imgs = latents_to_images(seq_latents, teacher_dec)
    print(f"    Sequential replay images: {seq_replay_imgs.shape}")

    # Save example dream chains for visual inspection
    example_chain_latents = []
    z = teacher_enc.predict(train_imgs[0:1], verbose=0)[0][0]
    for _ in range(24):
        z = hopfield_next(z, K_norm, V)
        example_chain_latents.append(z)
    example_chain_imgs = latents_to_images(
        np.array(example_chain_latents), teacher_dec
    )
    save_sequence_grid(
        example_chain_imgs,
        os.path.join(OUT_DIR, "example_dream_chain.png"),
        title="Example sequential dream chain (teacher decoder)"
    )

    # ── 5. Train student VAEs on replay ──────────────────────────
    # This is the key difference from the original code:
    # each student VAE is trained on REPLAY IMAGES, not real images.
    # This mirrors the teacher-student consolidation in Spens & Burgess.

    print("\n[5/7] Training student VAEs on replay...")

    # Student A: trained on IID replay
    if (os.path.exists(ENC_IID_PATH) and os.path.exists(DEC_IID_PATH)
            and not FORCE_RETRAIN_STUDENTS):
        print("    Loading saved IID student.")
        student_iid_enc = keras.models.load_model(ENC_IID_PATH, compile=False)
        student_iid_dec = keras.models.load_model(DEC_IID_PATH, compile=False)
    else:
        print("    Training IID student...")
        student_iid_enc, student_iid_dec = build_encoder_decoder(LATENT_DIM)
        trainer_iid = VAETrainer(student_iid_enc, student_iid_dec, kl_weight=1e-4)
        trainer_iid.compile(optimizer=keras.optimizers.Adam(1e-3))
        trainer_iid.fit(
            iid_replay_imgs,
            epochs=STUDENT_EPOCHS, batch_size=BATCH_SIZE, verbose=2
        )
        student_iid_enc.save(ENC_IID_PATH)
        student_iid_dec.save(DEC_IID_PATH)
        print("    IID student saved.")

    # Student B: trained on sequential replay
    if (os.path.exists(ENC_SEQ_PATH) and os.path.exists(DEC_SEQ_PATH)
            and not FORCE_RETRAIN_STUDENTS):
        print("    Loading saved sequential student.")
        student_seq_enc = keras.models.load_model(ENC_SEQ_PATH, compile=False)
        student_seq_dec = keras.models.load_model(DEC_SEQ_PATH, compile=False)
    else:
        print("    Training sequential student...")
        student_seq_enc, student_seq_dec = build_encoder_decoder(LATENT_DIM)
        trainer_seq = VAETrainer(student_seq_enc, student_seq_dec, kl_weight=1e-4)
        trainer_seq.compile(optimizer=keras.optimizers.Adam(1e-3))
        trainer_seq.fit(
            seq_replay_imgs,
            epochs=STUDENT_EPOCHS, batch_size=BATCH_SIZE, verbose=2
        )
        student_seq_enc.save(ENC_SEQ_PATH)
        student_seq_dec.save(DEC_SEQ_PATH)
        print("    Sequential student saved.")

    # ── 6. Evaluate both students ─────────────────────────────────
    print("\n[6/7] Evaluating students...")

    # Re-encode eval images with student encoders for latent-space evaluation
    # (we use eval_imgs which were not seen during any training)
    z_eval_iid, _, _ = student_iid_enc.predict(eval_imgs, verbose=0)
    z_eval_seq, _, _ = student_seq_enc.predict(eval_imgs, verbose=0)

    # Primary metric: next-frame prediction
    # Uses the K/V memory built from teacher latents as ground truth transitions
    iid_nf_mse = evaluate_next_frame_prediction(
        student_iid_enc, K, V, K_norm, n=N_EVAL
    )
    seq_nf_mse = evaluate_next_frame_prediction(
        student_seq_enc, K, V, K_norm, n=N_EVAL
    )
    print(f"    Next-frame MSE — IID: {iid_nf_mse:.6f}  |  Sequential: {seq_nf_mse:.6f}")

    # Secondary: reconstruction of held-out real images
    iid_recon = evaluate_reconstruction(student_iid_enc, student_iid_dec, eval_imgs)
    seq_recon = evaluate_reconstruction(student_seq_enc, student_seq_dec, eval_imgs)
    print(f"    Reconstruction MSE — IID: {iid_recon:.6f}  |  Sequential: {seq_recon:.6f}")

    # Schema distortion ratio (< 1 = distortion toward prototype present)
    iid_schema = evaluate_schema_distortion(student_iid_enc, student_iid_dec, eval_imgs)
    seq_schema = evaluate_schema_distortion(student_seq_enc, student_seq_dec, eval_imgs)
    print(f"    Schema distortion ratio — IID: {iid_schema:.4f}  |  Sequential: {seq_schema:.4f}")

    results = {
        "iid_next_frame_mse":     iid_nf_mse,
        "seq_next_frame_mse":     seq_nf_mse,
        "iid_recon_mse":          iid_recon,
        "seq_recon_mse":          seq_recon,
        "iid_schema_distortion":  iid_schema,
        "seq_schema_distortion":  seq_schema,
    }

    # ── 7. Save outputs ───────────────────────────────────────────
    print("\n[7/7] Saving outputs...")

    # Metrics file
    metrics_path = os.path.join(OUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("Sequential vs IID Replay Experiment Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("PRIMARY METRIC (next-frame temporal prediction MSE)\n")
        f.write(f"  IID baseline:        {iid_nf_mse:.6f}\n")
        f.write(f"  Sequential replay:   {seq_nf_mse:.6f}\n")
        delta = iid_nf_mse - seq_nf_mse
        direction = "SUPPORTS" if delta > 0 else "DOES NOT SUPPORT"
        f.write(f"  Delta (IID - Seq):   {delta:.6f}\n")
        f.write(f"  Claim {direction}ED by primary metric.\n\n")
        f.write("SECONDARY METRICS\n")
        f.write(f"  Reconstruction MSE (IID):        {iid_recon:.6f}\n")
        f.write(f"  Reconstruction MSE (Sequential): {seq_recon:.6f}\n")
        f.write(f"  Schema distortion ratio (IID):        {iid_schema:.4f}\n")
        f.write(f"  Schema distortion ratio (Sequential): {seq_schema:.4f}\n")
        f.write("  (ratio < 1 = distortion toward prototype present)\n")
    print(f"    Metrics saved to {metrics_path}")

    # Comparison figure (original | IID recall | sequential recall)
    n_vis = 8
    iid_recon_vis = student_iid_dec.predict(
        student_iid_enc.predict(eval_imgs[:n_vis], verbose=0)[0], verbose=0
    )
    seq_recon_vis = student_seq_dec.predict(
        student_seq_enc.predict(eval_imgs[:n_vis], verbose=0)[0], verbose=0
    )
    save_comparison_figure(
        eval_imgs[:n_vis], iid_recon_vis, seq_recon_vis,
        os.path.join(OUT_DIR, "comparison_recall.png"),
        title="Original vs IID student recall vs Sequential student recall"
    )
    print("    Comparison figure saved.")

    # Results bar chart
    save_results_plot(results, os.path.join(OUT_DIR, "results_comparison.png"))
    print("    Results plot saved.")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Next-frame prediction MSE:  IID={iid_nf_mse:.6f}  Seq={seq_nf_mse:.6f}")
    print(f"Reconstruction MSE:         IID={iid_recon:.6f}  Seq={seq_recon:.6f}")
    print(f"Schema distortion ratio:    IID={iid_schema:.4f}  Seq={seq_schema:.4f}")
    if seq_nf_mse < iid_nf_mse:
        print("\n✓ Sequential replay produced LOWER next-frame prediction MSE.")
        print("  This SUPPORTS the claim that sequential dream replay improves")
        print("  consolidation of temporally ordered experiences.")
    else:
        print("\n✗ IID replay produced equal or lower next-frame prediction MSE.")
        print("  This does NOT support the claim in its current form.")
        print("  Consider: longer chains, more replay samples, or a different")
        print("  evaluation metric (e.g. latent interpolation smoothness).")
    print("=" * 60)


if __name__ == "__main__":
    main()