import os, numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import defaultdict

# --------------------
# Paths
# --------------------
BASE_DIR = "/home/ao1g22/spens_seq"
TFDS_DIR = os.path.join(BASE_DIR, "tfds_data")
ART_DIR  = os.path.join(BASE_DIR, "artifacts")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ENC_PATH = os.path.join(ART_DIR, "encoder.keras")
DEC_PATH = os.path.join(ART_DIR, "decoder.keras")
K_PATH   = os.path.join(ART_DIR, "K.npy")
V_PATH   = os.path.join(ART_DIR, "V.npy")

# --------------------
# Config
# --------------------
N_SAMPLES = 2000
LATENT_DIM = 32
BATCH_SIZE = 32
EPOCHS = 30

SEQ_LABEL = "label_orientation"
GROUP_LABELS = ("label_floor_hue","label_wall_hue","label_object_hue","label_scale","label_shape")

FORCE_RETRAIN = False
FORCE_REBUILD_KV = False


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
    imgs = (np.stack(imgs).astype("float32") / 255.0)
    return imgs, labels


def build_encoder_decoder(latent_dim):
    enc_in = keras.Input(shape=(64,64,3))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(enc_in)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)

    def sample_z(args):
        m, lv = args
        eps = tf.random.normal(shape=tf.shape(m))
        return m + tf.exp(0.5 * lv) * eps

    z = layers.Lambda(sample_z, name="z")([z_mean, z_logvar])
    encoder = keras.Model(enc_in, [z_mean, z_logvar, z], name="encoder")

    lat_in = keras.Input(shape=(latent_dim,))
    x = layers.Dense(8*8*128, activation="relu")(lat_in)
    x = layers.Reshape((8,8,128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)
    decoder = keras.Model(lat_in, out, name="decoder")
    return encoder, decoder


class VAETrainer(keras.Model):
    def __init__(self, encoder, decoder, kl_weight=1e-4):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        data = tf.cast(data, tf.float32)
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encoder(data, training=True)
            recon = self.decoder(z, training=True)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(data - recon), axis=(1,2,3)))
            kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1))
            loss = recon_loss + self.kl_weight * kl
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl}


def build_KV(imgs, labels, encoder):
    z_means = encoder.predict(imgs, verbose=0)[0]  # (N,D)

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


def hopfield_next(z_query, K_norm, V, beta=60.0, topk=50):
    zq = np.atleast_2d(z_query).astype("float32")
    zq = zq / (np.linalg.norm(zq, axis=1, keepdims=True) + 1e-8)

    scores = (zq @ K_norm.T)[0]
    if topk is not None and topk < len(scores):
        idx = np.argpartition(scores, -topk)[-topk:]
        sub = beta * scores[idx]
        sub = sub - sub.max()
        w = np.exp(sub); w = w / w.sum()
        return (w @ V[idx]).astype("float32")
    else:
        sub = beta * scores
        sub = sub - sub.max()
        w = np.exp(sub); w = w / w.sum()
        return (w @ V).astype("float32")


def transition_mse(K, V, K_norm, beta=60.0, topk=50, n=400):
    idxs = np.random.choice(len(K), size=min(n, len(K)), replace=False)
    errs = []
    for i in idxs:
        pred = hopfield_next(K[i], K_norm, V, beta=beta, topk=topk)
        errs.append(np.mean((pred - V[i])**2))
    return float(np.mean(errs))


def save_sequence_grid(seq, out_path, ncols=12, title=""):
    T = seq.shape[0]
    nrows = int(np.ceil(T / ncols))
    plt.figure(figsize=(ncols*1.4, nrows*1.4))
    for i in range(T):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.imshow(seq[i])
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    imgs, labels = load_shapes3d_with_labels(N_SAMPLES)

    # Load or train encoder/decoder
    if os.path.exists(ENC_PATH) and os.path.exists(DEC_PATH) and not FORCE_RETRAIN:
        encoder = keras.models.load_model(ENC_PATH, compile=False)
        decoder = keras.models.load_model(DEC_PATH, compile=False)
    else:
        encoder, decoder = build_encoder_decoder(LATENT_DIM)
        trainer = VAETrainer(encoder, decoder, kl_weight=1e-4)
        trainer.compile(optimizer=keras.optimizers.Adam(1e-3))
        trainer.fit(imgs, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
        encoder.save(ENC_PATH)
        decoder.save(DEC_PATH)

    # Build or load K/V
    if os.path.exists(K_PATH) and os.path.exists(V_PATH) and not FORCE_REBUILD_KV:
        K = np.load(K_PATH); V = np.load(V_PATH)
    else:
        K, V = build_KV(imgs, labels, encoder)
        np.save(K_PATH, K); np.save(V_PATH, V)

    K_norm = l2_normalize(K)
    base = float(np.mean((K - V)**2))
    mse = transition_mse(K, V, K_norm, beta=60.0, topk=50, n=400)

    # Generate a dream chain
    z = encoder.predict(imgs[0:1], verbose=0)[0][0]
    frames = []
    for _ in range(24):
        z = hopfield_next(z, K_norm, V, beta=60.0, topk=50)
        frames.append(decoder.predict(z[None, :], verbose=0)[0])
    frames = np.array(frames)

    # Save outputs
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"Baseline MSE: {base}\n")
        f.write(f"Transition MSE: {mse}\n")

    save_sequence_grid(frames, os.path.join(OUT_DIR, "dream_grid.png"),
                       title="Heteroassociative Hopfield sequential replay (orientation)")

    print("Done.")
    print("Baseline MSE:", base)
    print("Transition MSE:", mse)
    print("Saved:", os.path.join(OUT_DIR, "dream_grid.png"))


if __name__ == "__main__":
    main()
