"""
Shared model architecture used by both students (IID and sequential).

Teacher and both students use the same encoder/decoder so that any
difference in evaluation is attributable only to the replay strategy.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── VAE components ────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package="mmnist_exp")
class SamplingLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_logvar = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps


def build_encoder(latent_dim: int):
    inp = keras.Input(shape=(64, 64, 3))
    x = layers.Conv2D(32, 4, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean   = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)
    z        = SamplingLayer(name="z")([z_mean, z_logvar])
    return keras.Model(inp, [z_mean, z_logvar, z], name="encoder")


def build_decoder(latent_dim: int):
    inp = keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 128, activation="relu")(inp)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64,  4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32,  4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)
    return keras.Model(inp, x, name="decoder")


def build_transition(latent_dim: int):
    """Residual MLP: z_t → z_{t+1}.  Predicts a delta added to z_t."""
    inp = keras.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(256, activation="relu")(x)
    delta = layers.Dense(latent_dim)(x)
    z_next = layers.Add()([inp, delta])
    return keras.Model(inp, z_next, name="transition")


# ── VAE trainer (IID reconstruction objective) ────────────────────────────────

class VAETrainer(keras.Model):
    def __init__(self, encoder, decoder, kl_weight=1e-4, **kw):
        super().__init__(**kw)
        self.encoder   = encoder
        self.decoder   = decoder
        self.kl_weight = kl_weight

    def call(self, x, training=False):
        z_mean, z_logvar, z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def train_step(self, data):
        x = data[0] if isinstance(data, (list, tuple)) else data
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encoder(x, training=True)
            x_hat = self.decoder(z, training=True)
            recon = tf.reduce_mean(tf.reduce_sum(tf.abs(x - x_hat), axis=[1, 2, 3]))
            kl    = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1))
            loss  = recon + self.kl_weight * kl
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "recon": recon, "kl": kl}


# ── Sequential student trainer (prediction objective) ─────────────────────────

class SeqTrainer(keras.Model):
    """Given (frame_t, frame_{t+1}) pairs, learn to predict frame_{t+1} from frame_t."""

    def __init__(self, encoder, decoder, transition, kl_weight=1e-4, **kw):
        super().__init__(**kw)
        self.encoder    = encoder
        self.decoder    = decoder
        self.transition = transition
        self.kl_weight  = kl_weight

    def call(self, x, training=False):
        z_mean, _, z = self.encoder(x, training=training)
        z_next = self.transition(z, training=training)
        return self.decoder(z_next, training=training)

    def train_step(self, data):
        frames_t, frames_t1 = data[0], data[1]
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encoder(frames_t, training=True)
            z_next   = self.transition(z, training=True)
            pred_t1  = self.decoder(z_next, training=True)
            pred_loss = tf.reduce_mean(
                tf.reduce_sum(tf.abs(frames_t1 - pred_t1), axis=[1, 2, 3]))
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1))
            loss = pred_loss + self.kl_weight * kl
        grads = tape.gradient(loss, self."""
models.py
─────────────────────────────────────────────────────────────────────────────
Three model families:

1. Teacher VAE  (frame-level, unchanged from original Spens & Burgess)
   build_frame_encoder / build_frame_decoder / VAETrainer

2. Sequence VAE  (new — the sequential student)
   Takes a full 15-frame episode as input.
   Encoder: per-frame teacher latents → positional embedding →
            1D CNN → sequence latent z_seq (dim=SEQ_LATENT_DIM)
   Decoder: z_seq → 15 predicted teacher latents (in parallel) →
            frozen teacher decoder → 15 reconstructed frames

3. IID frame VAE  (original Spens & Burgess student, kept as baseline)
   Same architecture as the teacher but trained on MHN-replayed frames.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from config import LATENT_DIM, SEQ_LATENT_DIM, SEQ_LENGTH


# ═══════════════════════════════════════════════════════════════════════════
# Shared sampling layer
# ═══════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="irp_v2")
class SamplingLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_logvar = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps


# ═══════════════════════════════════════════════════════════════════════════
# 1. Teacher / IID frame-level VAE
# ═══════════════════════════════════════════════════════════════════════════

def build_frame_encoder(latent_dim: int = LATENT_DIM):
    """CNN encoder: (64,64,3) → (z_mean, z_logvar, z) each of shape (latent_dim,)."""
    inp = keras.Input(shape=(64, 64, 3))
    x = layers.Conv2D(32,  4, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64,  4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean   = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)
    z        = SamplingLayer(name="z")([z_mean, z_logvar])
    return keras.Model(inp, [z_mean, z_logvar, z], name="frame_encoder")


def build_frame_decoder(latent_dim: int = LATENT_DIM):
    """CNN decoder: (latent_dim,) → (64,64,3)."""
    inp = keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 128, activation="relu")(inp)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64,  4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32,  4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)
    return keras.Model(inp, x, name="frame_decoder")


class VAETrainer(keras.Model):
    """Trains a frame-level VAE with reconstruction + KL objective."""
    def __init__(self, encoder, decoder, kl_weight=1e-4, **kw):
        super().__init__(**kw)
        self.encoder   = encoder
        self.decoder   = decoder
        self.kl_weight = kl_weight

    def call(self, x, training=False):
        z_mean, z_logvar, z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def train_step(self, data):
        x = data[0] if isinstance(data, (list, tuple)) else data
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encoder(x, training=True)
            x_hat = self.decoder(z, training=True)
            recon = tf.reduce_mean(
                tf.reduce_sum(tf.abs(x - x_hat), axis=[1, 2, 3]))
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_logvar - tf.square(z_mean) -
                              tf.exp(z_logvar), axis=1))
            loss = recon + self.kl_weight * kl
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "recon": recon, "kl": kl}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Sequence VAE
# ═══════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="irp_v2")
class PositionalEmbedding(layers.Layer):
    """
    Learned positional embedding for sequences of fixed length.
    Adds a trainable (SEQ_LENGTH, embed_dim) embedding to the input.
    """
    def __init__(self, seq_len: int, embed_dim: int, **kw):
        super().__init__(**kw)
        self.seq_len   = seq_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(self.seq_len, self.embed_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        # x: (batch, seq_len, embed_dim)
        return x + self.pos_embed

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "embed_dim": self.embed_dim})
        return cfg


def build_seq_encoder(latent_dim: int = LATENT_DIM,
                       seq_latent_dim: int = SEQ_LATENT_DIM,
                       seq_len: int = SEQ_LENGTH):
    """
    Sequence encoder.

    Input : (seq_len, latent_dim)  — sequence of teacher frame latents
    Output: z_mean, z_logvar, z_seq  each of shape (seq_latent_dim,)

    Architecture:
      1. Learned positional embedding added to frame latents
      2. 1D CNN: (seq_len, latent_dim) → (4, 128) feature map
      3. Flatten → Dense → (z_mean, z_logvar) → reparameterise
    """
    inp = keras.Input(shape=(seq_len, latent_dim), name="frame_latent_seq")

    # Positional embedding
    x = PositionalEmbedding(seq_len, latent_dim, name="pos_embed")(inp)

    # 1D CNN — using Conv2D with height=1 as a clean Keras-compatible Conv1D
    # Conv1D layer: (batch, seq_len, channels)
    x = layers.Conv1D(64,  3, strides=1, padding="same", activation="relu")(x)
    x = layers.Conv1D(128, 3, strides=2, padding="same", activation="relu")(x)  # →8
    x = layers.Conv1D(128, 3, strides=2, padding="same", activation="relu")(x)  # →4

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    z_mean   = layers.Dense(seq_latent_dim, name="z_seq_mean")(x)
    z_logvar = layers.Dense(seq_latent_dim, name="z_seq_logvar")(x)
    z_seq    = SamplingLayer(name="z_seq")([z_mean, z_logvar])

    return keras.Model(inp, [z_mean, z_logvar, z_seq], name="seq_encoder")


def build_seq_decoder(latent_dim: int = LATENT_DIM,
                       seq_latent_dim: int = SEQ_LATENT_DIM,
                       seq_len: int = SEQ_LENGTH):
    """
    Sequence decoder.

    Input : z_seq of shape (seq_latent_dim,)
    Output: predicted teacher latents of shape (seq_len, latent_dim)

    All seq_len frame latents are predicted in parallel from z_seq.
    Architecture: 4-layer MLP with growing width.
    """
    inp = keras.Input(shape=(seq_latent_dim,), name="z_seq")
    x = layers.Dense(256,  activation="relu")(inp)
    x = layers.Dense(512,  activation="relu")(x)
    x = layers.Dense(seq_len * 64, activation="relu")(x)
    x = layers.Reshape((seq_len, 64))(x)
    # Per-frame projection to latent_dim
    pred_latents = layers.TimeDistributed(
        layers.Dense(latent_dim), name="pred_teacher_latents")(x)
    # shape: (seq_len, latent_dim)
    return keras.Model(inp, pred_latents, name="seq_decoder")


class SeqVAETrainer(keras.Model):
    """
    Trains the sequence VAE.

    Forward pass:
      replay_frames (batch, seq_len, 64, 64, 3)
        → frozen teacher encoder → frame latents (batch, seq_len, latent_dim)
        → seq encoder → z_seq
        → seq decoder → pred teacher latents (batch, seq_len, latent_dim)
        → frozen teacher decoder → recon frames (batch, seq_len, 64, 64, 3)

    Loss: pixel reconstruction MSE + beta * KL(z_seq)
    """
    def __init__(self, seq_encoder, seq_decoder,
                 teacher_encoder, teacher_decoder,
                 kl_weight: float = 1e-4, **kw):
        super().__init__(**kw)
        self.seq_encoder     = seq_encoder
        self.seq_decoder     = seq_decoder
        self.teacher_encoder = teacher_encoder
        self.teacher_decoder = teacher_decoder
        self.kl_weight       = kl_weight

        # Freeze teacher
        self.teacher_encoder.trainable = False
        self.teacher_decoder.trainable = False

    def _encode_sequence(self, frames_seq, training=False):
        """
        Encode each frame independently through the frozen teacher encoder.
        frames_seq: (batch, seq_len, 64, 64, 3)
        Returns frame_latents: (batch, seq_len, latent_dim)
        """
        batch_size = tf.shape(frames_seq)[0]
        # Flatten batch and seq dims for teacher encoder
        frames_flat = tf.reshape(frames_seq,
                                  [-1, 64, 64, 3])           # (B*T, 64, 64, 3)
        z_mean_flat, _, _ = self.teacher_encoder(frames_flat, training=False)
        # Use z_mean (deterministic) as the frame embedding
        frame_latents = tf.reshape(z_mean_flat,
                                    [batch_size, SEQ_LENGTH, LATENT_DIM])
        return frame_latents

    def _decode_sequence(self, pred_latents, training=False):
        """
        Decode each predicted teacher latent independently through frozen decoder.
        pred_latents: (batch, seq_len, latent_dim)
        Returns recon_frames: (batch, seq_len, 64, 64, 3)
        """
        batch_size = tf.shape(pred_latents)[0]
        latents_flat = tf.reshape(pred_latents,
                                   [-1, LATENT_DIM])          # (B*T, latent_dim)
        frames_flat = self.teacher_decoder(latents_flat, training=False)
        return tf.reshape(frames_flat, [batch_size, SEQ_LENGTH, 64, 64, 3])

    def call(self, frames_seq, training=False):
        frame_latents = self._encode_sequence(frames_seq, training)
        z_mean, z_logvar, z_seq = self.seq_encoder(frame_latents,
                                                     training=training)
        pred_latents  = self.seq_decoder(z_seq, training=training)
        recon_frames  = self._decode_sequence(pred_latents, training)
        return recon_frames, z_mean, z_logvar

    def train_step(self, data):
        # data: (batch, seq_len, 64, 64, 3)
        frames_seq = data[0] if isinstance(data, (list, tuple)) else data
        with tf.GradientTape() as tape:
            recon, z_mean, z_logvar = self(frames_seq, training=True)
            # Pixel MSE over all frames in sequence
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(frames_seq - recon), axis=[2, 3, 4]))
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_logvar - tf.square(z_mean) -
                              tf.exp(z_logvar), axis=1))
            loss = recon_loss + self.kl_weight * kl
        # Only update seq_encoder and seq_decoder
        trainable = (self.seq_encoder.trainable_variables +
                     self.seq_decoder.trainable_variables)
        grads = tape.gradient(loss, trainable)
        self.optimizer.apply_gradients(zip(grads, trainable))
        return {"loss": loss, "recon": recon_loss, "kl": kl}trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "pred_loss": pred_loss, "kl": kl}
