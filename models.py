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
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "pred_loss": pred_loss, "kl": kl}
