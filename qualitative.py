import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# Important: import custom layers before loading .keras models
import models
from models import SamplingLayer

from data import load_shapes3d
from config import student_paths
from experiment import _reconstruct_episode


SEED = 42
SEQ_IDX = 0
MASK_IDX = list(range(5, 10))

ART_DIR = "artifacts_v2"
OUT_PATH = "qualitative_examples.png"


def load_keras_model(path):
    custom_objects = {
        "SamplingLayer": SamplingLayer,
        "irp_v2>SamplingLayer": SamplingLayer,
    }
    return keras.models.load_model(
        path,
        compile=False,
        custom_objects=custom_objects,
        safe_mode=False,
    )


def plot_row(ax, row, frames, label):
    for t in range(frames.shape[0]):
        ax[row, t].imshow(np.clip(frames[t], 0, 1))
        ax[row, t].axis("off")

        if t == 0:
            ax[row, t].set_ylabel(
                label,
                rotation=0,
                labelpad=45,
                va="center",
                fontsize=9,
            )


def main():
    print("Loading Shapes3D...")
    train_imgs, train_seqs, train_labels, test_imgs, test_seqs, test_labels = load_shapes3d()

    print("Loading models...")
    paths = student_paths(SEED)

    teacher_enc = load_keras_model(os.path.join(ART_DIR, "teacher_encoder.keras"))
    teacher_dec = load_keras_model(os.path.join(ART_DIR, "teacher_decoder.keras"))

    seq_enc = load_keras_model(paths["seq_vae_enc"])
    seq_dec = load_keras_model(paths["seq_vae_dec"])

    shuf_enc = load_keras_model(paths["shuf_vae_enc"])
    shuf_dec = load_keras_model(paths["shuf_vae_dec"])

    iid_enc = load_keras_model(paths["iid_enc"])
    iid_dec = load_keras_model(paths["iid_dec"])

    print(f"Using test sequence {SEQ_IDX}, seed {SEED}")
    frames = test_imgs[test_seqs[SEQ_IDX]]

    corrupted = frames.copy()
    corrupted[MASK_IDX] = 0.0

    print("Running reconstructions...")
    seq_recon = _reconstruct_episode(
        seq_enc, seq_dec, teacher_enc, teacher_dec, frames, MASK_IDX
    )

    shuf_recon = _reconstruct_episode(
        shuf_enc, shuf_dec, teacher_enc, teacher_dec, frames, MASK_IDX
    )

    iid_recon = _reconstruct_episode(
        iid_enc, iid_dec, teacher_enc, teacher_dec, frames, MASK_IDX
    )

    print("Plotting...")
    rows = [
        ("Ground truth", frames),
        ("Corrupted input", corrupted),
        ("Sequential", seq_recon),
        ("Shuffled", shuf_recon),
        ("IID", iid_recon),
    ]

    fig, ax = plt.subplots(len(rows), frames.shape[0], figsize=(15, 5))

    for row_idx, (label, row_frames) in enumerate(rows):
        plot_row(ax, row_idx, row_frames, label)

    for t in range(frames.shape[0]):
        ax[0, t].set_title(str(t), fontsize=8)

    plt.suptitle(
        "Partial-cue episode reconstruction: frames 5–9 masked",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved qualitative reconstruction figure to {OUT_PATH}")


if __name__ == "__main__":
    main()