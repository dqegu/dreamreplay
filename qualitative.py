from tensorflow import keras
from data import load_shapes3d
from experiment import _reconstruct_episode
from config import student_paths

import numpy as np
import matplotlib.pyplot as plt

# Load data
(train_imgs, train_seqs, _,
 test_imgs, test_seqs, _) = load_shapes3d()

seed = 42
paths = student_paths(seed)

# Load models (no training!)
teacher_enc = keras.models.load_model("artifacts_v2/teacher_encoder.keras", compile=False)
teacher_dec = keras.models.load_model("artifacts_v2/teacher_decoder.keras", compile=False)

seq_enc  = keras.models.load_model(paths["seq_vae_enc"], compile=False)
seq_dec  = keras.models.load_model(paths["seq_vae_dec"], compile=False)

shuf_enc = keras.models.load_model(paths["shuf_vae_enc"], compile=False)
shuf_dec = keras.models.load_model(paths["shuf_vae_dec"], compile=False)

iid_enc  = keras.models.load_model(paths["iid_enc"], compile=False)
iid_dec  = keras.models.load_model(paths["iid_dec"], compile=False)

# Pick test sequence
idx = 0
frames = test_imgs[test_seqs[idx]]

mask_idx = list(range(5, 10))

def reconstruct(enc, dec):
    return _reconstruct_episode(enc, dec, teacher_enc, teacher_dec, frames, mask_idx)

seq_recon  = reconstruct(seq_enc,  seq_dec)
shuf_recon = reconstruct(shuf_enc, shuf_dec)
iid_recon  = reconstruct(iid_enc,  iid_dec)

fig, ax = plt.subplots(5, 15, figsize=(15, 5))

def plot_row(row_data, row):
    for t in range(15):
        ax[row, t].imshow(row_data[t])
        ax[row, t].axis("off")

# Ground truth
plot_row(frames, 0)

# Corrupted
corrupted = frames.copy()
corrupted[5:10] = 0
plot_row(corrupted, 1)

# Models
plot_row(seq_recon, 2)
plot_row(shuf_recon, 3)
plot_row(iid_recon, 4)

plt.savefig("qualitative.png", dpi=200)
plt.show()