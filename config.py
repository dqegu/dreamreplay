import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.environ.get("IRP_BASE_DIR", "/iridisfs/home/ao1g22/comp6228/irp")
TFDS_DIR = os.path.join(BASE_DIR, "tfds_data")
ART_DIR  = os.path.join(BASE_DIR, "artifacts_v2")
OUT_DIR  = os.path.join(BASE_DIR, "outputs_v2")

for _d in (ART_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Shapes3D ──────────────────────────────────────────────────────────────────
GROUP_FACTORS = ["floor_hue", "wall_hue", "object_hue", "scale", "shape"]
SEQ_FACTOR    = "orientation"
SEQ_LENGTH    = 15

PROBE_FACTORS = {
    "shape":       4,
    "floor_hue":  10,
    "wall_hue":   10,
    "object_hue": 10,
    "scale":       8,
    "orientation": 15,
}

N_TRAIN_GROUPS = 4000
N_TEST_GROUPS  = 1000

# ── Model ─────────────────────────────────────────────────────────────────────
IMG_SHAPE      = (64, 64, 3)
LATENT_DIM     = 32          # teacher frame-level latent dimension
SEQ_LATENT_DIM = 64          # sequence VAE bottleneck dimension
BATCH_SIZE     = 32

# ── Teacher training ──────────────────────────────────────────────────────────
TEACHER_EPOCHS = 40

# ── Student training ──────────────────────────────────────────────────────────
STUDENT_EPOCHS     = 30
STUDENT_KL_WEIGHT  = 1e-4

# Checkpoints saved at these epochs for Exp 2 (semanticisation over time)
CHECKPOINT_EPOCHS  = [5, 10, 20, 30]

# ── IID replay (Spens & Burgess baseline) ─────────────────────────────────────
N_IID_REPLAY  = 8000
BETA_MHN      = 100.0
MHN_NOISE_STD = 0.3

# ── Sequential replay ─────────────────────────────────────────────────────────
N_CHAINS     = 400
CHAIN_LENGTH = 15
BETA_KV      = 60.0
TOPK         = 50

# ── Multi-seed ────────────────────────────────────────────────────────────────
SEEDS = [0, 7, 42]

# ── Artifact paths ────────────────────────────────────────────────────────────
TEACHER_ENC_PATH = os.path.join(ART_DIR, "teacher_encoder.keras")
TEACHER_DEC_PATH = os.path.join(ART_DIR, "teacher_decoder.keras")
K_PATH           = os.path.join(ART_DIR, "K.npy")
V_PATH           = os.path.join(ART_DIR, "V.npy")


def student_paths(seed: int) -> dict:
    s = seed
    return {
        # Sequential seq-VAE (ordered replay)
        "seq_vae_enc":  os.path.join(ART_DIR, f"seq_vae_encoder_s{s}.keras"),
        "seq_vae_dec":  os.path.join(ART_DIR, f"seq_vae_decoder_s{s}.keras"),
        # Shuffled seq-VAE (same architecture, random frame order)
        "shuf_vae_enc": os.path.join(ART_DIR, f"shuf_vae_encoder_s{s}.keras"),
        "shuf_vae_dec": os.path.join(ART_DIR, f"shuf_vae_decoder_s{s}.keras"),
        # IID seq-VAE (same architecture, independent MHN frames — no temporal structure)
        "iid_enc":      os.path.join(ART_DIR, f"iid_vae_encoder_s{s}.keras"),
        "iid_dec":      os.path.join(ART_DIR, f"iid_vae_decoder_s{s}.keras"),
        # Checkpoint directories for semanticisation probing across epochs
        "seq_ckpt_dir": os.path.join(ART_DIR, f"seq_vae_checkpoints_s{s}"),
    }