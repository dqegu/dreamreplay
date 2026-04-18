"""Download Shapes3D dataset to the configured TFDS directory."""
import tensorflow_datasets as tfds
from config import TFDS_DIR

print(f"Downloading shapes3d to {TFDS_DIR}...")
builder = tfds.builder("shapes3d", data_dir=TFDS_DIR)
builder.download_and_prepare()
print("Done.")
