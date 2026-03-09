# download_tfds.py
import tensorflow_datasets as tfds

DATA_DIR = "/home/ao1g22/spens_seq/tfds_data"
builder = tfds.builder("shapes3d", data_dir=DATA_DIR)
builder.download_and_prepare()
print("Done. TFDS cached at:", DATA_DIR)
