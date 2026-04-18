"""
Shapes3D data loading and sequence construction.

Strategy
────────
Shapes3D has 480,000 images covering all combinations of 6 generative
factors.  We construct synthetic temporal sequences by fixing 5 factors
(floor_hue, wall_hue, object_hue, scale, shape) and sweeping the 6th
(orientation, 15 steps).  Each such sweep is one "episode".

Loading is done in a single streaming pass so it works regardless of
whether the HPC stores shards in canonical or hash-shuffled order.
All 480,000 images are streamed but only the selected groups are kept
in RAM (≈ 5000 groups × 15 frames × 64×64×3 uint8 ≈ 880 MB).

Returns
───────
imgs        (N_groups × 15, 64, 64, 3)  float32 in [0, 1]
sequences   list of N_groups lists, each containing 15 consecutive indices
labels      dict mapping probe-factor name → int array of shape (N_groups,)
            (same value for every frame in a group, so one value per group)
"""

import numpy as np
import tensorflow_datasets as tfds
from collections import defaultdict

from config import (
    TFDS_DIR, GROUP_FACTORS, SEQ_FACTOR, SEQ_LENGTH,
    N_TRAIN_GROUPS, N_TEST_GROUPS, PROBE_FACTORS,
)


def _stream_groups(tfds_dir: str) -> dict:
    """
    Single-pass streaming over all 480k Shapes3D examples.
    Returns complete_groups: dict mapping group_key → sorted list of
    (orientation_int, image_uint8) pairs, for all groups with all 15
    orientations present.
    """
    ds = tfds.load("shapes3d", split="train", data_dir=tfds_dir,
                   shuffle_files=False)
    raw = defaultdict(dict)

    print("  Streaming Shapes3D (480k images)...")
    for ex in tfds.as_numpy(ds):
        gkey = tuple(int(ex[f"label_{f}"]) for f in GROUP_FACTORS)
        ori  = int(ex[f"label_{SEQ_FACTOR}"])
        raw[gkey][ori] = ex["image"]   # uint8, (64, 64, 3)

    complete = {k: v for k, v in raw.items() if len(v) == SEQ_LENGTH}
    print(f"  Found {len(complete):,} complete groups (expected 32,000)")
    return complete


def _build_arrays(complete_groups: dict, selected_keys: list):
    """
    Given a subset of group keys, stack their images into a flat array
    and build sequence index lists.

    Returns
        imgs       (N × SEQ_LENGTH, 64, 64, 3)  float32 in [0, 1]
        sequences  list of N lists, each length SEQ_LENGTH
        labels     dict: probe_factor → int array (N,)
    """
    imgs_list = []
    label_dict = defaultdict(list)

    for gkey in selected_keys:
        frames = complete_groups[gkey]
        # sort by orientation so the sequence is ordered 0→14
        seq_imgs = np.stack([frames[o] for o in range(SEQ_LENGTH)])  # (15, 64, 64, 3)
        imgs_list.append(seq_imgs.astype("float32") / 255.0)

        # one label per group (constant across all 15 frames)
        for i, factor in enumerate(GROUP_FACTORS):
            if factor in PROBE_FACTORS:
                label_dict[factor].append(gkey[i])

    imgs = np.concatenate(imgs_list, axis=0)   # (N × SEQ_LENGTH, 64, 64, 3)
    N    = len(selected_keys)
    sequences = [list(range(i * SEQ_LENGTH, (i + 1) * SEQ_LENGTH))
                 for i in range(N)]
    labels = {k: np.array(v, dtype=int) for k, v in label_dict.items()}
    return imgs, sequences, labels


def load_shapes3d(n_train: int = N_TRAIN_GROUPS,
                  n_test:  int = N_TEST_GROUPS,
                  seed:    int = 0):
    """
    Load Shapes3D and return train/test splits as (imgs, sequences, labels).

    Both splits contain disjoint groups (different factor combinations) so
    probing on the test set measures genuine generalisation.
    """
    complete = _stream_groups(TFDS_DIR)

    all_keys = sorted(complete.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(all_keys)

    n_total = n_train + n_test
    if n_total > len(all_keys):
        raise ValueError(
            f"Requested {n_total} groups but only {len(all_keys)} complete groups exist.")

    train_keys = all_keys[:n_train]
    test_keys  = all_keys[n_train:n_total]

    print(f"  Building train split ({n_train} groups × {SEQ_LENGTH} frames)...")
    train_imgs, train_seqs, train_labels = _build_arrays(complete, train_keys)

    print(f"  Building test split  ({n_test}  groups × {SEQ_LENGTH} frames)...")
    test_imgs,  test_seqs,  test_labels  = _build_arrays(complete, test_keys)

    print(f"  Train images: {train_imgs.shape}, Test images: {test_imgs.shape}")
    return (train_imgs, train_seqs, train_labels,
            test_imgs,  test_seqs,  test_labels)
