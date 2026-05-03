"""
Pack individual .npy feature files into a single memory-mapped file for fast I/O.

Reads all .npy files from the feature directory and writes:
  - features_p8s4.dat: single memmap file, shape (N_images, N_patches, feature_dim) float16
  - features_p8s4_index.json: maps image_path hash -> row index in the memmap

This converts 66K random small-file reads into indexed access on a single large file,
which is dramatically faster on network filesystems like ZFS.

Usage:
  python pack_features.py \
      --parquet_file /path/to/data_mil.parquet \
      --base_dir /path/to/images \
      --feature_dir features_p8s4 \
      --output_prefix features_p8s4_packed
"""

import os
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def image_to_feature_key(image_path):
    """Same key generation as extract_backbone_features.py"""
    path_hash = hashlib.md5(image_path.encode()).hexdigest()
    basename = os.path.splitext(os.path.basename(image_path))[0]
    return f"{basename}_{path_hash}"


def main(args):
    # Load parquet to get ordered image list
    df = pd.read_parquet(args.parquet_file)

    # Canonicalize (same as extract script)
    import unicodedata
    canon_flag = os.getenv("CANONICALIZE_PRINTSHOPS", "1")
    if canon_flag == "1":
        mapping = {
            '\ub300\ub3d9\uc778\uc1c4 \uc8fc\uc2dd\ud68c\uc0ac \u5927\u6771\u5370\u5237 \u682a\u5f0f\u6703\u793e': '\ub300\ub3d9\uc778\uc1c4\uc18c',
            '\ub300\ub3d9\uc778\uc1c4\uc18c \u5927\u6771\u5370\u5237\u6240': '\ub300\ub3d9\uc778\uc1c4\uc18c',
        }
        if 'printshop' in df.columns:
            # Normalize Unicode form first. Parquet may store Hangul as NFD, which would
            # silently fail to match NFC mapping keys.
            df['printshop'] = df['printshop'].apply(
                lambda s: unicodedata.normalize('NFC', s) if isinstance(s, str) else s
            )
            df['printshop'] = df['printshop'].replace(mapping)

    # Build ordered list of feature keys and paths
    image_paths = []
    for _, row in df.iterrows():
        image_path = os.path.join(os.path.expanduser(args.base_dir), row['image_path'])
        image_paths.append(image_path)

    n_images = len(image_paths)
    logger.info(f"Total images in parquet: {n_images}")

    # Probe first file to get shape
    first_key = image_to_feature_key(image_paths[0])
    first_file = os.path.join(args.feature_dir, f"{first_key}.npy")
    probe = np.load(first_file)
    n_patches, feature_dim = probe.shape
    dtype = probe.dtype
    logger.info(f"Feature shape per image: ({n_patches}, {feature_dim}), dtype={dtype}")

    # Create memmap file
    dat_path = f"{args.output_prefix}.dat"
    index_path = f"{args.output_prefix}_index.json"

    logger.info(f"Creating memmap: {dat_path} shape=({n_images}, {n_patches}, {feature_dim}) {dtype}")
    total_bytes = n_images * n_patches * feature_dim * np.dtype(dtype).itemsize
    logger.info(f"Total size: {total_bytes / 1e9:.1f} GB")

    memmap = np.memmap(dat_path, dtype=dtype, mode='w+',
                       shape=(n_images, n_patches, feature_dim))

    # Build index and pack
    index = {
        'n_images': n_images,
        'n_patches': n_patches,
        'feature_dim': feature_dim,
        'dtype': str(dtype),
        'mapping': {},  # feature_key -> row index
    }

    t0 = time.time()
    missing = 0
    for i, image_path in enumerate(image_paths):
        key = image_to_feature_key(image_path)
        npy_path = os.path.join(args.feature_dir, f"{key}.npy")

        if not os.path.exists(npy_path):
            logger.warning(f"Missing feature file: {npy_path}")
            # Leave as zeros in memmap
            missing += 1
            index['mapping'][key] = i
            continue

        features = np.load(npy_path)
        memmap[i] = features
        index['mapping'][key] = i

        if (i + 1) % 5000 == 0:
            memmap.flush()
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_images - i - 1) / rate
            logger.info(f"[{i+1}/{n_images}] {rate:.0f} img/s, ETA {eta/60:.1f} min, missing={missing}")

    memmap.flush()
    del memmap

    # Also store a simple ordered list: index i -> feature_key
    index['ordered_keys'] = [image_to_feature_key(p) for p in image_paths]

    with open(index_path, 'w') as f:
        json.dump(index, f)

    elapsed = time.time() - t0
    logger.info(f"Done. Packed {n_images} images ({missing} missing) in {elapsed:.0f}s")
    logger.info(f"Output: {dat_path} ({os.path.getsize(dat_path)/1e9:.1f} GB)")
    logger.info(f"Index: {index_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack .npy features into single memmap')
    parser.add_argument('--parquet_file', type=str, required=True)
    parser.add_argument('--base_dir', type=str, default='')
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--output_prefix', type=str, default='features_p8s4_packed')
    args = parser.parse_args()
    main(args)
