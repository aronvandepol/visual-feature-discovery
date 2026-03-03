"""
One-time extraction of frozen ResNet18 backbone features for all images.

For each image in the parquet:
  1. Load image as RGB
  2. Apply deterministic transforms (Resize 896, ToGray, CLAHE, Normalize)
  3. Convert to 3-channel tensor (3, 896, 896)
  4. Extract patches with unfold(p=8, s=4) -> (223, 223, 3, 8, 8)
  5. Reshape to (49729, 3, 8, 8)
  6. Run through frozen ResNet18 stages 1-3 in GPU batches
  7. After layer3 + adaptive_avg_pool2d(1,1): (49729, 256)
  8. Save as float16 .npy file

Usage:
  python extract_backbone_features.py \
      --parquet_file /path/to/data_mil.parquet \
      --base_dir /path/to/images \
      --output_dir features_p8s4 \
      --batch_size 5000 \
      --start_idx 0 --end_idx 1000
"""

import os
import argparse
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_deterministic_transform():
    """Same as ParquetPrintshopDataModule.val_test_transform"""
    return A.Compose([
        A.Resize(896, 896),
        A.ToGray(p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])


def build_backbone():
    """Build frozen ResNet18 stages 1-3 (same as AttriMIL backbone_stage1-3)."""
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # resnet.children(): conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
    # [:-2] removes avgpool + fc -> stages through layer4
    # [:7] keeps through layer3 -> output is 256-dim
    feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

    # We only need up to layer3 (index 7 in sequential, but [:7] = indices 0-6)
    # Actually feature_extractor[:7] = conv1, bn1, relu, maxpool, layer1, layer2, layer3
    backbone = nn.Sequential(*list(feature_extractor.children())[:7])

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    return backbone


def canonicalize_dataframe(df):
    """Apply the same printshop canonicalization as ParquetPrintshopDataset"""
    canon_flag = os.getenv("CANONICALIZE_PRINTSHOPS", "1")
    mapping_path = os.getenv("PRINTSHOP_MAP_JSON", "").strip()
    mapping = {}
    if canon_flag == "1":
        if mapping_path and os.path.exists(mapping_path):
            import json
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
        if not mapping:
            mapping = {
                '\ub300\ub3d9\uc778\uc1c4 \uc8fc\uc2dd\ud68c\uc0ac \u5927\u6771\u5370\u5237 \u682a\u5f0f\u6703\u793e': '\ub300\ub3d9\uc778\uc1c4\uc18c',
                '\ub300\ub3d9\uc778\uc1c4\uc18c \u5927\u6771\u5370\u5237\u6240': '\ub300\ub3d9\uc778\uc1c4\uc18c',
            }
        if 'printshop' in df.columns:
            df['printshop'] = df['printshop'].replace(mapping)
    return df


def image_to_feature_path(image_path, output_dir):
    """Generate a deterministic output path for an image's features."""
    # Use hash of the image path for a unique, filesystem-safe filename
    path_hash = hashlib.md5(image_path.encode()).hexdigest()
    basename = os.path.splitext(os.path.basename(image_path))[0]
    # Keep a readable prefix + hash for uniqueness
    filename = f"{basename}_{path_hash}.npy"
    return os.path.join(output_dir, filename)


def extract_features_for_image(img_path, transform, backbone, device, patch_size, stride, batch_size):
    """Extract backbone features for a single image."""
    # Load and transform image
    with Image.open(img_path) as img:
        img = img.convert('RGB')
        img_np = np.array(img)

    transformed = transform(image=img_np)
    img_tensor = transformed['image']  # (C, H, W), C=1 after ToGray

    # Ensure 3 channels (repeat grayscale)
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)
    elif img_tensor.shape[0] > 3:
        img_tensor = img_tensor[:3]

    img_tensor = img_tensor.float()

    # Extract patches: unfold(dim, size, step)
    c, h, w = img_tensor.shape
    patches = img_tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    # patches shape: (C, n_h, n_w, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    # patches shape: (n_h, n_w, C, patch_size, patch_size)
    n_h, n_w = patches.shape[0], patches.shape[1]
    n_patches = n_h * n_w
    patches = patches.view(n_patches, c, patch_size, patch_size)

    # Process through backbone in GPU batches
    all_features = []
    with torch.no_grad():
        for start in range(0, n_patches, batch_size):
            end = min(start + batch_size, n_patches)
            batch = patches[start:end].to(device)
            feats = backbone(batch)  # (batch, 256, h', w')
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)  # (batch, 256)
            all_features.append(feats.cpu())

    features = torch.cat(all_features, dim=0)  # (n_patches, 256)
    return features.half().numpy(), (n_h, n_w)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load parquet and canonicalize
    logger.info(f"Loading parquet file: {args.parquet_file}")
    df = pd.read_parquet(args.parquet_file)
    df = canonicalize_dataframe(df)

    # Build image paths
    samples = []
    for _, row in df.iterrows():
        image_path = os.path.join(args.base_dir, row['image_path'])
        samples.append(image_path)

    total = len(samples)
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else total
    end_idx = min(end_idx, total)

    logger.info(f"Total images: {total}, processing range [{start_idx}, {end_idx})")

    # Build backbone and transform
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    backbone = build_backbone().to(device)
    transform = build_deterministic_transform()

    # Process images
    processed = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for idx in range(start_idx, end_idx):
        img_path = samples[idx]
        out_path = image_to_feature_path(img_path, args.output_dir)

        # Skip if already extracted
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            features, (n_h, n_w) = extract_features_for_image(
                img_path, transform, backbone, device,
                args.patch_size, args.stride, args.batch_size
            )
            np.save(out_path, features)
            processed += 1

            if processed % 100 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                remaining = (end_idx - start_idx - processed - skipped) / max(rate, 1e-6)
                logger.info(
                    f"[{idx}/{end_idx}] Processed {processed}, skipped {skipped}, "
                    f"errors {errors} | {rate:.1f} img/s | ETA {remaining/60:.1f} min"
                )
                logger.info(f"  Feature shape: {features.shape}, dtype: {features.dtype}")

        except Exception as e:
            errors += 1
            logger.error(f"Error processing {img_path}: {e}")

    elapsed = time.time() - t0
    logger.info(
        f"Done. Processed {processed}, skipped {skipped}, errors {errors} "
        f"in {elapsed:.1f}s ({elapsed/60:.1f} min)"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frozen ResNet18 backbone features')
    parser.add_argument('--parquet_file', type=str, required=True,
                        help='Path to the parquet file')
    parser.add_argument('--base_dir', type=str, default='',
                        help='Base directory for image paths')
    parser.add_argument('--output_dir', type=str, default='features_p8s4',
                        help='Directory to save extracted features')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='GPU batch size for backbone inference')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Patch size for extraction')
    parser.add_argument('--stride', type=int, default=4,
                        help='Stride for patch extraction')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Start index (for processing a subset of the dataset)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index (for processing a subset of the dataset)')

    args = parser.parse_args()
    main(args)
