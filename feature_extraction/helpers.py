#!/usr/bin/env python3
"""Utility helpers for patch extraction.

These helpers provide lightweight stand-ins for the original project utilities.
They cover image preprocessing, tensor reshaping into patch grids, and
percentile-based selection utilities that the extractor relies on.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision import transforms as _T
except ImportError:  # pragma: no cover - torchvision optional
    _T = None


def preprocess_image(path: str, input_size: int) -> torch.Tensor:
    """Load image and apply training preprocessing (grayscale + CLAHE) when available.

    Returns a tensor of shape (1, C, H, W) with C=3 to match model expectations.
    Falls back to the simplified torchvision pipeline if Albumentations is not installed.
    """
    img = Image.open(path).convert("RGB")

    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        A = None  # type: ignore[misc]

    if A is not None:
        img_np = np.array(img)
        transform = A.Compose([
            A.Resize(input_size, input_size),
            A.ToGray(p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2(),
        ])
        img_tensor = transform(image=img_np)["image"]
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.expand(3, -1, -1)
        img_tensor = img_tensor.unsqueeze(0)
        try:
            if torch.cuda.is_available():
                return img_tensor.pin_memory()
        except Exception:
            pass
        return img_tensor

    if _T is None:
        arr = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        arr = arr.permute(2, 0, 1)
        # Simple resize via interpolate to avoid torchvision dependency
        arr = arr.unsqueeze(0)
        arr = torch.nn.functional.interpolate(arr, size=(input_size, input_size), mode="bilinear", align_corners=False)
        return arr

    transform = _T.Compose([
        _T.Resize(input_size if isinstance(input_size, (tuple, list)) else (input_size, input_size)),
        _T.CenterCrop(input_size),
        _T.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor


def tensor_to_patch_grid(img_t: torch.Tensor, patch_size: int, stride: int) -> Tuple[torch.Tensor, int, int]:
    """Convert an image tensor into a grid of patches.

    Returns the flattened patch tensor (N, C, patch_size, patch_size) and grid dimensions.
    """
    if img_t.dim() != 4:
        raise ValueError(f"Expected tensor rank 4 (B,C,H,W); got {img_t.shape}")

    batch, channels, height, width = img_t.shape
    if batch != 1:
        raise ValueError("Only batch size 1 supported for patch extraction")

    gh = ((height - patch_size) // stride) + 1
    gw = ((width - patch_size) // stride) + 1

    patches = F.unfold(img_t, kernel_size=patch_size, stride=stride)
    patches = patches.transpose(1, 2)  # (B, L, C * patch_size * patch_size)
    patches = patches.reshape(-1, channels, patch_size, patch_size)
    return patches.contiguous(), int(gh), int(gw)


def normalize_minmax(values: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0,1] range with min-max scaling."""
    v_min = values.min()
    v_max = values.max()
    denom = torch.clamp(v_max - v_min, min=1e-8)
    return (values - v_min) / denom


def select_by_percentile(values: torch.Tensor, percentile: float):
    """Return mask selecting entries above the given percentile."""
    if values.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool), float("nan")
    pct = max(0.0, min(100.0, float(percentile)))
    threshold = torch.quantile(values, pct / 100.0)
    mask = values >= threshold
    return mask, float(threshold.item())
