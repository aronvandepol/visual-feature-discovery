#!/usr/bin/env python3
"""
Batch Accumulator for Large GPU Batch Processing
===============================================

Accumulates patches across images for efficient large-batch GPU processing.
Integrates with the LRU image cache for optimal performance.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

from .cache_manager import ImageCache

logger = logging.getLogger(__name__)

class BatchAccumulator:
    """Accumulate patches across images for large GPU batches."""

    def __init__(self,
                 target_batch_size: int = 256,
                 image_cache: Optional[ImageCache] = None,
                 preprocessing_config: Optional[Dict] = None):
        """
        Initialize the batch accumulator.

        Args:
            target_batch_size: Target batch size for GPU processing
            image_cache: Image cache for storing transformed images
            preprocessing_config: Configuration for preprocessing pipeline
        """
        self.target_batch_size = target_batch_size
        self.image_cache = image_cache or ImageCache()
        self.preprocessing_config = preprocessing_config or self._get_default_preprocessing()

        self.patches = []
        self.metadata = []
        self.current_batch_size = 0

        # Initialize preprocessing transform
        self.transform = self._build_transform()

        logger.info(f"Initialized batch accumulator with target batch size: {target_batch_size}")

    def _get_default_preprocessing(self) -> Dict:
        """Get default preprocessing configuration."""
        return {
            'grayscale': True,
            'clahe': {'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
            'normalize': {'mean': [0.485], 'std': [0.229]}
        }

    def _build_transform(self) -> A.Compose:
        """Build the preprocessing transform pipeline."""
        transforms = [
            A.Resize(896, 896),
            A.ToGray(p=1.0) if self.preprocessing_config.get('grayscale', True) else None,
        ]

        # Add CLAHE if configured
        if 'clahe' in self.preprocessing_config:
            clahe_config = self.preprocessing_config['clahe']
            transforms.append(A.CLAHE(**clahe_config, p=1.0))

        # Add normalization
        if 'normalize' in self.preprocessing_config:
            norm_config = self.preprocessing_config['normalize']
            transforms.append(A.Normalize(**norm_config))

        # Filter out None transforms and add ToTensorV2
        transforms = [t for t in transforms if t is not None]
        transforms.append(ToTensorV2())

        return A.Compose(transforms)

    def add_patches_from_image(self,
                              image_path: str,
                              patches_df: pd.DataFrame,
                              patch_size: int = 64,
                              stride: int = 32) -> bool:
        """
        Add patches from an image to the accumulator.

        Args:
            image_path: Path to the source image
            patches_df: DataFrame with patch information (coordinates, attention, etc.)
            patch_size: Size of patches to extract
            stride: Stride for patch extraction

        Returns:
            True if batch is ready for processing, False otherwise
        """
        # Load or get cached transformed image tensor (C, H, W)
        img_tensor = self.image_cache.get(image_path)
        if img_tensor is None:
            try:
                # Load and transform image
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
                    img_np = np.array(img)

                transformed = self.transform(image=img_np)
                img_tensor = transformed['image']

                # Ensure 3 channels (expand grayscale to RGB if needed)
                if img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.repeat(3, 1, 1)

                # Store in cache as numpy for memory efficiency
                self.image_cache.put(image_path, img_tensor.numpy())

            except Exception as e:
                logger.error(f"Error loading/transforming {image_path}: {e}")
                return False

        # Extract patches from the transformed image
        for _, patch_row in patches_df.iterrows():
            h_coord = int(patch_row['patch_coord_h'])
            w_coord = int(patch_row['patch_coord_w'])

            # Calculate pixel boundaries
            y1 = h_coord * patch_size
            x1 = w_coord * patch_size
            y2 = y1 + patch_size
            x2 = x1 + patch_size

            # Extract patch
            patch_tensor = img_tensor[:, y1:y2, x1:x2]

            # Store patch and metadata
            self.patches.append(patch_tensor)

            metadata = {
                'patch_id': patch_row.get('patch_id', f"{image_path}_{h_coord}_{w_coord}"),
                'source_image_path': patch_row['source_image_path'],
                'local_image_path': image_path,
                'patch_coord_h': h_coord,
                'patch_coord_w': w_coord,
                'attention_weight': patch_row.get('attention_weight', 0.0),
                'attribute_score': patch_row.get('attribute_score', 0.0),
                'model_patch_size': patch_size,
                'true_label_name': patch_row.get('true_label_name', 'unknown'),
                'image_name': patch_row.get('image_name', ''),
            }
            self.metadata.append(metadata)

            self.current_batch_size += 1

            # Check if batch is ready
            if self.current_batch_size >= self.target_batch_size:
                return True

        return False

    def get_batch(self) -> Tuple[Optional[torch.Tensor], List[Dict]]:
        """
        Get accumulated batch for processing.

        Returns:
            Tuple of (batch_tensor, batch_metadata) or (None, []) if empty
        """
        if not self.patches:
            return None, []

        # Stack patches into batch tensor
        batch_tensor = torch.stack(self.patches)

        # Copy metadata
        batch_metadata = self.metadata.copy()

        # Reset accumulator
        self.patches.clear()
        self.metadata.clear()
        self.current_batch_size = 0

        return batch_tensor, batch_metadata

    def has_data(self) -> bool:
        """Check if accumulator has patches."""
        return len(self.patches) > 0

    def get_current_size(self) -> int:
        """Get current number of accumulated patches."""
        return self.current_batch_size

    def clear(self):
        """Clear all accumulated data."""
        self.patches.clear()
        self.metadata.clear()
        self.current_batch_size = 0

    def __len__(self) -> int:
        """Return number of accumulated patches."""
        return self.current_batch_size


class AdaptiveBatchAccumulator(BatchAccumulator):
    """Batch accumulator with adaptive batch sizing based on memory pressure."""

    def __init__(self,
                 target_batch_size: int = 256,
                 min_batch_size: int = 8,
                 max_batch_size: int = 512,
                 memory_threshold: float = 0.85,
                 image_cache: Optional[ImageCache] = None,
                 preprocessing_config: Optional[Dict] = None):
        """
        Initialize adaptive batch accumulator.

        Args:
            target_batch_size: Initial target batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            memory_threshold: Memory usage threshold for adaptation
            image_cache: Image cache instance
            preprocessing_config: Preprocessing configuration
        """
        super().__init__(target_batch_size, image_cache, preprocessing_config)

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.adaptation_count = 0

        logger.info(f"Initialized adaptive batch accumulator: {min_batch_size}-{max_batch_size}")

    def adapt_batch_size(self, memory_usage: float, gpu_memory_usage: float) -> bool:
        """
        Adapt batch size based on memory usage.

        Args:
            memory_usage: System memory usage (0-1)
            gpu_memory_usage: GPU memory usage (0-1)

        Returns:
            True if batch size was adapted
        """
        adapted = False

        # Check memory pressure
        if memory_usage > self.memory_threshold or gpu_memory_usage > self.memory_threshold:
            # Reduce batch size
            old_size = self.target_batch_size
            self.target_batch_size = max(self.min_batch_size, self.target_batch_size // 2)
            if self.target_batch_size != old_size:
                adapted = True
                self.adaptation_count += 1
                logger.info(f"Reduced batch size from {old_size} to {self.target_batch_size} "
                          f"(mem={memory_usage:.2f}, gpu={gpu_memory_usage:.2f})")
        elif memory_usage < self.memory_threshold * 0.7 and gpu_memory_usage < self.memory_threshold * 0.7:
            # Increase batch size if memory is available
            old_size = self.target_batch_size
            self.target_batch_size = min(self.max_batch_size, self.target_batch_size * 2)
            if self.target_batch_size != old_size:
                adapted = True
                self.adaptation_count += 1
                logger.info(f"Increased batch size from {old_size} to {self.target_batch_size} "
                          f"(mem={memory_usage:.2f}, gpu={gpu_memory_usage:.2f})")

        return adapted

    def get_stats(self) -> Dict:
        """Get accumulator statistics."""
        return {
            'current_batch_size': self.current_batch_size,
            'target_batch_size': self.target_batch_size,
            'min_batch_size': self.min_batch_size,
            'max_batch_size': self.max_batch_size,
            'adaptation_count': self.adaptation_count,
            'memory_threshold': self.memory_threshold
        }