#!/usr/bin/env python3
"""
LRU Cache Manager for Image Caching
====================================

High-performance LRU cache for loaded images to eliminate redundant I/O operations.
Used in the integrated feature extraction pipeline for optimal performance.
"""

import os
import torch
import numpy as np
from collections import OrderedDict
from threading import Lock
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ImageCache:
    """LRU cache for loaded images to eliminate redundant I/O."""

    def __init__(self, max_size_gb: float = 8.0):
        """
        Initialize the image cache.

        Args:
            max_size_gb: Maximum cache size in GB
        """
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.cache = OrderedDict()
        self.lock = Lock()
        self.current_size = 0
        self.hits = 0
        self.misses = 0

        logger.info(f"Initialized image cache with {max_size_gb:.1f}GB capacity")

    def get(self, image_path: str) -> Optional[np.ndarray]:
        """
        Get image from cache.

        Args:
            image_path: Path to the image

        Returns:
            Cached image array or None if not found
        """
        with self.lock:
            if image_path in self.cache:
                self.hits += 1
                img_array = self.cache.pop(image_path)
                self.cache[image_path] = img_array  # Move to end (most recent)
                return img_array.copy()
            self.misses += 1
            return None

    def put(self, image_path: str, img_array: np.ndarray):
        """
        Add image to cache, evicting oldest if necessary.

        Args:
            image_path: Path to the image
            img_array: Image array to cache
        """
        # Support both numpy arrays and torch tensors (store as numpy arrays)
        if hasattr(img_array, 'nbytes'):
            img_size = img_array.nbytes
        elif hasattr(img_array, 'element_size') and hasattr(img_array, 'nelement'):
            img_size = img_array.element_size() * img_array.nelement()
        else:
            # Fallback size estimation
            img_size = img_array.size * 4  # Assume float32

        with self.lock:
            # Remove if already exists
            if image_path in self.cache:
                old_img = self.cache.pop(image_path)
                if hasattr(old_img, 'nbytes'):
                    old_size = old_img.nbytes
                elif hasattr(old_img, 'element_size') and hasattr(old_img, 'nelement'):
                    old_size = old_img.element_size() * old_img.nelement()
                else:
                    old_size = old_img.size * 4
                self.current_size -= old_size

            # Evict oldest items if necessary
            while self.current_size + img_size > self.max_size_bytes and self.cache:
                oldest_path, oldest_img = self.cache.popitem(last=False)
                if hasattr(oldest_img, 'nbytes'):
                    oldest_size = oldest_img.nbytes
                elif hasattr(oldest_img, 'element_size') and hasattr(oldest_img, 'nelement'):
                    oldest_size = oldest_img.element_size() * oldest_img.nelement()
                else:
                    oldest_size = oldest_img.size * 4
                self.current_size -= oldest_size

            # Add new image
            self.cache[image_path] = img_array.copy()
            self.current_size += img_size

    def clear(self):
        """Clear all cached images."""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'cache_size': len(self.cache),
                'memory_usage_gb': self.current_size / 1024**3,
                'hit_rate': hit_rate,
                'hits': self.hits,
                'misses': self.misses,
                'max_capacity_gb': self.max_size_bytes / 1024**3
            }

    def __len__(self) -> int:
        """Return number of cached images."""
        with self.lock:
            return len(self.cache)

    def __contains__(self, image_path: str) -> bool:
        """Check if image path is in cache."""
        with self.lock:
            return image_path in self.cache


class CacheManager:
    """Manager for multiple caches with different configurations."""

    def __init__(self):
        self.caches = {}
        self.default_cache_size_gb = 8.0

    def get_cache(self, name: str = 'default', size_gb: Optional[float] = None) -> ImageCache:
        """
        Get or create a cache by name.

        Args:
            name: Cache name
            size_gb: Cache size in GB (uses default if None)

        Returns:
            ImageCache instance
        """
        if name not in self.caches:
            cache_size = size_gb or self.default_cache_size_gb
            self.caches[name] = ImageCache(cache_size)

        return self.caches[name]

    def clear_all(self):
        """Clear all managed caches."""
        for cache in self.caches.values():
            cache.clear()

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.

        Returns:
            Dictionary mapping cache names to their statistics
        """
        return {name: cache.stats() for name, cache in self.caches.items()}


# Global cache manager instance
cache_manager = CacheManager()

def get_image_cache(name: str = 'default', size_gb: Optional[float] = None) -> ImageCache:
    """
    Get a global image cache instance.

    Args:
        name: Cache name
        size_gb: Cache size in GB

    Returns:
        ImageCache instance
    """
    return cache_manager.get_cache(name, size_gb)