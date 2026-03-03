#!/usr/bin/env python3
"""
Configuration parameters for the comprehensive feature extraction pipeline.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# Canonical printshop name mapping (merge Daedong variants into one label)
PRINTSHOP_CANONICAL_MAP = {
    "대동인쇄 주식회사 大東印刷 株式會社": "대동인쇄소",
    "대동인쇄소 大東印刷所": "대동인쇄소",
}

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base directories -- override via environment variables
BASE_DIR = os.environ.get("DATASET_BASE_DIR", "")
PARQUET_FILE = os.environ.get("PARQUET_FILE", "")

# Local paths (for development/testing)
LOCAL_BASE_DIR = os.environ.get("LOCAL_DATASET_ROOT", "")
LOCAL_PARQUET_FILE = os.environ.get("LOCAL_PARQUET_FILE", "")

# Output directories
RESULTS_BASE = os.environ.get("RESULTS_BASE", "./feature_extraction/results")
LAST_COORDS_HINT = f"{RESULTS_BASE}/last_coordinates_parquet.txt"
EMBEDDINGS_DIR = f"{RESULTS_BASE}/embeddings"
SAMPLE_PATCHES_DIR = f"{RESULTS_BASE}/sample_patches"
VISUALIZATIONS_DIR = f"{RESULTS_BASE}/visualizations"
ENRICHED_PARQUET_DIR = f"{RESULTS_BASE}/enriched_parquet"
ANALYSIS_DIR = f"{RESULTS_BASE}/analysis"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# AttriMIL model parameters
N_SPLITS = 5
USE_GPU = True
BATCH_SIZE = 32
PATCH_SIZE = 8  # Patch size for AttriMIL model (matches AMIL_ATR_16_8_0)

# Best model selection (from existing extract_features.py logic)
EXTRACT_BEST_MODELS = True
EXPERIMENT_NAME = "AMIL_ATR_8_4_0"  # Default series to use
DEFAULT_FOLD = 0                   # Default fold when not using best-of-series

# ============================================================================
# COMET ML CONFIGURATION
# ============================================================================

# Comet ML settings for model downloading
COMET_PROJECT_NAME = "amil-kfold-cho"
COMET_WORKSPACE = os.environ.get("COMET_WORKSPACE", None)
USE_COMET_MODELS = True  # Download models from Comet vs local checkpoints
MODEL_CACHE_DIR = "./model_cache"  # Cache downloaded models locally

# Model selection preferences
PREFER_FINAL_MODEL = True  # Prefer final_model.pth over other model files
MODEL_DOWNLOAD_TIMEOUT = 300  # Timeout for model downloads (seconds)

# ============================================================================
# SAMPLING CONFIGURATION
# ============================================================================

# Dataset sampling mode
SAMPLE_MODE = "none"  # Options: "none" (full dataset), "random" (random sampling), "stratified"
SAMPLE_SIZE = 1000    # Number of rows to sample when SAMPLE_MODE != "none"
RANDOM_SEED = 42      # For reproducible sampling

# Statistical sampling parameters for patch extraction
MAX_PATCHES_PER_IMAGE = 50
MIN_PATCHES_PER_IMAGE = 5
ATTENTION_PERCENTILE_THRESHOLD = 75  # Use top 25% attention within each image
HIGH_ATTENTION_PERCENTILE = 90       # Top 10% are always included
TARGET_TOTAL_SAMPLES = 2000          # Total patch images to extract (per request)
SAMPLE_ZSCORE_K = 2.0                # Z-score threshold (mean + k*std) for high-attention option

# ============================================================================
# UMAP CONFIGURATION
# ============================================================================

# UMAP parameters for dimensionality reduction
UMAP_PARAMS = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'metric': 'cosine',
    'n_components': 2,
    'random_state': 42
}

# ============================================================================
# CLUSTERING CONFIGURATION
# ============================================================================

# HDBSCAN parameters (adapted for sample size)
HDBSCAN_PARAMS = {
    'min_cluster_size': 15,   # ~1.5% of 1000 samples
    'min_samples': 10,        # Smaller for test dataset
    'cluster_selection_method': 'eom',
    'max_cluster_size': 200,  # ~20% of 1000 samples
    'metric': 'euclidean',
    'cluster_selection_epsilon': 0.0,
    'alpha': 2.0,
    'n_jobs': -1
}

# Full dataset HDBSCAN parameters (for production runs)
HDBSCAN_PARAMS_FULL = {
    'min_cluster_size': 150,
    'min_samples': 125,
    'cluster_selection_method': 'eom',
    'max_cluster_size': 1800,
    'metric': 'euclidean',
    'cluster_selection_epsilon': 0.0,
    'alpha': 2.0,
    'n_jobs': -1
}

def get_hdbscan_params(dataset_size):
    """Get appropriate HDBSCAN parameters based on dataset size."""
    if dataset_size < 10000:
        print(f"Using small dataset HDBSCAN parameters for {dataset_size:,} samples")
        return HDBSCAN_PARAMS
    else:
        print(f"Using full dataset HDBSCAN parameters for {dataset_size:,} samples")
        return HDBSCAN_PARAMS_FULL

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Image display parameters
IMAGE_DISPLAY_SIZE = (32, 32)  # Size of patch images in UMAP plot
UMAP_FIGURE_SIZE = (20, 20)    # Size of UMAP visualization
CLUSTER_MOSAIC_SIZE = (800, 800)  # Size of cluster mosaic images
CLUSTER_IMAGE_SIZE = (64, 64)   # Size of individual images in cluster mosaics

# Plot parameters
MAX_IMAGES_IN_UMAP_PLOT = 2000  # Limit images displayed in UMAP (for performance)
MAX_IMAGES_PER_CLUSTER_PLOT = 100  # Max images per cluster visualization

# ============================================================================
# PROCESSING CONFIGURATION
# ============================================================================

# Memory and performance settings - OPTIMIZED FOR SLURM
CHUNK_SIZE = 10000           # Process embeddings in chunks
MAX_WORKERS = 4              # Parallel processing workers
BATCH_PROCESSING = True      # Enable batch processing for large datasets

# Memory optimization settings
MEMORY_LIMIT_GB = 64.0       # Maximum memory usage in GB
MEMORY_PRESSURE_THRESHOLD = 0.85   # Trigger optimization at 85% usage
BATCH_SIZE_MIN = 8           # Minimum batch size under memory pressure
BATCH_SIZE_MAX = 32          # Maximum batch size when memory is available
CHECKPOINT_INTERVAL = 50     # Save checkpoint every N row groups
ENABLE_MEMORY_MONITORING = True    # Real-time memory monitoring
EMERGENCY_CLEANUP_THRESHOLD = 0.95 # Emergency cleanup at 95% usage

# Adaptive batch sizing
ADAPTIVE_BATCHING = True     # Enable dynamic batch size adjustment
INITIAL_BATCH_SIZE = 16      # Starting batch size

# Parquet processing
PARQUET_ROW_GROUP_SIZE = 50000  # Row group size for efficient parquet operations

# ============================================================================
# CLUSTER DEPLOYMENT CONFIGURATION
# ============================================================================

# SLURM settings -- override via environment variables
SLURM_PARTITION = os.environ.get("SLURM_PARTITION_NAME", "YOUR_PARTITION")
SLURM_NODES = 1
SLURM_GPUS = 2
SLURM_MEMORY = "64G"
SLURM_TIME = "24:00:00"
SLURM_ACCOUNT = os.environ.get("SLURM_ACCOUNT_NAME", "YOUR_ACCOUNT")

# CUDA settings
CUDA_VISIBLE_DEVICES = "0,1"
NUMBA_CACHE_DIR = "/tmp/numba_cache_$$"
NUMBA_NUM_THREADS = 4

# ============================================================================
# OUTPUT FILE NAMES
# ============================================================================

# Generated file names
FULL_EMBEDDINGS_FILE = f"{EMBEDDINGS_DIR}/full_embeddings.npy"
EMBEDDING_METADATA_FILE = f"{EMBEDDINGS_DIR}/embedding_metadata.parquet"
UMAP_COORDINATES_FILE = f"{EMBEDDINGS_DIR}/umap_coordinates.npy"


# ====================================================================================
# UTILITIES
# ============================================================================


@lru_cache(maxsize=1)
def get_parquet_file() -> str:
    """Resolve the parquet file path, honoring environment overrides."""
    env_path = os.environ.get("PARQUET_FILE")
    if env_path and os.path.exists(env_path):
        return env_path
    if PARQUET_FILE and os.path.exists(PARQUET_FILE):
        return PARQUET_FILE
    if LOCAL_PARQUET_FILE and os.path.exists(LOCAL_PARQUET_FILE):
        return LOCAL_PARQUET_FILE
    # Fallback: return whatever env specified even if missing (caller will handle error)
    return env_path or PARQUET_FILE


@lru_cache(maxsize=1)
def get_image_path_resolver():
    """Return a callable that resolves an image path relative to configured roots."""
    candidates: List[str] = []
    env_root = os.environ.get("LOCAL_DATASET_ROOT")
    if env_root:
        candidates.append(env_root)
    if LOCAL_BASE_DIR:
        candidates.append(LOCAL_BASE_DIR)
    if BASE_DIR:
        candidates.append(BASE_DIR)
    normalized = [os.path.abspath(p) for p in candidates if p]

    def _resolver(path: str) -> Optional[str]:
        if not path:
            return None

        if os.path.isabs(path) and os.path.exists(path):
            return path

        def _add_variant(acc: list[str], candidate: str) -> None:
            candidate = candidate.strip()
            if candidate and candidate not in acc:
                acc.append(candidate)

        variants: list[str] = []
        _add_variant(variants, path)
        _add_variant(variants, path.lstrip("./"))

        cleaned = variants[0].lstrip("./")
        if cleaned.startswith("printshop_dataset/"):
            _, rest = cleaned.split("printshop_dataset/", 1)
            _add_variant(variants, rest)
            _add_variant(variants, os.path.join("printshop_dataset", rest))
            if rest.startswith("images/"):
                rest_data = rest.replace("images/", "data/", 1)
                _add_variant(variants, rest_data)
                _add_variant(variants, os.path.join("printshop_dataset", rest_data))

        if "images/" in cleaned:
            _add_variant(variants, cleaned.replace("images/", "data/", 1))

        for candidate in variants:
            # Try relative to known dataset roots
            for root in normalized:
                full = os.path.join(root, candidate)
                if os.path.exists(full):
                    return full
            # Fall back to CWD-relative lookup
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return None

    return _resolver


@lru_cache(maxsize=1)
def get_class_names() -> Tuple[str, ...]:
    """Return ordered class names expected by the model.

    The lookup checks an optional CLASS_NAMES_FILE env var (newline-delimited), then a
    baked-in default to avoid runtime surprises if the file is missing.
    """

    class_file = os.environ.get("CLASS_NAMES_FILE")
    if class_file and os.path.exists(class_file):
        with open(class_file, "r", encoding="utf-8") as fh:
            names = [line.strip() for line in fh if line.strip()]
        if names:
            return tuple(names)

    default_path = Path(RESULTS_BASE) / "class_names.txt"
    if default_path.exists():
        with default_path.open("r", encoding="utf-8") as fh:
            names = [line.strip() for line in fh if line.strip()]
        if names:
            return tuple(names)

    # Final fallback: infer from the parquet metadata to avoid single-class defaults.
    try:
        import pyarrow.parquet as pq

        parquet_path = get_parquet_file()
        if parquet_path and os.path.exists(parquet_path):
            pf = pq.ParquetFile(parquet_path)
            names_set: set[str] = set()
            for batch in pf.iter_batches(columns=["printshop"], batch_size=8192):
                column = batch.column(0)
                for raw_name in column.to_pylist():
                    if not raw_name:
                        continue
                    canonical = PRINTSHOP_CANONICAL_MAP.get(raw_name, raw_name)
                    names_set.add(canonical)
            if names_set:
                # Preserve deterministic order for downstream reproducibility
                return tuple(sorted(names_set))
    except Exception:
        pass

    # Fallback: single placeholder class; callers can override via env/file.
    return ("unknown",)
