"""
DataModule for loading pre-extracted backbone features.

Supports two modes:
  1. Packed memmap (fast): single .dat file + index JSON. Use --feature_dir to point
     to the prefix, e.g. --feature_dir features_p8s4_packed  (loads .dat + _index.json)
  2. Individual .npy files (slow fallback): --feature_dir points to a directory of .npy files

Feature-space augmentation (dropout + Gaussian noise) compensates for the loss of
pixel-level augmentation.
"""

import os
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import pytorch_lightning as pl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PackedFeatureDataset(Dataset):
    """Dataset backed by a single memory-mapped file for fast I/O."""

    def __init__(self, parquet_file, packed_prefix, base_dir=""):
        self.base_dir = os.path.expanduser(base_dir)

        # Load index
        index_path = f"{packed_prefix}_index.json"
        with open(index_path, 'r') as f:
            self.index = json.load(f)

        n_images = self.index['n_images']
        n_patches = self.index['n_patches']
        feature_dim = self.index['feature_dim']
        dtype = np.dtype(self.index['dtype'])

        # Open memmap read-only
        dat_path = f"{packed_prefix}.dat"
        self.memmap = np.memmap(dat_path, dtype=dtype, mode='r',
                                shape=(n_images, n_patches, feature_dim))

        logger.info(f"Packed memmap: {dat_path} shape={self.memmap.shape} dtype={dtype}")

        # Load parquet for labels
        self.df = pd.read_parquet(parquet_file)
        self._canonicalize()

        self.unique_printshops = self.df['printshop'].unique()
        self.printshop_to_idx = {p: idx for idx, p in enumerate(sorted(self.unique_printshops))}
        self.idx_to_printshop = {idx: p for p, idx in self.printshop_to_idx.items()}

        # Build sample list: (memmap_row, class_idx, book_id)
        self.samples = []
        self._prepare_samples()

        logger.info(f"Packed feature dataset: {len(self.samples)} images, "
                    f"{len(self.unique_printshops)} classes")

    def _canonicalize(self):
        import unicodedata
        canon_flag = os.getenv("CANONICALIZE_PRINTSHOPS", "1")
        mapping_path = os.getenv("PRINTSHOP_MAP_JSON", "").strip()
        mapping = {}
        if canon_flag == "1":
            if mapping_path and os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
            if not mapping:
                mapping = {
                    '대동인쇄 주식회사 大東印刷 株式會社': '대동인쇄소',
                    '대동인쇄소 大東印刷所': '대동인쇄소',
                }
            if 'printshop' in self.df.columns:
                # Normalize Unicode form first. Parquet may store Hangul as NFD, which would
                # silently fail to match NFC mapping keys.
                self.df['printshop'] = self.df['printshop'].apply(
                    lambda s: unicodedata.normalize('NFC', s) if isinstance(s, str) else s
                )
                self.df['printshop'] = self.df['printshop'].replace(mapping)

    def _prepare_samples(self):
        key_to_row = self.index['mapping']
        for _, row in self.df.iterrows():
            image_path = os.path.join(self.base_dir, row['image_path'])
            class_idx = self.printshop_to_idx[row['printshop']]
            book_id = row.get('book_id', row.get('book', ''))

            path_hash = hashlib.md5(image_path.encode()).hexdigest()
            basename = os.path.splitext(os.path.basename(image_path))[0]
            key = f"{basename}_{path_hash}"

            memmap_row = key_to_row.get(key)
            if memmap_row is None:
                logger.warning(f"Feature key not found in index: {key}")
                memmap_row = 0  # fallback to first row (will be wrong but won't crash)

            self.samples.append((memmap_row, class_idx, book_id))

    def __getitem__(self, idx):
        row, label, _ = self.samples[idx]
        features = torch.from_numpy(self.memmap[row].copy()).float()
        return features, label

    def __len__(self):
        return len(self.samples)

    @property
    def class_names(self):
        return [self.idx_to_printshop[i] for i in range(len(self.idx_to_printshop))]

    @property
    def book_ids(self):
        return [s[2] for s in self.samples]

    @property
    def labels(self):
        return [s[1] for s in self.samples]


class FeaturePrintshopDataset(Dataset):
    """Dataset that loads pre-extracted backbone features from individual .npy files."""

    def __init__(self, parquet_file, feature_dir, base_dir=""):
        self.feature_dir = feature_dir
        self.base_dir = os.path.expanduser(base_dir)

        self.df = pd.read_parquet(parquet_file)
        self._canonicalize()

        self.unique_printshops = self.df['printshop'].unique()
        self.printshop_to_idx = {p: idx for idx, p in enumerate(sorted(self.unique_printshops))}
        self.idx_to_printshop = {idx: p for p, idx in self.printshop_to_idx.items()}

        self.samples = []
        self._prepare_samples()

        logger.info(f"Feature dataset: {len(self.samples)} images, "
                    f"{len(self.unique_printshops)} classes, "
                    f"feature_dir={self.feature_dir}")

    def _canonicalize(self):
        import unicodedata
        canon_flag = os.getenv("CANONICALIZE_PRINTSHOPS", "1")
        mapping_path = os.getenv("PRINTSHOP_MAP_JSON", "").strip()
        mapping = {}
        if canon_flag == "1":
            if mapping_path and os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
            if not mapping:
                mapping = {
                    '대동인쇄 주식회사 大東印刷 株式會社': '대동인쇄소',
                    '대동인쇄소 大東印刷所': '대동인쇄소',
                }
            if 'printshop' in self.df.columns:
                # Normalize Unicode form first. Parquet may store Hangul as NFD, which would
                # silently fail to match NFC mapping keys.
                self.df['printshop'] = self.df['printshop'].apply(
                    lambda s: unicodedata.normalize('NFC', s) if isinstance(s, str) else s
                )
                self.df['printshop'] = self.df['printshop'].replace(mapping)

    def _prepare_samples(self):
        for _, row in self.df.iterrows():
            image_path = os.path.join(self.base_dir, row['image_path'])
            class_idx = self.printshop_to_idx[row['printshop']]
            book_id = row.get('book_id', row.get('book', ''))

            path_hash = hashlib.md5(image_path.encode()).hexdigest()
            basename = os.path.splitext(os.path.basename(image_path))[0]
            feature_file = os.path.join(self.feature_dir, f"{basename}_{path_hash}.npy")

            self.samples.append((feature_file, class_idx, book_id))

    def __getitem__(self, idx):
        feat_path, label, _ = self.samples[idx]
        try:
            features = np.load(feat_path)
            features = torch.from_numpy(features).float()
        except Exception as e:
            logger.error(f"Error loading features {feat_path}: {e}")
            features = torch.zeros(49729, 256, dtype=torch.float32)
        return features, label

    def __len__(self):
        return len(self.samples)

    @property
    def class_names(self):
        return [self.idx_to_printshop[i] for i in range(len(self.idx_to_printshop))]

    @property
    def book_ids(self):
        return [s[2] for s in self.samples]

    @property
    def labels(self):
        return [s[1] for s in self.samples]


class FeaturePrintshopDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for pre-extracted features.
    Supports packed memmap (fast) or individual .npy files (fallback).
    """

    def __init__(self,
                 parquet_file: str,
                 feature_dir: str,
                 base_dir: str = "",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 fold: int = 0,
                 n_splits: int = 5,
                 random_state: int = 42,
                 fold_file: str = None,
                 feature_noise_std: float = 0.0,
                 feature_dropout: float = 0.0):
        super().__init__()
        self.parquet_file = parquet_file
        self.feature_dir = feature_dir
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold
        self.n_splits = n_splits
        self.random_state = random_state
        self.fold_file = fold_file
        self.feature_noise_std = feature_noise_std
        self.feature_dropout = feature_dropout

        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def _is_packed(self):
        """Check if feature_dir points to a packed memmap prefix."""
        return os.path.exists(f"{self.feature_dir}.dat") and \
               os.path.exists(f"{self.feature_dir}_index.json")

    def prepare_data(self):
        if not os.path.exists(self.parquet_file):
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_file}")
        if not self._is_packed() and not os.path.isdir(self.feature_dir):
            raise FileNotFoundError(
                f"Feature source not found. Expected either:\n"
                f"  Packed: {self.feature_dir}.dat + {self.feature_dir}_index.json\n"
                f"  Directory: {self.feature_dir}/")
        if self.fold_file and not os.path.exists(self.fold_file):
            logger.warning(f"Fold file not found: {self.fold_file}")
            self.fold_file = None

    def setup(self, stage: str = None):
        if self.full_dataset is None:
            if self._is_packed():
                logger.info("Using packed memmap features (fast mode)")
                self.full_dataset = PackedFeatureDataset(
                    parquet_file=self.parquet_file,
                    packed_prefix=self.feature_dir,
                    base_dir=self.base_dir,
                )
            else:
                logger.info("Using individual .npy files (slow mode)")
                self.full_dataset = FeaturePrintshopDataset(
                    parquet_file=self.parquet_file,
                    feature_dir=self.feature_dir,
                    base_dir=self.base_dir,
                )

            if self.fold_file and os.path.exists(self.fold_file):
                self._setup_from_fold_file()
            else:
                raise ValueError(
                    f"Fold file is required but not found: {self.fold_file}. "
                    f"Evaluation requires exact same splits as training."
                )

        if stage == 'fit' or stage is None and self.train_dataset is None:
            if self.fold < 0 or self.fold >= self.n_splits:
                raise ValueError(f"Fold {self.fold} out of range (0 to {self.n_splits-1})")

            if not hasattr(self, 'fold_assignments') or not self.fold_assignments:
                raise ValueError("Fold assignments not loaded.")

            fold_data = self.fold_assignments['folds'][self.fold]
            train_indices = fold_data['train_indices']
            val_indices = fold_data['val_indices']

            logger.info(f"Fold {self.fold}: {len(train_indices)} train, {len(val_indices)} val")

            self.train_dataset = _AugmentedFeatureSubset(
                self.full_dataset, train_indices,
                noise_std=self.feature_noise_std,
                dropout_p=self.feature_dropout,
                training=True,
            )
            self.val_dataset = Subset(self.full_dataset, val_indices)

            self._verify_split_integrity(train_indices, val_indices)

    def _setup_from_fold_file(self):
        logger.info(f"Loading fold assignments from {self.fold_file}")
        with open(self.fold_file, 'r') as f:
            self.fold_assignments = json.load(f)

        metadata = self.fold_assignments['metadata']
        if metadata['total_samples'] != len(self.full_dataset):
            logger.warning(
                f"Fold file has {metadata['total_samples']} samples, "
                f"dataset has {len(self.full_dataset)}."
            )
        logger.info(f"Loaded {len(self.fold_assignments['folds'])} folds")

    def _verify_split_integrity(self, train_indices, val_indices):
        train_set = set(train_indices)
        val_set = set(val_indices)
        overlap = train_set.intersection(val_set)
        if overlap:
            logger.warning(f"WARNING: {len(overlap)} samples in both train and val!")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def class_names(self):
        return self.full_dataset.class_names


class _AugmentedFeatureSubset(Dataset):
    """Subset wrapper that applies feature-space augmentation during training."""

    def __init__(self, dataset, indices, noise_std=0.0, dropout_p=0.0, training=True):
        self.dataset = dataset
        self.indices = indices
        self.noise_std = noise_std
        self.dropout_p = dropout_p
        self.training = training

    def __getitem__(self, idx):
        features, label = self.dataset[self.indices[idx]]

        if self.training:
            if self.noise_std > 0:
                noise = torch.randn_like(features) * self.noise_std
                features = features + noise

            if self.dropout_p > 0:
                mask = torch.bernoulli(torch.full_like(features, 1.0 - self.dropout_p))
                features = features * mask / (1.0 - self.dropout_p)

        return features, label

    def __len__(self):
        return len(self.indices)
