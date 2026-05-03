#!/usr/bin/env python3
"""
Train CNN/Swin models with page-level splits (DATA LEAKAGE BASELINE).

Uses standard KFold which splits at page level - pages from the same book
CAN appear in both train and test. This replicates the original (flawed)
evaluation methodology for comparison with book-level splits.

Each fold can be run as a separate SLURM array job.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

try:
    from pytorch_lightning.loggers import CometLogger
except ImportError:
    CometLogger = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.lightning_timm import LightningTIMM  # noqa: E402


def extract_book(filename: str) -> str:
    """Extract book name from filename pattern: class/bookname-pagenum.jpg"""
    parts = filename.split('/')
    if len(parts) >= 2:
        book_part = parts[1]
        match = re.match(r'(.+)-\d+\.jpg', book_part, re.IGNORECASE)
        if match:
            return match.group(1)
    return filename


class PageSplitDataset(Dataset):
    """Dataset that tracks book information for leakage analysis."""

    def __init__(self, data_dir: str, transform=None, use_random_crops: bool = False,
                 num_crops: int = 4, crop_size: int = 224):
        self.data_dir = Path(data_dir).expanduser()
        self.transform = transform
        self.use_random_crops = use_random_crops
        self.num_crops = num_crops
        self.crop_size = crop_size

        self.samples = []
        self.books = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Load all samples
        class_dirs = sorted([d for d in os.listdir(self.data_dir)
                            if os.path.isdir(self.data_dir / d)])

        for class_idx, class_name in enumerate(class_dirs):
            class_dir = self.data_dir / class_name
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = str(class_dir / img_name)
                    filename = f"{class_name}/{img_name}"
                    book = extract_book(filename)

                    self.samples.append({
                        'path': img_path,
                        'class_idx': class_idx,
                        'book': book
                    })
                    self.books.append(book)

        self.books = np.array(self.books)

        if use_random_crops:
            self.crop_transform = A.Compose([
                A.RandomCrop(width=crop_size, height=crop_size)
            ])

    def __len__(self):
        if self.use_random_crops:
            return len(self.samples) * self.num_crops
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_random_crops:
            sample_idx = idx // self.num_crops
        else:
            sample_idx = idx

        sample = self.samples[sample_idx]

        with Image.open(sample['path']) as img:
            img = img.convert('RGB')
            img = np.array(img)

            if self.use_random_crops:
                cropped = self.crop_transform(image=img)
                img = cropped['image']

            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']

        return img, sample['class_idx']


class IndexedSubset(Dataset):
    """Subset that properly handles multicrop indexing."""

    def __init__(self, dataset, indices, use_random_crops=False, num_crops=4):
        self.dataset = dataset
        self.indices = indices
        self.use_random_crops = use_random_crops
        self.num_crops = num_crops

    def __len__(self):
        if self.use_random_crops:
            return len(self.indices) * self.num_crops
        return len(self.indices)

    def __getitem__(self, idx):
        if self.use_random_crops:
            sample_idx = idx // self.num_crops
            real_idx = self.indices[sample_idx]
            sample = self.dataset.samples[real_idx]

            with Image.open(sample['path']) as img:
                img = img.convert('RGB')
                img = np.array(img)

                cropped = self.dataset.crop_transform(image=img)
                img = cropped['image']

                if self.dataset.transform:
                    transformed = self.dataset.transform(image=img)
                    img = transformed['image']

            return img, sample['class_idx']
        else:
            real_idx = self.indices[idx]
            sample = self.dataset.samples[real_idx]

            with Image.open(sample['path']) as img:
                img = img.convert('RGB')
                img = np.array(img)

                if self.dataset.transform:
                    transformed = self.dataset.transform(image=img)
                    img = transformed['image']

            return img, sample['class_idx']


class PageSplitDataModule(pl.LightningDataModule):
    """DataModule with page-level KFold splitting (DATA LEAKAGE)."""

    def __init__(
        self,
        data_dir: str,
        fold: int,
        n_folds: int = 5,
        batch_size: int = 32,
        num_workers: int = 4,
        use_random_crops: bool = False,
        num_crops: int = 4,
        crop_size: int = 224,
        val_split: float = 0.15
    ):
        super().__init__()
        self.data_dir = data_dir
        self.fold = fold
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_random_crops = use_random_crops
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.val_split = val_split

        # Training transforms
        self.train_transform = A.Compose([
            A.Resize(224, 224),
            A.ToGray(p=1),
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(p=0.1),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.3
            ),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0,
                    contrast_limit=1,
                    p=1
                ),
                A.CLAHE(clip_limit=2),
            ], p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.PixelDropout(0.01),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # Validation/Test transforms
        self.val_transform = A.Compose([
            A.Resize(224, 224),
            A.ToGray(p=1),
            A.CLAHE(clip_limit=2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        self.split_info = None

    def setup(self, stage: str = None):
        # Create base dataset
        full_dataset = PageSplitDataset(
            data_dir=self.data_dir,
            transform=self.val_transform,
            use_random_crops=self.use_random_crops,
            num_crops=self.num_crops,
            crop_size=self.crop_size
        )

        # Get page-level split using KFold (NO book grouping = DATA LEAKAGE)
        books = full_dataset.books
        labels = np.array([s['class_idx'] for s in full_dataset.samples])
        indices = np.arange(len(full_dataset.samples))

        # Standard KFold - pages split randomly regardless of book
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for i, (train_val_idx, test_idx) in enumerate(kf.split(indices, labels)):
            if i == self.fold:
                # Further split train_val into train and val (also page-level)
                n_train_val = len(train_val_idx)
                n_val = int(n_train_val * self.val_split)

                # Shuffle and split
                rng = np.random.default_rng(42 + self.fold)
                rng.shuffle(train_val_idx)

                val_idx = train_val_idx[:n_val]
                train_idx = train_val_idx[n_val:]

                # Analyze book overlap (to confirm data leakage)
                train_books = set(books[train_idx])
                val_books = set(books[val_idx])
                test_books = set(books[test_idx])

                self.split_info = {
                    'fold': self.fold,
                    'split_type': 'page_level',
                    'train_samples': len(train_idx),
                    'val_samples': len(val_idx),
                    'test_samples': len(test_idx),
                    'train_books': len(train_books),
                    'val_books': len(val_books),
                    'test_books': len(test_books),
                    'total_unique_books': len(set(books)),
                    # Data leakage metrics
                    'train_val_book_overlap': len(train_books & val_books),
                    'train_test_book_overlap': len(train_books & test_books),
                    'val_test_book_overlap': len(val_books & test_books),
                    'train_val_overlap_pct': len(train_books & val_books) / len(val_books) * 100 if len(val_books) > 0 else 0,
                    'train_test_overlap_pct': len(train_books & test_books) / len(test_books) * 100 if len(test_books) > 0 else 0,
                }

                break

        # Create subset datasets
        self.train_dataset = IndexedSubset(
            full_dataset, train_idx.tolist(),
            use_random_crops=self.use_random_crops,
            num_crops=self.num_crops
        )
        self.val_dataset = IndexedSubset(
            full_dataset, val_idx.tolist(),
            use_random_crops=self.use_random_crops,
            num_crops=self.num_crops
        )
        self.test_dataset = IndexedSubset(
            full_dataset, test_idx.tolist(),
            use_random_crops=self.use_random_crops,
            num_crops=self.num_crops
        )

        # Store transforms
        full_dataset.transform = self.val_transform

        # Create separate dataset instance for train with train_transform
        self.train_full_dataset = PageSplitDataset(
            data_dir=self.data_dir,
            transform=self.train_transform,
            use_random_crops=self.use_random_crops,
            num_crops=self.num_crops,
            crop_size=self.crop_size
        )
        self.train_dataset = IndexedSubset(
            self.train_full_dataset, train_idx.tolist(),
            use_random_crops=self.use_random_crops,
            num_crops=self.num_crops
        )

        # Val and test use val_transform
        self.val_full_dataset = full_dataset
        self.test_full_dataset = full_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def main(args):
    comet_api_key = os.environ.get("COMET_API_KEY")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n{'='*60}")
    print(f"Training {args.model_name} - Fold {args.fold + 1}/{args.n_folds}")
    print("PAGE-LEVEL split (DATA LEAKAGE BASELINE)")
    print(f"{'='*60}\n")

    # Set up data module with page-level splits
    data_module = PageSplitDataModule(
        data_dir=args.data_dir,
        fold=args.fold,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_random_crops=args.use_random_crops,
        num_crops=args.num_crops,
        crop_size=args.crop_size
    )

    # Call setup to get split info
    data_module.setup()

    print("Split Statistics (PAGE-LEVEL - DATA LEAKAGE):")
    for k, v in data_module.split_info.items():
        print(f"  {k}: {v}")
    print()

    # Warn about data leakage
    overlap = data_module.split_info['train_test_book_overlap']
    overlap_pct = data_module.split_info['train_test_overlap_pct']
    print(f"WARNING: DATA LEAKAGE: {overlap} books ({overlap_pct:.1f}%) appear in BOTH train and test!")
    print()

    # Initialize model
    model = LightningTIMM(
        num_classes=4,
        model_name=args.model_name,
        learning_rate=args.learning_rate
    )

    # Set up logger
    exp_name = f"{args.model_name}_page_fold{args.fold}_{timestamp}"
    if comet_api_key and CometLogger is not None:
        logger = CometLogger(
            api_key=comet_api_key,
            project_name=args.comet_project,
        )
        logger.experiment.set_name(exp_name)
        logger.experiment.add_tags([
            "page_split",
            "data_leakage_baseline",
            f"fold_{args.fold}",
            f"{args.n_folds}fold_cv",
            "training",
            args.model_name
        ])
        logger.experiment.log_parameters({
            "model_name": args.model_name,
            "fold": args.fold,
            "n_folds": args.n_folds,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "use_random_crops": args.use_random_crops,
            "num_crops": args.num_crops if args.use_random_crops else None,
            "crop_size": args.crop_size,
            "split_mode": "page_level_kfold",
            **data_module.split_info
        })
    else:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger("logs", name=exp_name)
        print("No COMET_API_KEY found. Using CSVLogger. Set COMET_API_KEY for Comet ML tracking.")

    # Callbacks
    checkpoint_dir = f'checkpoints_page_split/{args.model_name}_fold{args.fold}_{timestamp}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{args.model_name}-fold{args.fold}-{{epoch:02d}}-{{val_loss:.3f}}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=100,
        accumulate_grad_batches=2,
    )

    # Train
    trainer.fit(model, data_module)

    # Test
    test_results = trainer.test(model, data_module)

    # Save final model
    model_save_path = f'models_page_split/{args.model_name}_fold{args.fold}_{timestamp}.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.model.state_dict(), model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    # Log model to Comet
    logger.experiment.log_model(f'model_fold{args.fold}', model_save_path)

    # Save results JSON
    results = {
        'model_name': args.model_name,
        'fold': args.fold,
        'n_folds': args.n_folds,
        'split_info': data_module.split_info,
        'test_results': test_results[0] if test_results else None,
        'model_path': model_save_path,
        'checkpoint_dir': checkpoint_dir
    }

    results_path = f'results_page_split/{args.model_name}_fold{args.fold}_{timestamp}.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    print(f"\nComet experiment: {logger.experiment.url}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with page-level splits (DATA LEAKAGE BASELINE)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='TIMM model name (e.g., convnextv2_base, swin_s3_base_224)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--fold', type=int, required=True,
                        help='Fold number (0-indexed)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Total number of folds')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Data loading workers')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum epochs')
    parser.add_argument('--use_random_crops', action='store_true',
                        help='Use random crops (for Swin multicrop)')
    parser.add_argument('--num_crops', type=int, default=4,
                        help='Number of crops per image')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Crop size')
    parser.add_argument('--comet_project', type=str, default='visual-feature-discovery-page-split',
                        help='Comet ML project name')

    args = parser.parse_args()
    main(args)
