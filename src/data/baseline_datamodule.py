# src/data/baseline_datamodule.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KoreanPrintshopBaseDataset(Dataset):
    def __init__(self, data_dir, transform=None, samples_per_class=None):
        self.data_dir = os.path.expanduser(data_dir)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._load_dataset(samples_per_class)

    def _load_dataset(self, samples_per_class=None):
        logger.info(f"Loading dataset from directory: {self.data_dir}")
        class_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for class_idx, class_name in enumerate(sorted(class_dirs)):
            class_dir = os.path.join(self.data_dir, class_name)
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name

            valid_images = []
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    valid_images.append(img_path)

            if samples_per_class and len(valid_images) > samples_per_class:
                valid_images = random.sample(valid_images, samples_per_class)

            for img_path in valid_images:
                self.samples.append((img_path, class_idx))

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = np.array(img)
            
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']
            
        return img, label

    def __len__(self):
        return len(self.samples)

class KoreanPrintshopMultiCropDataset(Dataset):
    def __init__(self, data_dir, transform=None, samples_per_class=None, num_crops=4, crop_size=224):
        self.data_dir = os.path.expanduser(data_dir)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.num_crops = num_crops
        self.crop_size = crop_size
        self._load_dataset(samples_per_class)
        
        self.crop_transform = A.Compose([
            A.RandomCrop(width=self.crop_size, height=self.crop_size)
        ])

    def _load_dataset(self, samples_per_class=None):
        logger.info(f"Loading dataset from directory: {self.data_dir}")
        class_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for class_idx, class_name in enumerate(sorted(class_dirs)):
            class_dir = os.path.join(self.data_dir, class_name)
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name

            valid_images = []
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    valid_images.append(img_path)

            if samples_per_class and len(valid_images) > samples_per_class:
                valid_images = random.sample(valid_images, samples_per_class)

            for img_path in valid_images:
                self.samples.append((img_path, class_idx))

    def __getitem__(self, idx):
        img_path, label = self.samples[idx // self.num_crops]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = np.array(img)
            
            # Apply random crop
            cropped = self.crop_transform(image=img)
            img = cropped['image']
            
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']
            
        return img, label

    def __len__(self):
        return len(self.samples) * self.num_crops

class BaselineDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int = 32, 
        num_workers: int = 4,
        samples_per_class: int = None,
        use_random_crops: bool = False,
        num_crops: int = 4,
        crop_size: int = 224
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_per_class = samples_per_class
        self.use_random_crops = use_random_crops
        self.num_crops = num_crops
        self.crop_size = crop_size

        # Common transforms
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
        # Enhanced training transforms
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

    def setup(self, stage: str = None):
        # Choose dataset class based on whether we're using random crops
        if self.use_random_crops:
            dataset_class = KoreanPrintshopMultiCropDataset
            dataset_kwargs = {
                'num_crops': self.num_crops,
                'crop_size': self.crop_size
            }
        else:
            dataset_class = KoreanPrintshopBaseDataset
            dataset_kwargs = {}

        # Create dataset with appropriate arguments
        full_dataset = dataset_class(
            data_dir=self.data_dir,
            transform=self.val_transform,
            samples_per_class=self.samples_per_class,
            **dataset_kwargs
        )

        # Calculate splits
        total_len = len(full_dataset)
        train_size = int(0.7 * total_len)
        val_size = int(0.15 * total_len)
        test_size = total_len - train_size - val_size

        # Create splits
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Set transforms
        self.train_dataset.dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.val_transform
        self.test_dataset.dataset.transform = self.val_transform

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