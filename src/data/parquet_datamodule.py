import os
import unicodedata
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

# Load the NotoSansCJK.ttc font for Korean text
noto_font_path = "NotoSansCJK.ttc"
if os.path.exists(noto_font_path):
    try:
        fm.fontManager.addfont(noto_font_path)
    except Exception:
        pass
import json
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import pytorch_lightning as pl
import logging
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# Import PATCH_SIZE from config
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from feature_extraction.config import PATCH_SIZE
except ImportError:
    PATCH_SIZE = 64  # Fallback default

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParquetPrintshopDataset(Dataset):
    def __init__(self, parquet_file, base_dir="", transform=None, patch_size=None, stride=None):
        """
        Dataset for loading Korean printshop data from a parquet file
        
        Args:
            parquet_file (str): Path to the parquet file
            base_dir (str): Base directory for image paths if they are relative in the parquet file
            transform: Image transformations to apply
            patch_size (int): Size of image patches
            stride (int): Stride for patch extraction
        """
        self.df = pd.read_parquet(parquet_file)

        # Debug: Check if dataframe loaded correctly
        logger.info(f"Loaded parquet file with shape: {self.df.shape}")
        logger.info(f"Columns: {list(self.df.columns)}")
        logger.info(f"Printshop column exists: {'printshop' in self.df.columns}")

        # Optional: Canonicalize/merge printshop names to control class count
        # Enable by setting CANONICALIZE_PRINTSHOPS=1 and optionally provide PRINTSHOP_MAP_JSON
        # Enable by default to merge Daedong variants into one class
        canon_flag = os.getenv("CANONICALIZE_PRINTSHOPS", "1")
        mapping_path = os.getenv("PRINTSHOP_MAP_JSON", "").strip()
        mapping = {}
        if canon_flag == "1":
            # Load mapping from JSON if provided; otherwise use a sensible default merge
            if mapping_path and os.path.exists(mapping_path):
                try:
                    import json
                    with open(mapping_path, 'r') as f:
                        mapping = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load PRINTSHOP_MAP_JSON '{mapping_path}': {e}")
                    mapping = {}
            if not mapping:
                # Default canonicalization: merge Daedong variants into a single label '대동인쇄소'
                mapping = {
                    '대동인쇄 주식회사 大東印刷 株式會社': '대동인쇄소',
                    '대동인쇄소 大東印刷所': '대동인쇄소',
                }

            # Check if printshop column exists and has data
            if 'printshop' in self.df.columns:
                logger.info(f"Printshop column info: dtype={self.df['printshop'].dtype}, nulls={self.df['printshop'].isnull().sum()}")
                # Normalize Unicode form before applying the mapping. Some parquet writers emit
                # decomposed Hangul (NFD), which would silently fail to match NFC mapping keys.
                self.df['printshop'] = self.df['printshop'].apply(
                    lambda s: unicodedata.normalize('NFC', s) if isinstance(s, str) else s
                )
                before = sorted(self.df['printshop'].dropna().unique())
                self.df['printshop'] = self.df['printshop'].replace(mapping)
                after = sorted(self.df['printshop'].dropna().unique())
                if before != after:
                    logger.info(f"Canonicalized printshops: {len(before)} -> {len(after)} classes")
            else:
                logger.warning("Printshop column not found in dataframe!")
        self.base_dir = os.path.expanduser(base_dir)
        self.transform = transform
        self.patch_size = patch_size if patch_size is not None else PATCH_SIZE
        self.stride = stride if stride is not None else (PATCH_SIZE // 2)
        
        # Extract unique printshops and create mapping
        self.unique_printshops = self.df['printshop'].unique()
        self.printshop_to_idx = {printshop: idx for idx, printshop in enumerate(sorted(self.unique_printshops))}
        self.idx_to_printshop = {idx: printshop for printshop, idx in self.printshop_to_idx.items()}
        
        # List of tuples (image_path, class_idx, book_id) for each sample
        self.samples = []
        self._prepare_samples()
        
        logger.info(f"Loaded dataset with {len(self.samples)} images across {len(self.unique_printshops)} printshop classes.")
    
    def _prepare_samples(self):
        for idx, row in self.df.iterrows():
            image_path = os.path.join(self.base_dir, row['image_path'])
            class_idx = self.printshop_to_idx[row['printshop']]
            book = row['book']
            book_id = row['book_id'] if 'book_id' in row else book  # Use book_id if available, otherwise use book name
            
            self.samples.append((image_path, class_idx, book_id))
    
    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img = np.array(img)
                
                if self.transform:
                    transformed = self.transform(image=img)
                    img = transformed['image']
                
                # Handle different channel counts
                if len(img.shape) == 3:  # CHW format
                    c, h, w = img.shape
                    if c > 3:
                        # For images with >3 channels, take only first 3 channels (RGB)
                        img = img[:3]
                    elif c == 1:
                        # Repeat single channel across RGB dimensions
                        img = img.repeat(3, 1, 1)
                elif len(img.shape) == 2:  # HW format (grayscale)
                    img = img.unsqueeze(0).repeat(3, 1, 1)

                # Convert to PyTorch tensor and make it float
                if isinstance(img, torch.Tensor):
                    img = img.float()
                else:
                    img = torch.from_numpy(img).float()
                patches = self._extract_patches(img)
                return patches, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a dummy sample - you might want to handle this differently
            dummy_img = torch.zeros((3, 896, 896), dtype=torch.float32)
            dummy_patches = self._extract_patches(dummy_img)
            return dummy_patches, label
    
    def _extract_patches(self, img):
        # img should be a tensor with shape (3, H, W)
        c, h, w = img.shape
        patches = img.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        return patches
    
    def __len__(self):
        return len(self.samples)
    
    @property
    def class_names(self):
        return [self.idx_to_printshop[i] for i in range(len(self.idx_to_printshop))]
    
    @property
    def book_ids(self):
        """Return a list of book_ids for all samples"""
        return [sample[2] for sample in self.samples]
    
    @property
    def labels(self):
        """Return a list of class labels for all samples"""
        return [sample[1] for sample in self.samples]

class ParquetPrintshopDataModule(pl.LightningDataModule):
    def __init__(self, 
                 parquet_file: str, 
                 base_dir: str = "", 
                 batch_size: int = 32, 
                 num_workers: int = 4, 
                 patch_size: int = 64, 
                 stride: int = 32,
                 fold: int = 0,
                 n_splits: int = 5,
                 random_state: int = 42,
                 fold_file: str = None):
        """
        PyTorch Lightning DataModule for Korean Printshop dataset loaded from parquet file
        with standard k-fold cross-validation
        
        Args:
            parquet_file (str): Path to the parquet file
            base_dir (str): Base directory for image paths if they are relative
            batch_size (int): Batch size for training
            num_workers (int): Number of workers for data loading
            patch_size (int): Size of image patches
            stride (int): Stride for patch extraction
            fold (int): Which fold to use for validation (0 to n_splits-1)
            n_splits (int): Number of folds for cross-validation
            random_state (int): Random seed for reproducibility
            fold_file (str): Optional path to a pre-generated fold assignments file
        """
        super().__init__()
        self.parquet_file = parquet_file
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.stride = stride
        self.fold = fold
        self.n_splits = n_splits
        # Removed test_size - using standard k-fold
        self.random_state = random_state
        self.fold_file = fold_file
        
        self.train_transform = A.Compose([
            A.Resize(896, 896),
            A.ToGray(p=1.0),  # Convert images to grayscale
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(std_range=(0.0125, 0.028), mean_range=(0, 0), p=0.3),
            A.ImageCompression(quality_range=(85, 100), p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2(),
        ])

        self.val_test_transform = A.Compose([
            A.Resize(896, 896),
            A.ToGray(p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2(),
        ])
        
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
    
    def prepare_data(self):
        # Check if parquet file exists
        if not os.path.exists(self.parquet_file):
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_file}")
        
        # Check if fold file exists if specified
        if self.fold_file and not os.path.exists(self.fold_file):
            logger.warning(f"Fold file not found: {self.fold_file}. Will generate folds dynamically.")
            self.fold_file = None
    
    def setup(self, stage: str = None):
        if self.full_dataset is None:
            self.full_dataset = ParquetPrintshopDataset(
                parquet_file=self.parquet_file,
                base_dir=self.base_dir,
                transform=self.val_test_transform,
                patch_size=self.patch_size,
                stride=self.stride
            )
            
            if self.fold_file and os.path.exists(self.fold_file):
                # Load pre-generated fold assignments
                self._setup_from_fold_file()
            else:
                # REQUIRE fold file - no fallback allowed
                raise ValueError(f"Fold file is required but not found or not provided: {self.fold_file}. "
                               f"Evaluation requires exact same splits as training.")
            
            # No separate test dataset in standard k-fold
        
        if stage == 'fit' or stage is None and self.train_dataset is None:
            # Create train and validation datasets
            if self.fold < 0 or self.fold >= self.n_splits:
                raise ValueError(f"Fold index {self.fold} is out of range (0 to {self.n_splits-1})")
            
            # REQUIRE fold assignments to be loaded - no fallback
            if not hasattr(self, 'fold_assignments') or not self.fold_assignments:
                raise ValueError(f"Fold assignments not loaded. This should not happen if fold file was provided.")
            
            if self.fold >= len(self.fold_assignments['folds']):
                raise ValueError(f"Fold {self.fold} not found in fold assignments. Available folds: 0-{len(self.fold_assignments['folds'])-1}")
            
            fold_data = self.fold_assignments['folds'][self.fold]
            train_indices = fold_data['train_indices']
            val_indices = fold_data['val_indices']
            
            logger.info(f"Using fold file splits - Fold {self.fold}: {len(train_indices)} train, {len(val_indices)} val")
            
            # Create train and validation datasets
            self.train_dataset = Subset(self.full_dataset, train_indices)
            self.val_dataset = Subset(self.full_dataset, val_indices)
            
            # Set the transforms
            self.train_dataset.dataset.transform = self.train_transform
            self.val_dataset.dataset.transform = self.val_test_transform
            
            logger.info(f"Fold {self.fold}: Train set size = {len(self.train_dataset)}, "
                       f"Validation set size = {len(self.val_dataset)}")
            
            # Verify no overlap between splits
            self._verify_split_integrity(train_indices, val_indices)
    
    def _setup_from_fold_file(self):
        """Load fold assignments from a pre-generated file"""
        logger.info(f"Loading fold assignments from {self.fold_file}")
        with open(self.fold_file, 'r') as f:
            self.fold_assignments = json.load(f)
        
        # Verify the fold file is compatible with our dataset
        metadata = self.fold_assignments['metadata']
        if metadata['total_samples'] != len(self.full_dataset):
            logger.warning(f"Fold file was generated for a dataset with {metadata['total_samples']} samples, "
                         f"but current dataset has {len(self.full_dataset)} samples. This may cause issues.")
        
        logger.info(f"Loaded fold assignments with {len(self.fold_assignments['folds'])} folds")
    
    def _setup_dynamic_folds(self):
        """Generate folds dynamically - no longer needed for standard k-fold"""
        # Standard k-fold handles splits directly in setup()
        pass
    
    def _verify_split_integrity(self, train_indices, val_indices):
        """Verify there's no overlap between train and validation sets"""
        train_set = set(train_indices)
        val_set = set(val_indices)
        
        # Check train vs val
        train_val_overlap = train_set.intersection(val_set)
        if train_val_overlap:
            logger.warning(f"WARNING: Found {len(train_val_overlap)} samples in both train and validation sets!")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, collate_fn=self.custom_collate)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, collate_fn=self.custom_collate)
    
    # Removed test_dataloader - using standard k-fold validation only
    
    @staticmethod
    def custom_collate(batch):
        patches, labels = zip(*batch)
        stacked_patches = torch.stack(patches)
        stacked_labels = torch.tensor(labels)
        return stacked_patches, stacked_labels
    
    @property
    def class_names(self):
        return self.full_dataset.class_names
    
    def visualize_patches(self, num_samples=10):
        figures = []
        dataset = self.train_dataset if self.train_dataset else self.full_dataset
        
        for i in range(min(num_samples, len(dataset))):
            idx = torch.randint(len(dataset), (1,)).item()
            patches, label = dataset[idx]

            num_patches_h, num_patches_w = patches.shape[:2]
            grid = make_grid(patches.view(-1, *patches.shape[2:]), nrow=num_patches_w)

            grid_np = grid.permute(1, 2, 0).numpy()
            grid_np = (grid_np * np.array([0.229])) + np.array([0.485])
            grid_np = np.clip(grid_np, 0, 1)

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(grid_np, cmap='gray')
            ax.axis('off')
            
            # Get class name
            if isinstance(dataset, Subset):
                class_idx = label
                class_name = dataset.dataset.idx_to_printshop.get(class_idx, f"Unknown (ID: {class_idx})")
            else:
                class_name = dataset.idx_to_printshop.get(label, f"Unknown (ID: {label})")
                
            ax.set_title(f'Sample {i} Patches (Label: {class_name})')

            figures.append(fig)

        return figures
