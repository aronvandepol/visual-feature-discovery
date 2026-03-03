# Preprocessing Pipeline

This document details the image preprocessing and data augmentation pipeline used for training the classification models. The preprocessing is shared across CNN-based models (ConvNeXt, Swin) and the Multiple Instance Learning (MIL) approach, with minor variations in input resolution.

---

## 1. Overview

All models follow a similar preprocessing philosophy:
1. Convert historical document scans to **grayscale** (removes color artifacts, focuses on text/layout)
2. Apply **contrast enhancement** (CLAHE) to improve text visibility
3. **Resize** to model-specific input dimensions
4. **Normalize** using ImageNet statistics
5. Apply **augmentations** during training to improve generalization

---

## 2. Image Loading

### 2.1 Basic Loading Process

```python
from PIL import Image
import numpy as np

# Load image
with Image.open(img_path) as img:
    # Convert to RGB (handles grayscale/RGBA source images)
    img = img.convert('RGB')
    # Convert to numpy array for albumentations
    img = np.array(img)
```

### 2.2 Supported Formats
- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)

### 2.3 Image Validation
Images are validated before inclusion in the dataset:

```python
def _is_valid_image(self, img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False
```

---

## 3. Core Preprocessing Transforms

All models use [Albumentations](https://albumentations.ai/) for preprocessing. The core transforms applied to all images (training, validation, and test):

### 3.1 Grayscale Conversion

```python
A.ToGray(p=1.0)  # Always convert to grayscale
```

**Rationale**: Historical Korean documents are inherently grayscale. Converting to grayscale:
- Removes scanning artifacts and color variations
- Reduces input dimensionality
- Forces model to focus on structural/textural features

### 3.2 CLAHE (Contrast Limited Adaptive Histogram Equalization)

```python
A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
```

**Rationale**: Historical documents often have:
- Faded ink
- Uneven illumination
- Age-related degradation

CLAHE enhances local contrast while preventing over-amplification of noise, making text more readable for the model.

**Parameters**:
- `clip_limit=2.0`: Limits contrast amplification to prevent noise enhancement
- `tile_grid_size=(8, 8)`: Divides image into 8x8 tiles for local histogram equalization

### 3.3 Resizing

| Model Type | Input Resolution | Notes |
|------------|------------------|-------|
| CNN (ConvNeXt) | 224 x 224 | Full page resized |
| Swin (Cutout) | 224 x 224 | Random crop from page |
| MIL | 896 x 896 | Then extracted into patches |

```python
# CNN/Swin
A.Resize(224, 224)

# MIL
A.Resize(896, 896)  # self.resize parameter
```

### 3.4 Normalization

```python
# CNN/Swin (3-channel input expected by pretrained models)
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# MIL (single channel)
A.Normalize(mean=[0.485], std=[0.229])
```

**Note**: Despite converting to grayscale, CNN/Swin models maintain 3 channels (grayscale value replicated) to match pretrained model expectations.

---

## 4. Validation/Test Transform Pipeline

The minimal transform pipeline applied during evaluation:

### 4.1 CNN/Swin Models

```python
val_transform = A.Compose([
    A.Resize(224, 224),
    A.ToGray(p=1),
    A.CLAHE(clip_limit=2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

### 4.2 MIL Model

```python
val_test_transform = A.Compose([
    A.Resize(896, 896),
    A.ToGray(p=1.0),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Normalize(mean=[0.485], std=[0.229]),
    ToTensorV2(),
])
```

---

## 5. Training Augmentation Pipeline

Additional augmentations applied only during training to improve generalization:

### 5.1 CNN/Swin Training Augmentations

```python
train_transform = A.Compose([
    # Core preprocessing
    A.Resize(224, 224),
    A.ToGray(p=1),

    # Geometric augmentations
    A.RandomRotate90(),                    # 90° rotation variants
    A.Flip(),                              # Horizontal/vertical flips
    A.Transpose(p=0.1),                    # Diagonal flip
    A.ShiftScaleRotate(
        shift_limit=0.0625,                # Small translations (±6.25%)
        scale_limit=0.2,                   # Scale ±20%
        rotate_limit=15,                   # Rotation ±15°
        p=0.3
    ),

    # Distortion augmentations
    A.OneOf([
        A.OpticalDistortion(p=0.3),        # Lens-like distortion
        A.GridDistortion(p=0.1),           # Grid-based warping
    ], p=0.2),

    # Intensity augmentations
    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0,
            contrast_limit=1,              # Strong contrast variation
            p=1
        ),
        A.CLAHE(clip_limit=2),             # Alternative contrast enhancement
    ], p=0.3),
    A.RandomBrightnessContrast(p=0.1),     # Mild brightness/contrast

    # Noise/dropout
    A.PixelDropout(0.01),                  # Random pixel dropout (1%)

    # Normalization
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

### 5.2 MIL Training Augmentations

```python
train_transform = A.Compose([
    # Core preprocessing
    A.Resize(896, 896),
    A.ToGray(p=1.0),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),

    # Blur augmentation
    A.GaussianBlur(blur_limit=(3, 3), p=0.2),

    # Intensity augmentations
    A.RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.5
    ),

    # Noise augmentations
    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),

    # Compression artifacts (simulates JPEG degradation)
    A.ImageCompression(quality_lower=85, quality_upper=100, p=0.5),

    # Sharpening
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),

    # Normalization
    A.Normalize(mean=[0.485], std=[0.229]),
    ToTensorV2(),
])
```

---

## 6. Augmentation Rationale

### 6.1 Geometric Augmentations

| Augmentation | Purpose |
|--------------|---------|
| RandomRotate90 | Documents may be scanned at different orientations |
| Flip | Increases dataset size; text patterns remain recognizable |
| ShiftScaleRotate | Simulates slight scanning misalignments |
| Transpose | Additional orientation variation |

### 6.2 Distortion Augmentations

| Augmentation | Purpose |
|--------------|---------|
| OpticalDistortion | Simulates lens distortion from scanning equipment |
| GridDistortion | Simulates page warping/curvature |

### 6.3 Intensity Augmentations

| Augmentation | Purpose |
|--------------|---------|
| RandomBrightnessContrast | Simulates varying scan quality/lighting |
| CLAHE | Alternative contrast enhancement method |

### 6.4 Noise/Degradation Augmentations

| Augmentation | Purpose |
|--------------|---------|
| PixelDropout | Simulates dust/damage on pages |
| GaussNoise | Simulates sensor noise |
| GaussianBlur | Simulates focus issues |
| ImageCompression | Simulates JPEG artifacts from digitization |
| Sharpen | Counteracts blur, enhances edges |

---

## 7. Cutout/Multi-Crop Approach

For the Swin model using the cutout approach, an additional random cropping step is applied:

### 7.1 Random Crop Transform

```python
crop_transform = A.Compose([
    A.RandomCrop(width=224, height=224)
])
```

### 7.2 Multi-Crop Dataset

Each image generates multiple crops during training:

```python
class KoreanPrintshopMultiCropDataset(Dataset):
    def __init__(self, ..., num_crops=4, crop_size=224):
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.crop_transform = A.Compose([
            A.RandomCrop(width=crop_size, height=crop_size)
        ])

    def __getitem__(self, idx):
        # Get base image index (each image yields num_crops samples)
        img_path, label = self.samples[idx // self.num_crops]

        # Load and crop
        img = load_image(img_path)
        cropped = self.crop_transform(image=img)
        img = cropped['image']

        # Apply standard transforms
        if self.transform:
            img = self.transform(image=img)['image']

        return img, label

    def __len__(self):
        return len(self.samples) * self.num_crops
```

**Parameters**:
- `num_crops=4`: Each page generates 4 random crops per epoch
- `crop_size=224`: Crop dimensions match model input size

---

## 8. MIL Patch Extraction

For the MIL approach, images are divided into patches after preprocessing:

### 8.1 Patch Parameters

The paper reports results for two patch configurations:

```python
# Configuration 1: 16x16 px patches (best performance)
patch_size = 16   # Each patch is 16x16 pixels
stride = 8        # 50% overlap between patches
resize = 896      # Image resized before patch extraction

# Configuration 2: 8x8 px patches
patch_size = 8
stride = 4
resize = 896
```

### 8.2 Patch Extraction Code

```python
def _extract_patches(self, img):
    # img shape: (C, H, W) = (1, 896, 896)
    c, h, w = img.shape

    # Use unfold to extract patches with overlap
    patches = img.unfold(1, self.patch_size, self.stride)  # (C, n_h, W, patch_size)
    patches = patches.unfold(2, self.patch_size, self.stride)  # (C, n_h, n_w, patch_size, patch_size)

    # Rearrange to (n_h, n_w, C, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()

    return patches
```

### 8.3 Patch Grid Dimensions

With `resize=896`, `patch_size=16`, `stride=8`:
- Number of patches per dimension: `(896 - 16) / 8 + 1 = 111`
- Total patches per image: `111 x 111 = 12,321` patches
- Each patch: `16 x 16 x 1` (grayscale)

With `resize=896`, `patch_size=8`, `stride=4`:
- Number of patches per dimension: `(896 - 8) / 4 + 1 = 223`
- Total patches per image: `223 x 223 = 49,729` patches
- Each patch: `8 x 8 x 1` (grayscale)

---

## 9. Transform Pipeline Summary

### 9.1 Visual Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW IMAGE                                │
│                   (Variable resolution)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LOAD & CONVERT                              │
│              PIL.Image → RGB → NumPy array                      │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    ┌───────────┐       ┌───────────┐       ┌───────────┐
    │  ConvNeXt │       │    Swin   │       │    MIL    │
    │ Full Page │       │  Cutout   │       │  Patches  │
    └───────────┘       └───────────┘       └───────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    ┌───────────┐       ┌───────────┐       ┌───────────┐
    │  Resize   │       │  Random   │       │  Resize   │
    │  224×224  │       │   Crop    │       │  896×896  │
    └───────────┘       │  224×224  │       └───────────┘
          │             └───────────┘             │
          │                   │                   │
          └─────────┬─────────┘                   │
                    │                             │
                    ▼                             ▼
          ┌─────────────────┐           ┌─────────────────┐
          │    ToGray       │           │    ToGray       │
          │   (p=1.0)       │           │   (p=1.0)       │
          └─────────────────┘           └─────────────────┘
                    │                             │
                    ▼                             ▼
          ┌─────────────────┐           ┌─────────────────┐
          │     CLAHE       │           │     CLAHE       │
          │  (clip_limit=2) │           │  (clip_limit=2) │
          └─────────────────┘           └─────────────────┘
                    │                             │
                    ▼                             │
          ┌─────────────────┐                     │
          │  AUGMENTATIONS  │                     │
          │ (training only) │                     │
          └─────────────────┘                     │
                    │                             │
                    ▼                             ▼
          ┌─────────────────┐           ┌─────────────────┐
          │   Normalize     │           │   Normalize     │
          │ (ImageNet stats)│           │ (ImageNet stats)│
          └─────────────────┘           └─────────────────┘
                    │                             │
                    ▼                             ▼
          ┌─────────────────┐           ┌─────────────────┐
          │   ToTensorV2    │           │   ToTensorV2    │
          │  → PyTorch      │           │  → PyTorch      │
          └─────────────────┘           └─────────────────┘
                    │                             │
                    ▼                             ▼
          ┌─────────────────┐           ┌─────────────────┐
          │  MODEL INPUT    │           │ PATCH EXTRACT   │
          │   (1,3,224,224) │           │ (49729,1,8,8)   │
          └─────────────────┘           └─────────────────┘
```

### 9.2 Configuration Summary Table

| Parameter | ConvNeXt | Swin (Cutout) | MIL (16x16) | MIL (8x8) |
|-----------|----------|---------------|-------------|-----------|
| Input Resolution | 224x224 | 224x224 | 896x896 | 896x896 |
| Grayscale | Yes | Yes | Yes | Yes |
| CLAHE | clip_limit=2 | clip_limit=2 | clip_limit=2, 8x8 grid | clip_limit=2, 8x8 grid |
| Normalization | ImageNet (3ch) | ImageNet (3ch) | ImageNet (1ch) | ImageNet (1ch) |
| Patch Size | N/A | N/A | 16x16 | 8x8 |
| Patch Stride | N/A | N/A | 8 | 4 |
| Patches/Image | 1 | 1 | 12,321 | 49,729 |
| Crops/Image | 1 | 4 | 1 | 1 |

---

## 10. Implementation Notes

### 10.1 Library Versions
- **Albumentations**: 1.4.18+
- **PyTorch**: 2.0+
- **PIL/Pillow**: 9.0+

### 10.2 Memory Considerations
- MIL patch extraction increases memory usage significantly (49,729 patches per image at 8x8)
- Multi-crop approach multiplies effective dataset size by `num_crops`
- CLAHE is computationally intensive; applied after resize to reduce cost

### 10.3 Reproducibility
- Random seed set to 42 for data splits
- Augmentations use per-call randomization (not seeded)
- Transform order is deterministic

---

## 11. References

- Albumentations Documentation: https://albumentations.ai/docs/
- CLAHE: Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"
- ImageNet Normalization: Standard values from torchvision pretrained models

---

*Document prepared: February 2026*
