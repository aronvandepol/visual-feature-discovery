#!/usr/bin/env python3
"""
Generate GradCAM visualizations for book-split trained models.

Samples from the TEST set which contains books never seen during training,
giving a true picture of what the model attends to for generalization.
"""

import os
import re
import argparse
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

# Set up Korean font (optional - gracefully skip if not available)
font_path = os.path.expanduser('~/.font/NanumGothic-Regular.ttf')
if os.path.exists(font_path):
    try:
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['font.sans-serif'] = ['NanumGothic'] + plt.rcParams['font.sans-serif']
    except Exception:
        pass  # Font loading failed, use default fonts


def extract_book(filename: str) -> str:
    """Extract book name from filename pattern: class/bookname-pagenum.jpg"""
    parts = filename.split('/')
    if len(parts) >= 2:
        book_part = parts[1]
        match = re.match(r'(.+)-\d+\.jpg', book_part, re.IGNORECASE)
        if match:
            return match.group(1)
    return filename


class BookSplitTestDataset(Dataset):
    """Dataset that returns test samples from a specific book-split fold."""

    def __init__(self, data_dir: str, fold: int, n_folds: int = 5, transform=None,
                 use_random_crops: bool = False, crop_size: int = 224):
        self.data_dir = Path(data_dir).expanduser()
        self.transform = transform
        self.use_random_crops = use_random_crops
        self.crop_size = crop_size

        # Load all samples
        self.all_samples = []
        self.all_books = []
        self.class_to_idx = {}
        self.idx_to_class = {}

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

                    self.all_samples.append({
                        'path': img_path,
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'book': book,
                        'filename': img_name
                    })
                    self.all_books.append(book)

        self.all_books = np.array(self.all_books)

        # Get test indices for this fold using GroupKFold
        labels = np.array([s['class_idx'] for s in self.all_samples])
        indices = np.arange(len(self.all_samples))

        gkf = GroupKFold(n_splits=n_folds)
        for i, (train_idx, test_idx) in enumerate(gkf.split(indices, labels, groups=self.all_books)):
            if i == fold:
                self.test_indices = test_idx
                break

        # Get test samples
        self.samples = [self.all_samples[i] for i in self.test_indices]

        # Get unique test books
        test_books = set(self.all_books[i] for i in self.test_indices)
        print(f"Test set: {len(self.samples)} samples from {len(test_books)} unseen books")

        if use_random_crops:
            self.crop_transform = A.Compose([
                A.RandomCrop(width=crop_size, height=crop_size)
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with Image.open(sample['path']) as img:
            img = img.convert('RGB')
            img = np.array(img)

            if self.use_random_crops:
                cropped = self.crop_transform(image=img)
                img = cropped['image']

            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']
                return img, sample['class_idx'], sample
            else:
                # Return raw numpy array when no transform
                return img, sample['class_idx'], sample


def load_model(model_path: str, model_name: str, num_classes: int = 4):
    """Load model from .pth file."""

    # Create TIMM model
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    # Get target layers for GradCAM
    if 'swin' in model_name:
        target_layers = [model.layers[-1].blocks[-1].norm1]
    else:  # convnext
        target_layers = [model.stages[-1].blocks[-1].norm]

    return model, target_layers


def reshape_transform(tensor):
    """Reshape Swin Transformer output for GradCAM."""
    if tensor.ndim == 3:
        batch_size, num_patches, num_channels = tensor.size()
        height = width = int(num_patches ** 0.5)
        assert height * width == num_patches, "Feature map is not square."
        tensor = tensor.reshape(batch_size, height, width, num_channels)
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 4:
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    return tensor


def generate_gradcam_visualizations(
    model_path: str,
    model_name: str,
    data_dir: str,
    output_dir: str,
    fold: int,
    n_folds: int = 5,
    samples_per_class: int = 50,
    use_random_crops: bool = False,
    crop_size: int = 224
):
    """Generate HIGH-RESOLUTION GradCAM visualizations from test set of book-split."""

    print(f"\n{'='*60}")
    print(f"GradCAM Visualization - Book Split (HIGH RESOLUTION)")
    print(f"Model: {model_name}")
    print(f"Fold: {fold}")
    print(f"Samples per class: {samples_per_class}")
    print(f"{'='*60}\n")

    # Load model
    model, target_layers = load_model(model_path, model_name)
    print(f"Loaded model from: {model_path}")

    # Transform for model input (224x224)
    model_transform = A.Compose([
        A.Resize(224, 224),
        A.ToGray(p=1),
        A.CLAHE(clip_limit=2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Create dataset with NO transform (we'll handle it manually for high-res)
    dataset = BookSplitTestDataset(
        data_dir=data_dir,
        fold=fold,
        n_folds=n_folds,
        transform=None,  # No transform - we'll do it manually
        use_random_crops=use_random_crops,
        crop_size=crop_size
    )

    # Initialize GradCAM
    use_reshape = 'swin' in model_name
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform if use_reshape else None
    )

    # Create output directory
    model_type = "swin_cutout" if 'swin' in model_name else "convnext_fullpage"
    output_subdir = Path(output_dir) / f"gradcam_{model_type}_fold{fold}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Track samples per class
    class_names = dataset.idx_to_class
    samples_collected = {i: 0 for i in range(len(class_names))}

    correct_predictions = 0
    total_predictions = 0

    # Shuffle indices for random sampling
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    print(f"\nGenerating high-resolution visualizations...")
    for idx in tqdm(indices):
        # Get sample info
        sample = dataset.samples[idx]
        label = sample['class_idx']

        # Skip if we have enough samples for this class
        if samples_collected[label] >= samples_per_class:
            if all(count >= samples_per_class for count in samples_collected.values()):
                break
            continue

        # Load original image
        with Image.open(sample['path']) as img:
            original_img = img.convert('RGB')
            original_img_np = np.array(original_img)
            original_h, original_w = original_img_np.shape[:2]

        # For CUTOUT models: use a random 224x224 crop
        # For FULL PAGE models: use the full image
        if use_random_crops:
            # Random crop for cutout model
            crop_transform = A.Compose([
                A.RandomCrop(width=crop_size, height=crop_size),
            ])
            cropped = crop_transform(image=original_img_np)
            working_img = cropped['image']
        else:
            working_img = original_img_np

        # Apply grayscale + CLAHE for display
        display_transform = A.Compose([
            A.ToGray(p=1),
            A.CLAHE(clip_limit=2),
        ])
        original_for_display = display_transform(image=working_img)['image']

        # Convert grayscale to RGB for overlay
        if len(original_for_display.shape) == 2:
            original_for_display = cv2.cvtColor(original_for_display, cv2.COLOR_GRAY2RGB)

        # Prepare model input (224x224)
        model_input = model_transform(image=working_img)['image']
        model_input = model_input.unsqueeze(0).cuda()

        total_predictions += 1

        # Get model's prediction
        with torch.no_grad():
            outputs = model(model_input)
        predicted_class = outputs.argmax().item()

        if predicted_class == label:
            correct_predictions += 1

        # Generate GradCAM heatmap (at 224x224)
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=model_input, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Shape: (224, 224)

        # Get display image dimensions
        display_h, display_w = original_for_display.shape[:2]

        # For FULL PAGE: upscale heatmap to original size
        # For CUTOUT: keep at crop size (224x224)
        if use_random_crops:
            # Cutout: resize heatmap to crop size (should already match)
            heatmap_display = cv2.resize(grayscale_cam, (display_w, display_h), interpolation=cv2.INTER_CUBIC)
        else:
            # Full page: upscale heatmap to original image size
            heatmap_display = cv2.resize(grayscale_cam, (display_w, display_h), interpolation=cv2.INTER_CUBIC)

        # Normalize original for display to [0, 1]
        original_normalized = original_for_display.astype(np.float32) / 255.0

        # Create GradCAM overlay
        visualization = show_cam_on_image(original_normalized, heatmap_display, use_rgb=True)

        # Get class names
        true_class = class_names[label]
        pred_class = class_names[predicted_class]

        # Add correctness indicator to filename
        correct_str = "correct" if predicted_class == label else f"wrong_pred_{pred_class}"

        # Define file paths
        base_name = f"{true_class}_{samples_collected[label]:02d}_{correct_str}"
        original_save_path = output_subdir / f"sample_{base_name}.png"
        gradcam_save_path = output_subdir / f"grad_{base_name}.png"

        # Save HIGH-RESOLUTION images
        cv2.imwrite(
            str(original_save_path),
            cv2.cvtColor(original_for_display, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )
        cv2.imwrite(
            str(gradcam_save_path),
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )

        samples_collected[label] += 1

    # Print summary
    print(f"\n{'='*60}")
    print("GradCAM Visualization Complete")
    print(f"{'='*60}")
    print(f"Output directory: {output_subdir}")
    print(f"\nSamples per class:")
    for class_idx, count in samples_collected.items():
        print(f"  {class_names[class_idx]}: {count}")
    print(f"\nAccuracy on sampled images: {correct_predictions}/{total_predictions} "
          f"({100*correct_predictions/total_predictions:.1f}%)")

    # Save summary
    summary = {
        'model_path': model_path,
        'model_name': model_name,
        'fold': fold,
        'samples_per_class': samples_per_class,
        'samples_collected': {class_names[k]: v for k, v in samples_collected.items()},
        'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
        'total_sampled': total_predictions
    }

    with open(output_subdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate GradCAM visualizations for book-split models')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model .pth file')
    parser.add_argument('--model_name', type=str, required=True,
                        help='TIMM model name (convnextv2_base or swin_s3_base_224)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./gradcam_book_split',
                        help='Directory to save visualizations')
    parser.add_argument('--fold', type=int, required=True,
                        help='Fold number (must match the fold used to train the model)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Total number of folds')
    parser.add_argument('--samples_per_class', type=int, default=50,
                        help='Number of samples to generate per class')
    parser.add_argument('--use_random_crops', action='store_true',
                        help='Use random crops (for Swin cutout model)')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Size of random crops')

    args = parser.parse_args()

    generate_gradcam_visualizations(
        model_path=args.model_path,
        model_name=args.model_name,
        data_dir=os.path.expanduser(args.data_dir),
        output_dir=args.output_dir,
        fold=args.fold,
        n_folds=args.n_folds,
        samples_per_class=args.samples_per_class,
        use_random_crops=args.use_random_crops,
        crop_size=args.crop_size
    )
