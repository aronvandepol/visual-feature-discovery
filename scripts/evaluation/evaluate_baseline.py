#!/usr/bin/env python3
"""
Evaluate CNN/Swin models with book-level splits to assess true generalization.

Supports:
1. Single fold evaluation (--fold N)
2. Original page-level split verification (--split_mode original)
3. Each fold logs to its own Comet ML experiment
"""

import argparse
import os
import re
import json
import numpy as np
import torch
import timm
from pathlib import Path
from PIL import Image
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None


def extract_book(filename: str) -> str:
    """Extract book name from filename pattern: class/bookname-pagenum.jpg"""
    parts = filename.split('/')
    if len(parts) >= 2:
        book_part = parts[1]
        match = re.match(r'(.+)-\d+\.jpg', book_part, re.IGNORECASE)
        if match:
            return match.group(1)
    return filename


class PrintshopDataset(Dataset):
    """Dataset for printshop classification images."""

    def __init__(self, data_dir: str, indices: list = None, transform=None,
                 use_random_crops: bool = False, num_crops: int = 4, crop_size: int = 224):
        self.data_dir = Path(data_dir).expanduser()
        self.transform = transform
        self.use_random_crops = use_random_crops
        self.num_crops = num_crops
        self.crop_size = crop_size

        self.all_samples = []
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
                        'filename': filename,
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'book': book
                    })

        if indices is not None:
            self.samples = [self.all_samples[i] for i in indices]
        else:
            self.samples = self.all_samples

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


def get_transforms(crop_size: int = 224):
    """Get evaluation transforms."""
    return A.Compose([
        A.Resize(crop_size, crop_size),
        A.ToGray(p=1),
        A.CLAHE(clip_limit=2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_book_kfold_split(data_dir: str, fold: int, n_splits: int = 5):
    """Get train/test indices for a specific fold using GroupKFold."""
    dataset = PrintshopDataset(data_dir)

    books = np.array([s['book'] for s in dataset.all_samples])
    labels = np.array([s['class_idx'] for s in dataset.all_samples])
    indices = np.arange(len(books))

    gkf = GroupKFold(n_splits=n_splits)

    for i, (train_idx, test_idx) in enumerate(gkf.split(indices, labels, groups=books)):
        if i == fold:
            train_books = set(books[j] for j in train_idx)
            test_books = set(books[j] for j in test_idx)
            overlap = train_books & test_books

            return {
                'train_idx': train_idx.tolist(),
                'test_idx': test_idx.tolist(),
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'train_books': len(train_books),
                'test_books': len(test_books),
                'overlap': len(overlap)
            }

    raise ValueError(f"Fold {fold} not found (n_splits={n_splits})")


def get_original_split(data_dir: str):
    """Get original page-level random split (to match Comet ML training)."""
    dataset = PrintshopDataset(data_dir)
    n_samples = len(dataset.all_samples)

    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_samples, generator=generator).tolist()

    train_idx = indices[:train_size]
    test_idx = indices[train_size + val_size:]  # Skip validation, use test

    books = [dataset.all_samples[i]['book'] for i in range(n_samples)]
    train_books = set(books[i] for i in train_idx)
    test_books = set(books[i] for i in test_idx)
    overlap = train_books & test_books

    return {
        'train_idx': train_idx,
        'test_idx': test_idx,
        'train_samples': len(train_idx),
        'test_samples': len(test_idx),
        'train_books': len(train_books),
        'test_books': len(test_books),
        'overlap': len(overlap),
        'overlap_pct': 100 * len(overlap) / len(test_books)
    }


def load_model(model_name: str, model_path: str, num_classes: int = 4, device: str = 'cuda'):
    """Load a trained model from .pth file."""
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    state_dict = torch.load(model_path, map_location=device)

    if any(k.startswith('model.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def evaluate_model(model, dataloader, device: str = 'cuda'):
    """Evaluate model and return metrics."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'labels': all_labels
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate models with book-level splits')
    parser.add_argument('--model_name', type=str, required=True,
                        help='TIMM model name (e.g., convnextv2_base, swin_s3_base_224)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved .pth model weights')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--split_mode', type=str, default='book_fold',
                        choices=['original', 'book_fold'],
                        help='Split mode: original (page-level) or book_fold (book-level)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number to evaluate (0-indexed)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Total number of folds')
    parser.add_argument('--use_random_crops', action='store_true',
                        help='Use random crops (for Swin multicrop model)')
    parser.add_argument('--num_crops', type=int, default=4,
                        help='Number of crops per image')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Crop size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--comet_project', type=str, default='visual-feature-discovery-baseline',
                        help='Comet ML project name')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSON file for results')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Initialize Comet ML experiment (optional)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    comet_api_key = os.environ.get("COMET_API_KEY")

    if args.split_mode == 'book_fold':
        exp_name = f"{args.model_name}_book_fold{args.fold}_{timestamp}"
        tags = ["evaluation", "book_split", f"fold_{args.fold}", f"{args.n_folds}fold_cv"]
    else:
        exp_name = f"{args.model_name}_original_split_eval_{timestamp}"
        tags = ["evaluation", "original_split", "page_level"]

    experiment = None
    if comet_api_key and Experiment is not None:
        experiment = Experiment(
            api_key=comet_api_key,
            project_name=args.comet_project,
            workspace=os.environ.get("COMET_WORKSPACE"),
        )
        experiment.set_name(exp_name)
        experiment.log_parameters({
            "model_name": args.model_name,
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "split_mode": args.split_mode,
            "fold": args.fold if args.split_mode == 'book_fold' else None,
            "n_folds": args.n_folds,
            "use_random_crops": args.use_random_crops,
            "num_crops": args.num_crops if args.use_random_crops else None,
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
        })
        experiment.add_tags(tags)
    else:
        print("No COMET_API_KEY found. Running evaluation without Comet ML tracking.")

    # Load model
    print(f"\nLoading model: {args.model_name}")
    print(f"From: {args.model_path}")
    model = load_model(args.model_name, args.model_path, device=device)

    transform = get_transforms(args.crop_size)

    # Get split
    if args.split_mode == 'book_fold':
        print(f"\n{'='*60}")
        print(f"BOOK-LEVEL SPLIT - Fold {args.fold + 1}/{args.n_folds}")
        print(f"{'='*60}")
        split = get_book_kfold_split(args.data_dir, args.fold, args.n_folds)
    else:
        print(f"\n{'='*60}")
        print("ORIGINAL SPLIT (page-level)")
        print(f"{'='*60}")
        split = get_original_split(args.data_dir)

    print("\nSplit Statistics:")
    print(f"  Train samples: {split['train_samples']}")
    print(f"  Test samples: {split['test_samples']}")
    print(f"  Train books: {split['train_books']}")
    print(f"  Test books: {split['test_books']}")
    print(f"  Book overlap: {split['overlap']}")

    if experiment:
        experiment.log_metrics({
            "train_samples": split['train_samples'],
            "test_samples": split['test_samples'],
            "train_books": split['train_books'],
            "test_books": split['test_books'],
            "book_overlap": split['overlap'],
        })

    # Create test dataloader
    test_dataset = PrintshopDataset(
        args.data_dir,
        indices=split['test_idx'],
        transform=transform,
        use_random_crops=args.use_random_crops,
        num_crops=args.num_crops,
        crop_size=args.crop_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Evaluate
    metrics = evaluate_model(model, test_loader, device)

    print("\nResults:")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    # Log to Comet (if available)
    if experiment:
        experiment.log_metrics({
            "test_accuracy": metrics['accuracy'],
            "test_f1": metrics['f1'],
            "test_precision": metrics['precision'],
            "test_recall": metrics['recall'],
        })
        class_names = list(test_dataset.class_to_idx.keys())
        experiment.log_confusion_matrix(
            y_true=metrics['labels'],
            y_predicted=metrics['predictions'],
            labels=class_names,
            title=f"Confusion Matrix - {args.model_name} - {'Fold ' + str(args.fold) if args.split_mode == 'book_fold' else 'Original Split'}"
        )

    # Save results
    results = {
        'model_name': args.model_name,
        'model_path': args.model_path,
        'split_mode': args.split_mode,
        'fold': args.fold if args.split_mode == 'book_fold' else None,
        'n_folds': args.n_folds,
        'use_random_crops': args.use_random_crops,
        'accuracy': float(metrics['accuracy']),
        'f1': float(metrics['f1']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'split_stats': {
            'train_samples': split['train_samples'],
            'test_samples': split['test_samples'],
            'book_overlap': split['overlap']
        }
    }

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

    if experiment:
        experiment.end()
        print(f"\nComet experiment: {experiment.url}")


if __name__ == '__main__':
    main()
