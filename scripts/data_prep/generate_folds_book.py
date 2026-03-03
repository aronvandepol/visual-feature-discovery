#!/usr/bin/env python3
"""
Generate fold assignments using GroupKFold for book-level splits.

This ensures that all pages from the same book are in either train or test,
never both - eliminating data leakage from book-specific features.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import json


def extract_book_from_filename(filename: str) -> str:
    """
    Extract book name from filename pattern: class/bookname-pagenum.jpg

    Examples:
        'printshop_name/book_title-123.jpg' -> 'book_title'
    """
    # Get the file part after the class directory
    parts = filename.split('/')
    if len(parts) >= 2:
        file_part = parts[-1]
        # Match bookname-pagenum.jpg pattern
        match = re.match(r'(.+)-\d+\.jpg', file_part, re.IGNORECASE)
        if match:
            return match.group(1)
    # Fallback: return the filename without extension
    return os.path.splitext(os.path.basename(filename))[0]


def main(args):
    print(f"Generating BOOK-LEVEL fold assignments for dataset: {args.parquet_file}")
    print("=" * 60)

    # Load the parquet file
    df = pd.read_parquet(args.parquet_file)
    print(f"Loaded dataset with {len(df)} samples")

    # Determine the filename column
    if 'filename' in df.columns:
        filename_col = 'filename'
    elif 'image_path' in df.columns:
        filename_col = 'image_path'
    else:
        raise ValueError(f"No filename column found. Columns: {df.columns.tolist()}")

    # Determine the class column
    if 'class' in df.columns:
        class_col = 'class'
    elif 'printshop' in df.columns:
        class_col = 'printshop'
    else:
        raise ValueError(f"No class column found. Columns: {df.columns.tolist()}")

    print(f"Using filename column: '{filename_col}'")
    print(f"Using class column: '{class_col}'")

    # Extract book names from filenames
    print("\nExtracting book names from filenames...")
    df['book'] = df[filename_col].apply(extract_book_from_filename)

    unique_books = df['book'].nunique()
    print(f"Found {unique_books} unique books")

    # Show book distribution per class
    print("\nBooks per class:")
    for cls in sorted(df[class_col].unique()):
        cls_books = df[df[class_col] == cls]['book'].nunique()
        cls_pages = len(df[df[class_col] == cls])
        print(f"  {cls}: {cls_books} books, {cls_pages} pages")

    # Generate indices and groups for GroupKFold
    all_indices = np.arange(len(df))
    labels = df[class_col].values
    groups = df['book'].values

    # Perform GroupKFold cross-validation
    gkf = GroupKFold(n_splits=args.n_splits)

    # Create fold assignments dictionary
    fold_assignments = {
        'metadata': {
            'dataset': os.path.basename(args.parquet_file),
            'split_type': 'book_level_groupkfold',
            'n_splits': args.n_splits,
            'total_samples': len(df),
            'total_books': unique_books,
            'filename_column': filename_col,
            'class_column': class_col
        },
        'folds': []
    }

    print("\n" + "=" * 60)
    print("Generating folds with GroupKFold (book-level splits)")
    print("=" * 60)

    # Generate folds - standard k-fold (train/val only, no separate test)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(all_indices, labels, groups=groups)):
        train_idx = [int(i) for i in train_idx]
        val_idx = [int(i) for i in val_idx]

        # Calculate book overlap (should be 0 for proper book-level splits)
        train_books = set(groups[train_idx])
        val_books = set(groups[val_idx])

        train_val_overlap = len(train_books & val_books)

        fold_data = {
            'fold': fold_idx,
            'train_indices': train_idx,
            'val_indices': val_idx,
            'stats': {
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                'train_books': len(train_books),
                'val_books': len(val_books),
                'train_val_book_overlap': train_val_overlap
            }
        }

        fold_assignments['folds'].append(fold_data)

        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_idx):,} samples from {len(train_books)} books")
        print(f"  Val:   {len(val_idx):,} samples from {len(val_books)} books")
        print(f"  Book overlap: {train_val_overlap}")

        if train_val_overlap > 0:
            print("  WARNING: Book overlap detected! This should not happen with GroupKFold.")
        else:
            print("  No book overlap (proper book-level split)")

    # Save fold assignments
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(fold_assignments, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Fold assignments saved to: {output_file}")
    print(f"{'=' * 60}")

    # Generate and save fold statistics
    stats_file = os.path.splitext(output_file)[0] + '_stats.json'
    fold_stats = []

    for fold_data in fold_assignments['folds']:
        fold_idx = fold_data['fold']

        # Train stats
        train_df = df.iloc[fold_data['train_indices']]
        train_stats = {
            'fold': f"{fold_idx}_train",
            'total': len(train_df),
            'books': train_df['book'].nunique(),
            'printshops': train_df[class_col].value_counts().to_dict()
        }

        # Val stats
        val_df = df.iloc[fold_data['val_indices']]
        val_stats = {
            'fold': f"{fold_idx}_val",
            'total': len(val_df),
            'books': val_df['book'].nunique(),
            'printshops': val_df[class_col].value_counts().to_dict()
        }

        fold_stats.extend([train_stats, val_stats])

    with open(stats_file, 'w') as f:
        json.dump(fold_stats, f, indent=2, ensure_ascii=False)

    print(f"Fold statistics saved to: {stats_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate book-level fold assignments using GroupKFold"
    )
    parser.add_argument('--parquet_file', type=str, required=True,
                        help='Path to the parquet file')
    parser.add_argument('--output_file', type=str, default="fold_assignments_book_split.json",
                        help='Path to save the fold assignments JSON file')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    main(args)
