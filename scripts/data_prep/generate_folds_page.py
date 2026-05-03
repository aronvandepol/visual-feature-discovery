# generate_folds_page.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import json

def main(args):
    print(f"Generating fold assignments for dataset: {args.parquet_file}")

    # Load the parquet file
    df = pd.read_parquet(args.parquet_file)
    print(f"Loaded dataset with {len(df)} samples")

    # Determine the class column for fold statistics
    if 'class' in df.columns:
        class_col = 'class'
    elif 'printshop' in df.columns:
        class_col = 'printshop'
    elif 'printshop_canonical' in df.columns:
        class_col = 'printshop_canonical'
    else:
        raise ValueError(f"No class column found. Columns: {df.columns.tolist()}")
    print(f"Using class column: '{class_col}'")

    # Generate indices for all samples
    all_indices = np.arange(len(df))

    # Perform k-fold cross-validation on the entire dataset
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    # Apply k-fold to all data (no separate test set)
    fold_splits_global_indices = []
    for train_idx, val_idx in kf.split(all_indices):
        fold_splits_global_indices.append((train_idx, val_idx))

    # Create a dictionary to store all fold assignments
    fold_assignments = {
        'metadata': {
            'dataset': os.path.basename(args.parquet_file),
            'split_type': 'page_level_kfold',
            'n_splits': args.n_splits,
            'random_state': args.random_state,
            'total_samples': len(df),
            'class_column': class_col
        },
        'folds': []
    }

    # Store each fold's train/val indices (global indices)
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits_global_indices):
        fold_assignments['folds'].append({
            'fold': fold_idx,
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist()
        })

        print(f"Fold {fold_idx}: Train set size = {len(train_idx)}, Validation set size = {len(val_idx)}")

    # Save fold assignments to a JSON file
    with open(args.output_file, 'w') as f:
        json.dump(fold_assignments, f, indent=2, ensure_ascii=False)

    print(f"Fold assignments saved to {args.output_file}")

    # Generate fold statistics
    fold_stats = []

    # Add each fold's stats
    for fold_data in fold_assignments['folds']:
        fold_idx = fold_data['fold']
        train_df = df.iloc[fold_data['train_indices']]
        val_df = df.iloc[fold_data['val_indices']]

        train_stats = {
            'fold': f"{fold_idx}_train",
            'total': len(train_df),
            'printshops': train_df[class_col].value_counts().to_dict()
        }

        val_stats = {
            'fold': f"{fold_idx}_val",
            'total': len(val_df),
            'printshops': val_df[class_col].value_counts().to_dict()
        }

        fold_stats.append(train_stats)
        fold_stats.append(val_stats)

    # No separate test set in standard k-fold

    # Save fold stats to a JSON file
    stats_file = os.path.splitext(args.output_file)[0] + '_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(fold_stats, f, indent=2, ensure_ascii=False)

    print(f"Fold statistics saved to {stats_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save fold assignments for standard k-fold cross-validation")
    parser.add_argument('--parquet_file', type=str, required=True,
                        help='Path to the parquet file')
    parser.add_argument('--output_file', type=str, default="fold_assignments.json",
                        help='Path to save the fold assignments JSON file')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    # Removed test_size parameter - using standard k-fold without holdout test

    args = parser.parse_args()
    main(args)
