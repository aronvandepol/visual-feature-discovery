"""Evaluate book-level split MIL checkpoints to extract F1/precision/recall."""
import argparse
import json
import logging
import os
import sys
import warnings

import pytorch_lightning as pl
import torch

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.parquet_datamodule import ParquetPrintshopDataModule  # noqa: E402
from src.models.lightning_attrimil import LightningAttriMIL  # noqa: E402

torch.set_float32_matmul_precision('medium')


def find_best_checkpoint(fold_dir):
    """Find the best checkpoint (lowest val_loss) in a fold directory."""
    ckpts = [f for f in os.listdir(fold_dir) if f.endswith('.ckpt')]
    if not ckpts:
        return None
    # Sort by val_loss extracted from filename (attrimil-epoch=XX-val_loss=Y.YY.ckpt)
    def get_val_loss(fname):
        try:
            return float(fname.split('val_loss=')[1].replace('.ckpt', ''))
        except (IndexError, ValueError):
            return float('inf')
    ckpts.sort(key=get_val_loss)
    return os.path.join(fold_dir, ckpts[0])


def main(args):
    results = {}

    for fold in range(args.n_folds):
        print(f"\n{'='*60}")
        print(f"Evaluating Fold {fold}")
        print(f"{'='*60}")

        # Find best checkpoint
        fold_dir = os.path.join(args.checkpoint_dir, f'fold{fold}')
        ckpt_path = find_best_checkpoint(fold_dir)
        if not ckpt_path:
            print(f"  No checkpoint found in {fold_dir}, skipping")
            continue
        print(f"  Checkpoint: {ckpt_path}")

        # Set up data module with same fold split
        data_module = ParquetPrintshopDataModule(
            parquet_file=args.parquet_file,
            base_dir=args.base_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            patch_size=args.patch_size,
            stride=args.stride,
            fold=fold,
            n_splits=args.n_folds,
            random_state=42,
            fold_file=args.fold_file
        )
        data_module.prepare_data()
        data_module.setup('fit')

        # Load model from checkpoint
        model = LightningAttriMIL.load_from_checkpoint(
            ckpt_path,
            num_classes=len(data_module.class_names),
            class_names=data_module.class_names,
            patch_size=args.patch_size,
            stride=args.stride,
        )

        # Run validation
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            logger=False,
            enable_progress_bar=True,
        )

        val_results = trainer.validate(model, datamodule=data_module)

        fold_result = val_results[0] if val_results else {}
        results[f'fold_{fold}'] = fold_result
        print(f"  Results: {json.dumps(fold_result, indent=4)}")

    # Compute means
    metrics = ['val_accuracy', 'val_f1', 'val_precision', 'val_recall']
    means = {}
    for m in metrics:
        vals = [results[f'fold_{f}'].get(m, 0) for f in range(args.n_folds) if f'fold_{f}' in results]
        if vals:
            means[f'mean_{m}'] = sum(vals) / len(vals)
            means[f'std_{m}'] = (sum((v - means[f'mean_{m}'])**2 for v in vals) / len(vals)) ** 0.5

    results['summary'] = means

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in means.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'mil_book_split_eval.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MIL book-split checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing fold checkpoint subdirectories')
    parser.add_argument('--parquet_file', type=str, required=True,
                        help='Path to the parquet dataset file')
    parser.add_argument('--base_dir', type=str, default='',
                        help='Base directory for image paths if they are relative')
    parser.add_argument('--fold_file', type=str, required=True,
                        help='Path to a pre-generated fold assignments file')
    parser.add_argument('--output_dir', type=str, default='results_book_split',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()
    main(args)
