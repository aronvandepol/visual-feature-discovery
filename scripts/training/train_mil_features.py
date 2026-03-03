import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# train_mil_features.py
# Training script for feature-based AttriMIL.
# Same CLI interface as train_mil.py but loads pre-extracted features.

import os
import json
import argparse
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import logging

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='Glyph*missing from font*')
warnings.filterwarnings('ignore', message='Font family*not found')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)
try:
    from pytorch_lightning.loggers import CometLogger
except ImportError:
    CometLogger = None

# Try to load NotoSansCJK font at module scope; fall back gracefully
try:
    noto_font_path = "NotoSansCJK.ttc"
    if os.path.exists(noto_font_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fm.fontManager.addfont(noto_font_path)
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
except Exception:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.data.feature_datamodule import FeaturePrintshopDataModule
from src.models.lightning_attrimil_features import LightningAttriMILFeatures
from datetime import datetime

torch.set_float32_matmul_precision('medium')


def main(args):
    # Load Comet API key from environment (optional)
    comet_api_key = os.environ.get("COMET_API_KEY")

    experiment_name = f"{args.experiment_name}_fold{args.fold}"
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Set up data module
    data_module = FeaturePrintshopDataModule(
        parquet_file=args.parquet_file,
        feature_dir=args.feature_dir,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fold=args.fold,
        n_splits=args.n_splits,
        random_state=args.random_state,
        fold_file=args.fold_file,
        feature_noise_std=args.feature_noise_std,
        feature_dropout=args.feature_dropout,
    )

    data_module.prepare_data()
    data_module.setup('fit')

    model = LightningAttriMILFeatures(
        num_classes=len(data_module.class_names),
        class_names=data_module.class_names,
        feature_dim=args.feature_dim,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        beta=args.beta,
    )

    print(f"\n{'='*60}")
    print(f"Feature-based AttriMIL Training")
    print(f"Feature dir: {args.feature_dir}")
    print(f"Feature dim: {args.feature_dim}")
    print(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")
    print(f"Feature noise: {args.feature_noise_std}, Feature dropout: {args.feature_dropout}")
    print(f"{'='*60}\n")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"checkpoints/fold{args.fold}", exist_ok=True)

    if comet_api_key and CometLogger is not None:
        os.makedirs("comet_logs", exist_ok=True)
        comet_logger = CometLogger(
            api_key=comet_api_key,
            project_name=args.comet_project_name,
            experiment_name=experiment_name,
            save_dir='comet_logs'
        )
        comet_logger.experiment.log_parameter("fold", args.fold)
        comet_logger.experiment.log_parameter("n_splits", args.n_splits)
        comet_logger.experiment.log_parameter("random_state", args.random_state)
        comet_logger.experiment.log_parameter("model_type", "feature_attrimil")
        comet_logger.experiment.log_parameter("feature_dim", args.feature_dim)
        comet_logger.experiment.log_parameter("feature_noise_std", args.feature_noise_std)
        comet_logger.experiment.log_parameter("feature_dropout", args.feature_dropout)
        if args.fold_file:
            comet_logger.experiment.log_parameter("fold_file", os.path.basename(args.fold_file))
    else:
        from pytorch_lightning.loggers import CSVLogger
        comet_logger = CSVLogger("logs", name=experiment_name)
        print("No COMET_API_KEY found. Using CSVLogger. Set COMET_API_KEY for Comet ML tracking.")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/fold{args.fold}',
        filename='attrimil-features-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=35,
        mode='min'
    )

    trainer_kwargs = {
        'max_epochs': 3 if args.test_mode else args.max_epochs,
        'devices': args.num_gpus,
        'logger': comet_logger,
        'callbacks': [checkpoint_callback, early_stop_callback],
        'log_every_n_steps': 5 if args.test_mode else 100,
        'strategy': 'ddp' if args.num_gpus > 1 else 'auto',
        'precision': "16-mixed",
        'accelerator': "gpu",
        'num_nodes': 1,
    }

    if args.test_mode:
        trainer_kwargs.update({
            'limit_val_batches': 15,
            'limit_train_batches': 15,
        })
        print("TEST MODE: Using limited batches and 3 epochs for quick testing")

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=data_module)

    final_model_path = os.path.join(
        f'checkpoints/fold{args.fold}',
        f'final_model_features_fold{args.fold}_{timestamp}.pth'
    )
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    comet_logger.experiment.log_model('final_model', final_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train feature-based AttriMIL model')
    parser.add_argument('--parquet_file', type=str, required=True)
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='Directory with pre-extracted .npy feature files')
    parser.add_argument('--base_dir', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Dimension of pre-extracted features')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--comet_project_name', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--fold_file', type=str, default=None)
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--feature_noise_std', type=float, default=0.0,
                        help='Std of Gaussian noise added to features during training')
    parser.add_argument('--feature_dropout', type=float, default=0.0,
                        help='Dropout rate applied to features during training')

    args = parser.parse_args()
    main(args)
