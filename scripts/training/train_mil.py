import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# train_mil.py
import os
import json
import argparse
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import logging
import pandas as pd

# Suppress all matplotlib warnings early
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

from src.data.parquet_datamodule import ParquetPrintshopDataModule
from src.models.lightning_attrimil import LightningAttriMIL
from datetime import datetime

import sys

torch.set_float32_matmul_precision('medium')

# Set up Korean font handling for HPC environment
def setup_korean_fonts():
    """Setup Korean fonts using matplotlib's font manager for reliable detection"""
    print("=" * 60)
    print("Setting up Korean fonts for HPC/SLURM environment...")
    print("=" * 60)

    # First, get all available fonts from matplotlib's font manager
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"Total fonts available in matplotlib: {len(available_fonts)}")

    # Look for Korean fonts using matplotlib's font manager
    korean_font_patterns = ['Nanum', 'Noto', 'UnDotum', 'Batang', 'Gulim', 'Dotum', 'Malgun']
    available_korean_fonts = []

    for pattern in korean_font_patterns:
        matching_fonts = [f for f in available_fonts if pattern in f]
        if matching_fonts:
            available_korean_fonts.extend(matching_fonts)
            print(f"Found {pattern} fonts: {matching_fonts}")

    # Also check font files directly for better debugging
    korean_font_files = []
    for font in fm.fontManager.ttflist:
        if any(pattern.lower() in font.name.lower() for pattern in korean_font_patterns):
            korean_font_files.append((font.name, font.fname))
            print(f"Korean font file: {font.name} -> {font.fname}")

    # Test Korean character support
    def test_korean_support(font_name):
        """Test if a font can render Korean characters"""
        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            # Test with common Korean characters that were causing warnings
            test_chars = "대한민국 테스트"  # "South Korea Test"
            ax.text(0.5, 0.5, test_chars, fontname=font_name, fontsize=12)
            plt.close(fig)
            return True
        except Exception as e:
            print(f"Font {font_name} failed Korean test: {e}")
            return False

    # Configure the best Korean font
    best_korean_font = None

    if available_korean_fonts:
        # Prioritize fonts in order of Korean character completeness
        font_priority = ['NanumBarunGothic', 'NanumSquare', 'NanumGothic', 'NotoSansCJK', 'NotoSansKR']

        # Try priority fonts first
        for priority_font in font_priority:
            for available_font in available_korean_fonts:
                if priority_font.lower() in available_font.lower():
                    if test_korean_support(available_font):
                        best_korean_font = available_font
                        break
            if best_korean_font:
                break

        # If no priority font works, try any available Korean font
        if not best_korean_font:
            for font in available_korean_fonts:
                if test_korean_support(font):
                    best_korean_font = font
                    break

        if best_korean_font:
            # Configure matplotlib to use the Korean font
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [best_korean_font] + plt.rcParams['font.sans-serif']

            print(f"Successfully configured Korean font: {best_korean_font}")
            print(f"Font supports Korean characters: {test_korean_support(best_korean_font)}")

            # Test rendering Korean characters
            try:
                fig, ax = plt.subplots(figsize=(2, 1))
                test_text = "한글 테스트 - Korean Test"
                ax.text(0.5, 0.5, test_text, fontname=best_korean_font, ha='center', va='center')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                plt.tight_layout()
                plt.close(fig)
                print("Korean character rendering test passed")
            except Exception as e:
                print(f"Korean character rendering test failed: {e}")

            print("=" * 60)
            return True
        else:
            print("Found Korean fonts but none support full Korean character set")

    # If no suitable Korean fonts found
    print("No suitable Korean fonts found in matplotlib font manager")
    print("\nAvailable fonts that might work:")
    fallback_fonts = [f for f in available_fonts if any(term in f.lower() for term in ['dejavu', 'liberation', 'arial', 'sans'])]
    for font in fallback_fonts[:5]:
        print(f"  - {font}")

    print("\nTo install Korean fonts on HPC system:")
    print("1. Run: bash install_korean_fonts.sh")
    print("2. Or manually: mkdir -p ~/.fonts && cd ~/.fonts")
    print("3. Download: wget https://github.com/naver/nanumfont/releases/download/VER2.6/NanumFont_TTF_ALL.zip")
    print("4. Extract: unzip NanumFont_TTF_ALL.zip")
    print("5. Update cache: fc-cache -f -v")

    # Use best available fallback
    plt.rcParams['font.family'] = 'sans-serif'
    if fallback_fonts:
        plt.rcParams['font.sans-serif'] = fallback_fonts[:3] + plt.rcParams['font.sans-serif']
        print(f"\nUsing fallback fonts: {fallback_fonts[:3]}")

    print("=" * 60)
    return False

# Setup Korean fonts
korean_font_available = setup_korean_fonts()


def visualize_and_log_patches(data_module, num_samples, logger):
    figures = data_module.visualize_patches(num_samples)
    for i, fig in enumerate(figures):
        logger.experiment.log_figure(figure_name=f'patch_visualization_sample{i+1}', figure=fig)
        plt.close(fig)  # Close the figure to free up memory


def get_class_names_from_parquet(parquet_file):
    """Read class names from parquet without creating the full dataset."""
    df = pd.read_parquet(parquet_file)
    # Apply same canonicalization as ParquetPrintshopDataset
    mapping = {
        '대동인쇄 주식회사 大東印刷 株式會社': '대동인쇄소',
        '대동인쇄소 大東印刷所': '대동인쇄소',
    }
    df['printshop'] = df['printshop'].replace(mapping)
    unique_printshops = sorted(df['printshop'].unique())
    return unique_printshops


def main(args):
    # Increase NCCL timeout for multi-GPU to avoid watchdog kills during setup
    if args.num_gpus > 1:
        os.environ.setdefault('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', '1800')

    # Load Comet API key from environment (optional)
    comet_api_key = os.environ.get("COMET_API_KEY")

    # Create experiment name with fold information
    experiment_name = f"{args.experiment_name}_fold{args.fold}"

    # Get the current date and time
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Get class names from parquet (lightweight, no image loading — better for DDP)
    class_names = get_class_names_from_parquet(args.parquet_file)

    # Set up data module (let Trainer handle prepare_data/setup for proper DDP coordination)
    data_module = ParquetPrintshopDataModule(
        parquet_file=args.parquet_file,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        stride=args.stride,
        fold=args.fold,
        n_splits=args.n_splits,
        random_state=args.random_state,
        fold_file=args.fold_file
    )

    # Set up model
    model = LightningAttriMIL(
        num_classes=len(class_names),
        class_names=class_names,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        beta=args.beta,
        patch_size=args.patch_size,
        stride=args.stride,
        compile_backbone=args.compile_backbone,
    )

    print(f"\n{'='*60}")
    print(f"Using AttriMIL (optimized attention, split backbone)")
    print(f"Patch size: {args.patch_size}, Stride: {args.stride}")
    print(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")
    print(f"torch.compile backbone: {args.compile_backbone}, GPUs: {args.num_gpus}")
    print(f"{'='*60}\n")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"checkpoints/fold{args.fold}", exist_ok=True)

    # Set up logger
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
        comet_logger.experiment.log_parameter("model_type", "attrimil")
        comet_logger.experiment.log_parameter("compile_backbone", args.compile_backbone)
        if args.fold_file:
            comet_logger.experiment.log_parameter("fold_file", os.path.basename(args.fold_file))
    else:
        from pytorch_lightning.loggers import CSVLogger
        comet_logger = CSVLogger("logs", name=experiment_name)
        print("No COMET_API_KEY found. Using CSVLogger. Set COMET_API_KEY for Comet ML tracking.")

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/fold{args.fold}',
        filename='attrimil-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=35,
        mode='min'
    )

    # Set up trainer
    trainer_kwargs = {
        'max_epochs': 3 if args.test_mode else args.max_epochs,
        'devices': args.num_gpus,
        'logger': comet_logger,
        'callbacks': [checkpoint_callback, early_stop_callback],
        'log_every_n_steps': 5 if args.test_mode else 100,
        'strategy': 'auto',
        'precision': "16-mixed",
        'accelerator': "gpu",
        'num_nodes': 1,
    }

    # Add batch limits for test mode
    if args.test_mode:
        trainer_kwargs.update({
            'limit_val_batches': 15,
            'limit_train_batches': 15,
        })
        print("TEST MODE: Using limited batches and 3 epochs for quick testing")

    trainer = pl.Trainer(**trainer_kwargs)

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Save the final model with fold information
    final_model_path = os.path.join(f'checkpoints/fold{args.fold}', f'final_model_fold{args.fold}_{timestamp}.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    # Log the final model to Comet.ml
    if comet_api_key:
        comet_logger.experiment.log_model('final_model', final_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AttriMIL model with parquet data and standard k-fold validation')
    parser.add_argument('--parquet_file', type=str, required=True,
                        help='Path to the parquet file containing dataset information')
    parser.add_argument('--base_dir', type=str, default="",
                        help='Base directory for image paths if they are relative in the parquet file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Size of image patches')
    parser.add_argument('--stride', type=int, default=4,
                        help='Stride for patch extraction')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Weight for spatial loss')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='Weight for ranking loss')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--comet_project_name', type=str, required=True,
                        help='Comet.ml project name')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Base name for this experiment run')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold to use for validation (0 to n_splits-1)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for cross-validation')
    # Removed test_size parameter - using standard k-fold
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--fold_file', type=str, default=None,
                        help='Path to a pre-generated fold assignments file')
    parser.add_argument('--test_mode', action='store_true',
                        help='Enable test mode with limited batches and epochs for quick testing')
    parser.add_argument('--compile_backbone', action=argparse.BooleanOptionalAction, default=True,
                        help='Use torch.compile on frozen backbone stages (default: True). Disable with --no-compile-backbone for multi-GPU DDP.')

    args = parser.parse_args()
    main(args)
