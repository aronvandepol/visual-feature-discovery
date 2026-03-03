#!/usr/bin/env python3
"""
Comprehensive smoke tests for visual-feature-discovery.

Runs on GPU with sample_data/ (10 images, 4 classes, 2 folds).
Each test prints PASS/FAIL and a summary is printed at the end.
"""

import sys
import os

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import traceback
import time
import json
import tempfile
import shutil

import numpy as np
import pandas as pd
import torch

# ── Paths ──────────────────────────────────────────────────────────────
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample_data")
PARQUET_FILE = os.path.join(SAMPLE_DIR, "sample_dataset.parquet")
FOLD_FILE = os.path.join(SAMPLE_DIR, "sample_folds.json")
IMAGE_DIR = os.path.join(SAMPLE_DIR, "images")

results = []


def run_test(name, fn):
    """Run a test function and record result."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"  PASS ({elapsed:.1f}s)")
        results.append((name, "PASS", elapsed, None))
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        print(f"  FAIL ({elapsed:.1f}s): {e}")
        print(tb)
        results.append((name, "FAIL", elapsed, str(e)))


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_parquet_dataset():
    """Test ParquetPrintshopDataset loads and returns correct shapes."""
    from src.data.parquet_datamodule import ParquetPrintshopDataset
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(896, 896),
        A.ToGray(p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])

    ds = ParquetPrintshopDataset(
        parquet_file=PARQUET_FILE,
        base_dir=IMAGE_DIR,
        transform=transform,
        patch_size=64,
        stride=64,
    )

    assert len(ds) == 10, f"Expected 10 samples, got {len(ds)}"
    assert len(ds.class_names) == 4, f"Expected 4 classes, got {len(ds.class_names)}"
    print(f"  Dataset size: {len(ds)}, Classes: {ds.class_names}")

    patches, label = ds[0]
    print(f"  patches shape: {patches.shape}, label: {label}")
    assert patches.ndim == 5, f"Expected 5D patches (H,W,C,ph,pw), got {patches.ndim}D"
    assert isinstance(label, int), f"Expected int label, got {type(label)}"


def test_parquet_datamodule():
    """Test ParquetPrintshopDataModule setup with fold file."""
    from src.data.parquet_datamodule import ParquetPrintshopDataModule

    dm = ParquetPrintshopDataModule(
        parquet_file=PARQUET_FILE,
        base_dir=IMAGE_DIR,
        batch_size=2,
        num_workers=0,
        patch_size=64,
        stride=64,
        fold=0,
        n_splits=2,
        fold_file=FOLD_FILE,
    )

    dm.prepare_data()
    dm.setup("fit")

    assert dm.train_dataset is not None, "train_dataset is None"
    assert dm.val_dataset is not None, "val_dataset is None"
    print(f"  Train: {len(dm.train_dataset)}, Val: {len(dm.val_dataset)}")

    # Test dataloader iteration
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))
    patches, labels = batch
    print(f"  Batch patches: {patches.shape}, labels: {labels.shape}")
    assert patches.ndim == 6, f"Expected 6D batch (B,H,W,C,ph,pw), got {patches.ndim}D"


def test_baseline_datamodule():
    """Test BookSplitDataModule and BookSplitDataset for baseline models."""
    from scripts.training.train_baseline_book import BookSplitDataModule

    dm = BookSplitDataModule(
        data_dir=IMAGE_DIR,
        fold=0,
        n_folds=2,
        batch_size=2,
        num_workers=0,
    )

    dm.setup()

    assert dm.split_info is not None, "split_info is None"
    print(f"  Split info: {dm.split_info}")

    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))
    imgs, labels = batch
    print(f"  Batch images: {imgs.shape}, labels: {labels.shape}")
    assert imgs.ndim == 4, f"Expected 4D (B,C,H,W), got {imgs.ndim}D"


# ═══════════════════════════════════════════════════════════════════════
# 2. MODEL FORWARD PASS TESTS (GPU)
# ═══════════════════════════════════════════════════════════════════════

def test_attrimil_forward():
    """Test AttriMIL model forward pass on GPU (with compile_backbone=False)."""
    from src.models.attrimil import AttriMIL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttriMIL(num_classes=4, ranking_strategy="hybrid", compile_backbone=False).to(device)

    # Simulate a batch: (B=2, patches_h=4, patches_w=4, C=3, H=64, W=64)
    x = torch.randn(2, 4, 4, 3, 64, 64, device=device)

    with torch.no_grad():
        y_prob, y_hat, A, H, extra_info = model(x)

    print(f"  y_prob: {y_prob.shape}, y_hat: {y_hat.shape}")
    print(f"  A: {A.shape}, H: {H.shape}")
    assert y_prob.shape == (2, 4), f"Expected (2,4), got {y_prob.shape}"
    assert y_hat.shape == (2,), f"Expected (2,), got {y_hat.shape}"
    assert A.shape == (2, 1, 16), f"Expected (2,1,16), got {A.shape}"
    assert H.shape == (2, 16, 512), f"Expected (2,16,512), got {H.shape}"

    # Test spatial loss
    spatial_loss = model.calculate_spatial_loss(A)
    print(f"  Spatial loss: {spatial_loss.item():.4f}")
    assert spatial_loss.ndim == 0, "Spatial loss should be scalar"

    # Test ranking loss
    labels = torch.tensor([0, 1], device=device)
    ranking_loss = model.calculate_ranking_loss(A, H, labels, extra_info)
    print(f"  Ranking loss: {ranking_loss.item():.4f}")


def test_attrimil_features_forward():
    """Test AttriMILFeatures model forward pass on GPU."""
    from src.models.attrimil_features import AttriMILFeatures

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttriMILFeatures(num_classes=4, feature_dim=256).to(device)

    # Simulate pre-extracted features: (B=2, N_patches=16, feature_dim=256)
    x = torch.randn(2, 16, 256, device=device)

    with torch.no_grad():
        y_prob, y_hat, A, H = model(x)

    print(f"  y_prob: {y_prob.shape}, y_hat: {y_hat.shape}")
    print(f"  A: {A.shape}, H: {H.shape}")
    assert y_prob.shape == (2, 4)


def test_flash_attention():
    """Test FlashAttention / OptimizedAttention module."""
    from src.models.flash_attention import OptimizedAttention

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn = OptimizedAttention(L=512, D=128).to(device)

    x = torch.randn(2, 16, 512, device=device)
    with torch.no_grad():
        out = attn(x)

    print(f"  Input: {x.shape}, Output: {out.shape}")
    assert out.shape == (2, 1, 16), f"Expected (2,1,16), got {out.shape}"


def test_lightning_timm():
    """Test LightningTIMM baseline wrapper forward pass."""
    from src.models.lightning_timm import LightningTIMM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightningTIMM(num_classes=4, model_name="resnet18").to(device)

    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(x)

    print(f"  logits: {logits.shape}")
    assert logits.shape == (2, 4), f"Expected (2,4), got {logits.shape}"


# ═══════════════════════════════════════════════════════════════════════
# 3. LIGHTNING MODULE TESTS (GPU)
# ═══════════════════════════════════════════════════════════════════════

def test_lightning_attrimil_training_step():
    """Test LightningAttriMIL training step."""
    from src.models.lightning_attrimil import LightningAttriMIL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightningAttriMIL(
        num_classes=4,
        class_names=["A", "B", "C", "D"],
        patch_size=64,
        stride=64,
        compile_backbone=False,
    ).to(device)

    x = torch.randn(2, 4, 4, 3, 64, 64, device=device)
    y = torch.tensor([0, 1], device=device)
    batch = (x, y)

    loss = model.training_step(batch, 0)
    print(f"  Training loss: {loss.item():.4f}")
    assert loss.ndim == 0 and not torch.isnan(loss), "Loss should be a finite scalar"


def test_lightning_attrimil_features_training_step():
    """Test LightningAttriMILFeatures training step."""
    from src.models.lightning_attrimil_features import LightningAttriMILFeatures

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightningAttriMILFeatures(
        num_classes=4,
        class_names=["A", "B", "C", "D"],
        feature_dim=256,
    ).to(device)

    x = torch.randn(2, 16, 256, device=device)
    y = torch.tensor([0, 1], device=device)
    batch = (x, y)

    loss = model.training_step(batch, 0)
    print(f"  Training loss: {loss.item():.4f}")
    assert loss.ndim == 0 and not torch.isnan(loss)


def test_lightning_timm_training_step():
    """Test LightningTIMM training step."""
    from src.models.lightning_timm import LightningTIMM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightningTIMM(num_classes=4, model_name="resnet18").to(device)

    x = torch.randn(2, 3, 224, 224, device=device)
    y = torch.tensor([0, 1], device=device)
    batch = (x, y)

    loss = model.training_step(batch, 0)
    print(f"  Training loss: {loss.item():.4f}")
    assert loss.ndim == 0 and not torch.isnan(loss)


# ═══════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_attention_vis():
    """Test attention visualization produces a figure."""
    from src.visualization.attention_vis import visualize_attention
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Simulate patches and attention
    patches = torch.randn(4, 4, 3, 64, 64)
    attention = torch.softmax(torch.randn(4, 4), dim=-1)
    y_prob = torch.tensor([0.1, 0.7, 0.1, 0.1])
    y_true = torch.tensor(1)
    class_names = ["A", "B", "C", "D"]

    fig = visualize_attention(patches, attention, y_prob, y_true, class_names, patch_size=64, stride=64)
    assert fig is not None, "visualize_attention returned None"
    print(f"  Figure created: {fig.get_size_inches()}")
    plt.close(fig)


def test_flash_attention_vis():
    """Test flash attention visualization produces a figure."""
    from src.visualization.flash_attention_vis import visualize_flash_attention
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    patches = torch.randn(4, 4, 3, 64, 64)
    attention = torch.softmax(torch.randn(1, 1, 16), dim=-1)
    y_prob = torch.tensor([0.1, 0.7, 0.1, 0.1])
    y_true = torch.tensor(1)
    class_names = ["A", "B", "C", "D"]

    fig = visualize_flash_attention(patches, attention, y_prob, y_true, class_names, patch_size=64, stride=64)
    assert fig is not None, "visualize_flash_attention returned None"
    print(f"  Figure created: {fig.get_size_inches()}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 5. DATA PREP TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_generate_folds_book():
    """Test book-level fold generation."""
    from scripts.data_prep.generate_folds_book import main as gen_folds_main
    import argparse

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test_folds.json")
        args = argparse.Namespace(
            parquet_file=PARQUET_FILE,
            output_file=output_file,
            n_splits=2,
            random_state=42,
        )
        gen_folds_main(args)

        assert os.path.exists(output_file), "Fold file not created"
        with open(output_file) as f:
            folds = json.load(f)

        assert len(folds["folds"]) == 2, f"Expected 2 folds, got {len(folds['folds'])}"
        for fold in folds["folds"]:
            overlap = fold["stats"]["train_val_book_overlap"]
            assert overlap == 0, f"Book overlap detected: {overlap}"
        print(f"  Generated {len(folds['folds'])} folds, no book overlap")


def test_extract_backbone_features():
    """Test backbone feature extraction on sample images."""
    from scripts.data_prep.extract_backbone_features import (
        build_backbone, build_deterministic_transform, extract_features_for_image,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_backbone().to(device)
    transform = build_deterministic_transform()

    # Get a sample image path
    df = pd.read_parquet(PARQUET_FILE)
    img_path = os.path.join(IMAGE_DIR, df.iloc[0]["image_path"])

    features, (n_h, n_w) = extract_features_for_image(
        img_path, transform, backbone, device,
        patch_size=64, stride=64, batch_size=512,
    )
    print(f"  Features shape: {features.shape}, dtype: {features.dtype}")
    print(f"  Patches grid: {n_h}x{n_w}")
    assert features.ndim == 2, f"Expected 2D features, got {features.ndim}D"
    assert features.shape[1] == 256, f"Expected 256-dim features, got {features.shape[1]}"
    assert features.shape[0] == n_h * n_w


# ═══════════════════════════════════════════════════════════════════════
# 6. TRAINING SMOKE TESTS (full Trainer, 1 epoch)
# ═══════════════════════════════════════════════════════════════════════

def test_train_mil_1epoch():
    """Smoke test: train AttriMIL for 1 epoch with sample data."""
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
    from src.data.parquet_datamodule import ParquetPrintshopDataModule
    from src.models.lightning_attrimil import LightningAttriMIL

    dm = ParquetPrintshopDataModule(
        parquet_file=PARQUET_FILE,
        base_dir=IMAGE_DIR,
        batch_size=2,
        num_workers=0,
        patch_size=64,
        stride=64,
        fold=0,
        n_splits=2,
        fold_file=FOLD_FILE,
    )
    dm.prepare_data()
    dm.setup("fit")

    model = LightningAttriMIL(
        num_classes=len(dm.class_names),
        class_names=dm.class_names,
        patch_size=64,
        stride=64,
        compile_backbone=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = CSVLogger(tmpdir, name="test_mil")
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            strategy="auto",
            precision="16-mixed",
            logger=logger,
            enable_checkpointing=False,
            log_every_n_steps=1,
        )
        trainer.fit(model, datamodule=dm)

    print(f"  Completed 1 epoch of AttriMIL training")


def test_train_baseline_1epoch():
    """Smoke test: train baseline ResNet18 for 1 epoch with sample data."""
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
    from scripts.training.train_baseline_book import BookSplitDataModule
    from src.models.lightning_timm import LightningTIMM

    dm = BookSplitDataModule(
        data_dir=IMAGE_DIR,
        fold=0,
        n_folds=2,
        batch_size=2,
        num_workers=0,
    )
    dm.setup()

    model = LightningTIMM(num_classes=4, model_name="resnet18")

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = CSVLogger(tmpdir, name="test_baseline")
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            strategy="auto",
            precision="16-mixed",
            logger=logger,
            enable_checkpointing=False,
            log_every_n_steps=1,
        )
        trainer.fit(model, dm)

    print(f"  Completed 1 epoch of baseline ResNet18 training")


# ═══════════════════════════════════════════════════════════════════════
# 7. FEATURE EXTRACTION PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_feature_extraction_config():
    """Test feature_extraction config module imports and has expected attrs."""
    from feature_extraction.config import PATCH_SIZE
    assert isinstance(PATCH_SIZE, int), f"PATCH_SIZE should be int, got {type(PATCH_SIZE)}"
    print(f"  PATCH_SIZE = {PATCH_SIZE}")


def test_feature_extraction_helpers():
    """Test feature_extraction helpers preprocess and patch extraction."""
    from feature_extraction.helpers import preprocess_image, tensor_to_patch_grid

    # preprocess_image expects a file path and input_size, use a sample image
    df = pd.read_parquet(PARQUET_FILE)
    img_path = os.path.join(IMAGE_DIR, df.iloc[0]["image_path"])
    tensor = preprocess_image(img_path, input_size=896)
    print(f"  Preprocessed tensor shape: {tensor.shape}")
    # Returns (1, C, H, W) batch tensor
    assert tensor.ndim == 4, f"Expected 4D tensor (1,C,H,W), got {tensor.ndim}D"

    # Test patch grid with (1, C, H, W) tensor (expected by tensor_to_patch_grid)
    patches, gh, gw = tensor_to_patch_grid(tensor, patch_size=64, stride=64)
    print(f"  Patch grid: {gh}x{gw}, patches shape: {patches.shape}")
    assert patches.ndim == 4, f"Expected 4D patches (N,C,ph,pw), got {patches.ndim}D"
    assert patches.shape[0] == gh * gw


def test_feature_extraction_schemas():
    """Test feature_extraction schemas module."""
    from feature_extraction.schemas import get_core_schema
    schema = get_core_schema()
    print(f"  Core schema fields: {schema.names}")
    assert len(schema) > 0, "Schema should have fields"


# ═══════════════════════════════════════════════════════════════════════
# 8. UTILITY TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_cache_manager():
    """Test ImageCache basic operations."""
    from src.utils.cache_manager import ImageCache

    cache = ImageCache(max_size_gb=0.01)
    key = "test_key"
    value = np.random.rand(100, 100, 3).astype(np.float32)

    cache.put(key, value)
    retrieved = cache.get(key)
    assert retrieved is not None, "Cache miss for just-inserted item"
    assert np.allclose(retrieved, value), "Retrieved value doesn't match"
    print(f"  Cache hit/miss working correctly")
    cache_stats = cache.stats()
    print(f"  Cache stats: {cache_stats}")


def test_batch_accumulator():
    """Test BatchAccumulator basic operations."""
    from src.utils.batch_accumulator import BatchAccumulator

    acc = BatchAccumulator(target_batch_size=4)

    # Manually add patches to the accumulator
    for i in range(6):
        patch = torch.randn(3, 64, 64)
        acc.patches.append(patch)
        acc.metadata.append({"idx": i})
        acc.current_batch_size += 1

        if acc.current_batch_size >= acc.target_batch_size:
            batch_tensor, batch_meta = acc.get_batch()
            print(f"  Got batch of {batch_tensor.shape[0]} patches at i={i}")

    # Get remaining
    if acc.has_data():
        batch_tensor, batch_meta = acc.get_batch()
        print(f"  Got remaining {batch_tensor.shape[0]} patches")

    print(f"  BatchAccumulator working correctly")


# ═══════════════════════════════════════════════════════════════════════
# 9. INFERENCE TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_inference_end_to_end():
    """Train a tiny AttriMIL model, save checkpoint, run inference."""
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from src.data.parquet_datamodule import ParquetPrintshopDataModule
    from src.models.lightning_attrimil import LightningAttriMIL
    from scripts.inference.inference import (
        load_model, get_val_transform, preprocess_image,
        run_inference_single, decode_prediction,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Train for 1 epoch and save a checkpoint
        dm = ParquetPrintshopDataModule(
            parquet_file=PARQUET_FILE,
            base_dir=IMAGE_DIR,
            batch_size=2,
            num_workers=0,
            patch_size=64,
            stride=64,
            fold=0,
            n_splits=2,
            fold_file=FOLD_FILE,
        )
        dm.prepare_data()
        dm.setup("fit")

        model = LightningAttriMIL(
            num_classes=len(dm.class_names),
            class_names=dm.class_names,
            patch_size=64,
            stride=64,
            compile_backbone=False,
        )

        ckpt_callback = ModelCheckpoint(
            dirpath=tmpdir,
            filename="test_model",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        logger = CSVLogger(tmpdir, name="test_inference")
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            strategy="auto",
            precision="16-mixed",
            logger=logger,
            callbacks=[ckpt_callback],
            log_every_n_steps=1,
        )
        trainer.fit(model, datamodule=dm)

        ckpt_path = ckpt_callback.best_model_path
        assert os.path.exists(ckpt_path), f"Checkpoint not saved: {ckpt_path}"
        print(f"  Checkpoint saved to {ckpt_path}")

        # 2. Pick a sample image
        df = pd.read_parquet(PARQUET_FILE)
        sample_img = os.path.join(IMAGE_DIR, df.iloc[0]["image_path"])
        assert os.path.exists(sample_img), f"Sample image not found: {sample_img}"

        # 3. Run inference using the inference module functions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model = load_model(ckpt_path, "attrimil", device)
        class_names = loaded_model.class_names

        transform = get_val_transform(image_size=896)
        patches, img_tensor = preprocess_image(sample_img, transform, 64, 64)
        y_prob, A = run_inference_single(loaded_model, patches, "attrimil", device)
        prediction = decode_prediction(y_prob, class_names)

        # 4. Verify predictions
        assert "predicted_class" in prediction, "Missing predicted_class"
        assert prediction["predicted_class"] in class_names, \
            f"Predicted class '{prediction['predicted_class']}' not in {class_names}"
        assert 0 < prediction["confidence"] <= 1.0, \
            f"Confidence {prediction['confidence']} out of range"
        assert len(prediction["top_k"]) > 0, "No top-k predictions"

        print(f"  Prediction: {prediction['predicted_class']} "
              f"(confidence: {prediction['confidence']:.3f})")
        print(f"  Top-k: {prediction['top_k']}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("VISUAL FEATURE DISCOVERY - COMPREHENSIVE TEST SUITE")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Sample data: {SAMPLE_DIR}")
    print("=" * 70)

    total_start = time.time()

    # ── Data loading ──
    run_test("ParquetPrintshopDataset loading", test_parquet_dataset)
    run_test("ParquetPrintshopDataModule setup & iteration", test_parquet_datamodule)
    run_test("BookSplitDataModule (baseline)", test_baseline_datamodule)

    # ── Model forward passes ──
    run_test("AttriMIL forward pass", test_attrimil_forward)
    run_test("AttriMILFeatures forward pass", test_attrimil_features_forward)
    run_test("FlashAttention / OptimizedAttention", test_flash_attention)
    run_test("LightningTIMM forward pass", test_lightning_timm)

    # ── Lightning training steps ──
    run_test("LightningAttriMIL training step", test_lightning_attrimil_training_step)
    run_test("LightningAttriMILFeatures training step", test_lightning_attrimil_features_training_step)
    run_test("LightningTIMM training step", test_lightning_timm_training_step)

    # ── Visualization ──
    run_test("Attention visualization", test_attention_vis)
    run_test("Flash attention visualization", test_flash_attention_vis)

    # ── Data prep ──
    run_test("Generate book-level folds", test_generate_folds_book)
    run_test("Extract backbone features", test_extract_backbone_features)

    # ── Feature extraction pipeline ──
    run_test("Feature extraction config", test_feature_extraction_config)
    run_test("Feature extraction helpers", test_feature_extraction_helpers)
    run_test("Feature extraction schemas", test_feature_extraction_schemas)

    # ── Utilities ──
    run_test("ImageCache", test_cache_manager)
    run_test("BatchAccumulator", test_batch_accumulator)

    # ── Full training smoke tests ──
    run_test("Train AttriMIL 1 epoch", test_train_mil_1epoch)
    run_test("Train baseline ResNet18 1 epoch", test_train_baseline_1epoch)

    # ── Inference ──
    run_test("Inference end-to-end", test_inference_end_to_end)

    # ── Summary ──
    total_elapsed = time.time() - total_start
    print("\n")
    print("=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, status, _, _ in results if status == "PASS")
    failed = sum(1 for _, status, _, _ in results if status == "FAIL")

    for name, status, elapsed, error in results:
        marker = "PASS" if status == "PASS" else "FAIL"
        line = f"  [{marker}] {name} ({elapsed:.1f}s)"
        if error:
            line += f" - {error[:80]}"
        print(line)

    print(f"\nTotal: {len(results)} tests, {passed} passed, {failed} failed")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    if failed > 0:
        print("\nFAILED TESTS:")
        for name, status, _, error in results:
            if status == "FAIL":
                print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)
