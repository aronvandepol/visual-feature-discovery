# Visual Feature Discovery

Code for "Interpretable Visual Feature Discovery using Multiple Instance Learning: A Case Study of Colonial Korean Print" (van de Pol, Prokic, and Mol, Computational Humanities Research).

## Overview

Classification of 57,583 page images from four colonial Korean printshops (1910--1945) in Kyongsong (Seoul). Three approaches are compared:

- **ConvNeXt (full page)**: CNN baseline on full page images
- **Swin (cutout)**: Vision transformer on text-block cutouts
- **AttriMIL (Multiple Instance Learning)**: Attention-based MIL that processes pages as bags of patches

### AttriMIL Architecture

AttriMIL treats each page as a bag of small patches (16x16 or 8x8 pixels). A frozen ResNet18 backbone extracts features from each patch, learnable adapter layers pass these through an attention mechanism that assigns importance scores, and a classifier predicts the printshop based on the attended features.

Because the backbone is frozen, it produces identical output for the same input every time. This means that in the end-to-end pipeline, a large portion of each training epoch is spent redundantly recomputing features that never change. At the 8x8 patch configuration this amounts to 49,729 patches per image across 57,583 images -- every epoch.

To avoid this, the repository provides a two-stage workflow. First, a one-time extraction script (`scripts/data_prep/extract_backbone_features.py`) runs the frozen backbone over the entire dataset and saves the resulting feature vectors to disk. Then a separate training script (`scripts/training/train_mil_features.py`) trains only the attention head and classifier on these pre-extracted features, bypassing the backbone entirely. This is significantly faster for hyperparameter tuning and experimentation, since each epoch only needs to load feature vectors from disk rather than run a CNN forward pass on every patch.

Both workflows produce equivalent models -- the attention head and classifier are identical. The end-to-end script (`scripts/training/train_mil.py`) is simpler to run and is used for final training and inference. The pre-extracted features workflow is used when iterating quickly on model design, and also feeds into the UMAP/HDBSCAN clustering pipeline for interpretability analysis.

The four printshops:
- Taedong Inswaeso (大東印刷所) -- includes 대동인쇄소 and 대동인쇄 주식회사, treated as the same shop
- Hansong Toso Chusik Hoesa (漢城圖書株式會社)
- Sinmungwan (新文館)
- Choson Inswae Chusik Hoesa (朝鮮印刷株式會社)

Page images are sourced from the Hyundam Mun'go digitization project.

## Installation

Requires Python >= 3.10 and PyTorch >= 2.5.1 with CUDA.

```bash
# Clone the repository
git clone <repo-url>
cd visual-feature-discovery

# Install with pip
pip install -e .

# Or with uv
uv pip install -e .

# With Comet ML experiment tracking
pip install -e ".[comet]"

# With GPU-accelerated UMAP/HDBSCAN
pip install -e ".[rapids]"
```

## Data Format

The dataset is defined by a parquet file and a directory of images. The parquet file must contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | Original filename, same as basename of `image_path` |
| `image_path` | str | Relative path to the image inside the base directory |
| `printshop` | str | Printshop (class) name |
| `book` | str | Book title -- all pages from the same physical book share this value |
| `book_id` | str | Unique book identifier (`{printshop}_{book}`) |

Images are organized in subdirectories by printshop:

```
dataset/
├── 대동인쇄소/
│   ├── page1.jpg
│   └── ...
├── 한성도서 주식회사/
├── 신문관/
└── 조선인쇄 주식회사/
```

The `book` column is used for book-level GroupKFold splits, ensuring no book appears in both train and test within the same fold. The two Taedong variants (대동인쇄소 and 대동인쇄 주식회사) are automatically canonicalized to a single class.

A small 10-page sample dataset is included in `sample_data/` for testing (see `sample_data/README.md`).

### Using your own data

To apply this pipeline to a different document collection, create a parquet file with the same schema. The `printshop` column becomes whatever class you want to distinguish (e.g., scriptoria, publishers, printers). The `book` column should group pages that belong to the same physical unit, so that book-level cross-validation prevents leakage. Place your images in a directory and set `image_path` to the relative path from the base directory. The canonicalization mapping in `src/data/parquet_dataset.py` can be removed or adapted to your own class merging needs. The patch size should also be reconsidered -- it depends on the scale of the distinguishing features in your material. Our 8x8 pixel patches (on 896x896 resized images) capture sub-character details like stroke terminals and serif weight. Larger patches may be more appropriate for features that span wider regions.

## Setup

1. **Dataset**: Set environment variables or pass paths directly to scripts:
   ```bash
   export DATASET_BASE_DIR=/path/to/dataset/
   export PARQUET_FILE=/path/to/dataset.parquet
   ```

2. **Comet ML** (optional): For experiment tracking, set your API key:
   ```bash
   export COMET_API_KEY=your_api_key_here
   ```
   Or copy `constants.json.example` to `constants.json` and fill in your key.
   If no API key is set, training scripts fall back to local CSV logging.

## Usage

All training and evaluation scripts are standard Python scripts that can be run on any machine with a GPU. No HPC cluster or job scheduler is required.

### Training AttriMIL

There are two ways to train AttriMIL. The end-to-end script runs the full pipeline (images -> backbone -> attention -> classifier) in a single command:

```bash
python scripts/training/train_mil.py \
    --parquet_file /path/to/dataset.parquet \
    --base_dir /path/to/dataset/ \
    --fold 0 \
    --n_splits 5 \
    --patch_size 8 \
    --stride 4 \
    --max_epochs 100 \
    --compile_backbone
```

To log to Comet ML, add `--comet_project_name amil-kfold --experiment_name AMIL_book_split`.

Use `--no-compile-backbone` to disable `torch.compile` on the frozen backbone stages (useful for debugging or multi-GPU DDP compatibility).

Because the ResNet18 backbone is frozen, it produces identical features every epoch. For faster training, you can pre-extract the backbone features once and then train only the attention head:

```bash
# Step 1: Extract backbone features (one-time)
python scripts/data_prep/extract_backbone_features.py \
    --parquet_file /path/to/dataset.parquet \
    --base_dir /path/to/dataset/ \
    --output_dir features/ \
    --patch_size 8 --stride 4

# Step 2: Train on pre-extracted features
python scripts/training/train_mil_features.py \
    --feature_dir features/ \
    --parquet_file /path/to/dataset.parquet \
    --fold 0 \
    --n_splits 5 \
    --max_epochs 100
```

This skips the backbone entirely during training, which is significantly faster when iterating on the attention mechanism or hyperparameters.

Both scripts save checkpoints to `checkpoints/fold{N}/` by default.

### Training Baselines (ConvNeXt / Swin)

```bash
# ConvNeXt (full page)
python scripts/training/train_baseline_book.py \
    --model_name convnextv2_base \
    --data_dir /path/to/dataset/ \
    --fold 0 --n_folds 5 \
    --max_epochs 100

# Swin (cutout)
python scripts/training/train_baseline_book.py \
    --model_name swin_s3_base_224 \
    --data_dir /path/to/dataset/ \
    --fold 0 --n_folds 5 \
    --use_random_crops --num_crops 4 --crop_size 224 \
    --max_epochs 100
```

### Evaluation

```bash
# Evaluate MIL model
python scripts/evaluation/evaluate_mil.py \
    --checkpoint_dir checkpoints/ \
    --parquet_file /path/to/dataset.parquet \
    --base_dir /path/to/dataset/ \
    --patch_size 8 --stride 4

# Evaluate baseline models
python scripts/evaluation/evaluate_baseline.py \
    --model_name convnextv2_base \
    --model_path models/convnextv2_base.pth \
    --data_dir /path/to/dataset/
```

### Clustering Pipeline

After training, high-attention patches can be clustered to inspect what the model treats as visually similar:

```bash
# UMAP + t-SNE reduction
python -m feature_extraction.postprocess --input features/patches.parquet

# HDBSCAN clustering
python -m feature_extraction.hdb_run --base_dir features/
```

## Project Structure

```
visual-feature-discovery/
├── src/
│   ├── models/          # AttriMIL, Lightning wrappers, TIMM wrapper
│   ├── data/            # Data modules (parquet, directory, feature, baseline)
│   ├── utils/           # Orchestrator, cache manager, batch accumulator
│   └── visualization/   # Attention and GradCAM visualization
├── feature_extraction/  # Feature extraction pipeline, HDBSCAN clustering
├── scripts/
│   ├── training/        # Training scripts for MIL and baselines
│   ├── evaluation/      # Evaluation, ensemble, GradCAM analysis
│   ├── data_prep/       # Fold generation, feature extraction, packing
│   └── inference/       # Inference pipelines
└── docs/                # Preprocessing pipeline, evaluation guide
```

## Documentation

- [docs/PREPROCESSING_PIPELINE.md](docs/PREPROCESSING_PIPELINE.md) - Data preprocessing and augmentation
- [docs/EVALUATION_USAGE.md](docs/EVALUATION_USAGE.md) - Evaluation scripts guide

## License

MIT License. See [LICENSE](LICENSE) for details.
