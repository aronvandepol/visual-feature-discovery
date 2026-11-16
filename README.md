# Visual Feature Discovery

Code for the article:

van de Pol, A., Prokic, J., & Mol, A. (2026). Interpretable visual feature discovery using multiple instance learning: A case study of colonial Korean print. *Computational Humanities Research*, 2, e22. https://doi.org/10.1017/chr.2026.10036

The trained model weights, dataset metadata, fold definitions, evaluation results and interpretability artifacts are on Zenodo under the concept DOI [`10.5281/zenodo.20015055`](https://doi.org/10.5281/zenodo.20015055).
A mirror of the weights is on Hugging Face at [aronvandepol/visual-feature-discovery](https://huggingface.co/aronvandepol/visual-feature-discovery).

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

The training pipeline works from a parquet file plus a directory of images. The minimal schema the loader expects is five columns:

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

The `book` column is used for book-level GroupKFold splits, ensuring no book appears in both train and test within the same fold.

### Released metadata

The full mapping for the 57,583 page corpus used in the paper is shipped under `release/`:

- `release/dataset_metadata.parquet` (1.5 MB) -- 31 columns, including the canonicalized class label, both raw printshop attributions, full bibliographic metadata, and back-links to the source catalog.
- `release/dataset_metadata.csv.gz` (749 KB) -- the same data as a gzipped CSV.
- `scripts/release/generate_dataset_csv.py` -- the script that produces the above from the local source SQLite metadata database.

Schema (30 columns):

| Group | Columns |
|---|---|
| File identity | `filename`, `image_path`, `original_image_path` |
| Class label | `printshop_canonical` (4 paper classes), `printshop_raw_korean`, `printshop_raw_hanja` (5 raw values), `printshop_db_korean`, `printshop_db_hanja` |
| Book grouping | `book`, `book_id`, `book_id_canonical` |
| Bibliographic | `title_korean`, `title_hanja`, `title_other`, `issue_number`, `total_issue_number`, `frequency`, `category`, `publication_date`, `printing_date`, `page_count_in_volume`, `width_mm`, `height_mm`, `price`, `publisher_korean`, `publisher_hanja`, `printsman_korean`, `printsman_hanja` |
| Source links | `metadata_url`, `images_url` |

#### Printshop canonicalization

The five raw printshop attributions in the source catalog correspond to four printing presses. The two Taedong corporate registrations (`대동인쇄 주식회사 / 大東印刷 株式會社` and `대동인쇄소 / 大東印刷所`) refer to the same press across two stages of corporate registration and share matrices and personnel; the paper merges them into one class. The released CSV preserves both the merged class and the original attribution so the merge is transparent and reversible.

| `printshop_canonical` (paper class) | `printshop_raw_korean` (DB) | `printshop_raw_hanja` (DB) | Pages |
|---|---|---|---|
| `대동인쇄소` | `대동인쇄 주식회사` | `大東印刷 株式會社` | 12,713 |
| `대동인쇄소` | `대동인쇄소` | `大東印刷所` | 12,622 |
| `한성도서 주식회사` | `한성도서 주식회사` | `漢城圖書 株式會社` | 18,438 |
| `신문관` | `신문관` | `新文館` | 7,709 |
| `조선인쇄 주식회사` | `조선인쇄 주식회사` | `朝鮮印刷 株式會社` | 6,101 |

Total: 57,583 pages, 4 canonical classes, 435 unique `book_id_canonical` values. Note that one book title (`장한몽 권1` / *Changhanmong* vol. 1) appears under both Hansŏng Tosŏ (182 pages) and Taedong (181 pages), which is why there are 435 distinct `book_id_canonical` values for 434 distinct book titles.

The loader (`src/data/parquet_datamodule.py`) applies this canonicalization at load time. NFC Unicode normalization is applied first because some parquet writers emit decomposed Hangul (NFD), which would silently fail to match precomposed mapping keys.

#### Source collection

The page images are derived from the digital catalog currently known as Hyundam Mun'go (현담문고). The collection was previously called Adan Mungo (아단문고). At the time of this release, the institute serves catalog records and image archives at the `adanmungo.org` domain, which is the URL scheme used in the `metadata_url` and `images_url` columns. The collection is the same regardless of which name is used to search for it.

The image files themselves are not redistributed in this repository because they are subject to the collection's terms of use. The released metadata CSV gives back-links to each catalog record (`metadata_url`) and to the corresponding image archive directory (`images_url`).

A 10-page sample is included in `sample_data/` for testing the pipeline without needing to obtain the full corpus (see `sample_data/README.md`).

### Using your own data

To apply this pipeline to a different document collection, create a parquet file with the minimal five-column schema. The `printshop` column becomes whatever class you want to distinguish (e.g., scriptoria, publishers, printers). The `book` column should group pages that belong to the same physical unit, so that book-level cross-validation prevents leakage. Place your images in a directory and set `image_path` to the relative path from the base directory. The canonicalization mapping in `src/data/parquet_datamodule.py` can be removed or adapted to your own class merging needs. The patch size should also be reconsidered -- it depends on the scale of the distinguishing features in your material. Our 8x8 pixel patches (on 896x896 resized images) capture sub-character details like stroke terminals and serif weight. Larger patches may be more appropriate for features that span wider regions.

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
│   ├── data/            # Data modules (parquet, feature, baseline)
│   ├── utils/           # Orchestrator, cache manager, batch accumulator
│   └── visualization/   # Attention and GradCAM visualization
├── feature_extraction/  # Feature extraction pipeline, HDBSCAN clustering
├── scripts/
│   ├── training/        # Training scripts for MIL and baselines
│   ├── evaluation/      # Evaluation, ensemble, GradCAM analysis
│   ├── data_prep/       # Fold generation, feature extraction, packing
│   ├── inference/       # Inference pipelines
│   └── release/         # Dataset metadata CSV generator
├── release/             # Released dataset metadata (parquet + gzipped CSV)
├── sample_data/         # 10-page test sample
├── tests/               # Test suite (test_all.py)
└── docs/                # Preprocessing pipeline, evaluation guide
```

## Reproducing the paper

The paper reports five-fold cross-validation results under both page-level (with leakage) and book-level (proper) splits. To reproduce:

### 1. Obtain the dataset

The 57,583 page images are not redistributed in this repository. Download the four printshops' materials from [Hyundam Mun'go](http://www.hyundammungo.org/) (the same collection is also reachable through the older `adanmungo.org` domain). The `release/dataset_metadata.csv.gz` file lists every page with a back-link to the source catalog (`metadata_url`) and to the image archive directory (`images_url`).

Place the images under a local directory matching the structure shown in the Data Format section above. Use `release/dataset_metadata.parquet` (or its CSV equivalent) as the loader's parquet input.

### 2. Generate the fold splits

Both page-level and book-level five-fold splits are shipped in `release/folds_page_5fold.json` and `release/folds_book_5fold.json`.
They can be regenerated with `scripts/data_prep/generate_folds_book.py` and `scripts/data_prep/generate_folds_page.py` using `--random_state 42`.

```bash
uv run python scripts/data_prep/generate_folds_book.py \
    --parquet_file release/dataset_metadata.parquet \
    --output_file release/folds_book_5fold.json \
    --n_splits 5 \
    --random_state 42

uv run python scripts/data_prep/generate_folds_page.py \
    --parquet_file release/dataset_metadata.parquet \
    --output_file release/folds_page_5fold.json \
    --n_splits 5 \
    --random_state 42
```

### 3. Train the AttriMIL models

End-to-end (matches the paper's training):

```bash
# AttriMIL 16x16, fold 0 (repeat for folds 1-4)
python scripts/training/train_mil.py \
    --parquet_file release/dataset_metadata.parquet \
    --base_dir /path/to/images/ \
    --fold_file fold_assignments_book_split.json \
    --fold 0 --n_splits 5 \
    --patch_size 16 --stride 8 \
    --max_epochs 100 --compile_backbone

# AttriMIL 8x8, fold 0
python scripts/training/train_mil.py \
    --parquet_file release/dataset_metadata.parquet \
    --base_dir /path/to/images/ \
    --fold_file fold_assignments_book_split.json \
    --fold 0 --n_splits 5 \
    --patch_size 8 --stride 8 \
    --max_epochs 100 --compile_backbone
```

The two-stage workflow (extract features once, then train head only) is significantly faster for hyperparameter sweeps; see the Training section above.

### 4. Train the baselines

```bash
# ConvNeXt full-page baseline
python scripts/training/train_baseline_book.py \
    --model_name convnextv2_base \
    --parquet_file release/dataset_metadata.parquet \
    --base_dir /path/to/images/ \
    --fold_file fold_assignments_book_split.json \
    --fold 0 --n_folds 5 --max_epochs 100

# Swin S3 cutout baseline
python scripts/training/train_baseline_book.py \
    --model_name swin_s3_base_224 \
    --parquet_file release/dataset_metadata.parquet \
    --base_dir /path/to/images/ \
    --fold_file fold_assignments_book_split.json \
    --fold 0 --n_folds 5 \
    --use_random_crops --num_crops 4 --crop_size 224 \
    --max_epochs 100
```

### 5. Evaluate

```bash
# AttriMIL evaluation (computes accuracy, F1, precision, recall)
python scripts/evaluation/evaluate_mil.py \
    --checkpoint_dir checkpoints/ \
    --parquet_file release/dataset_metadata.parquet \
    --base_dir /path/to/images/ \
    --patch_size 8 --stride 8

# Baseline evaluation
python scripts/evaluation/evaluate_baseline.py \
    --model_name convnextv2_base \
    --model_path models/convnextv2_base.pth \
    --data_dir /path/to/images/
```

Expected book-split metrics (mean across 5 folds):

| Approach | Accuracy | Paper Table |
|---|---|---|
| ConvNeXt full-page | 82.0 ± 2.1% | Table 2 |
| Swin S3 cutout | 84.1 ± 1.1% | Table 2 |
| AttriMIL 16x16 | 85.7% | Table 4 |
| AttriMIL 8x8 | 85.7% | Table 4 |

### 6. Reproduce the interpretability analysis

After training, extract high-attention patches and cluster them:

```bash
# Extract embeddings for high-attention patches
python -m feature_extraction.extract_features \
    --parquet release/dataset_metadata.parquet \
    --base_dir /path/to/images/ \
    --checkpoint checkpoints/fold0/attrimil-flash-best.ckpt \
    --patch_size 8 --stride 8 \
    --output_dir features/

# UMAP + HDBSCAN clustering
python -m feature_extraction.postprocess --input features/patches.parquet
python -m feature_extraction.hdb_run --base_dir features/
```

The clustering uses fixed UMAP and HDBSCAN parameters that are documented in `feature_extraction/config.py`. Run-to-run cluster IDs may differ due to UMAP randomness, but cluster contents (the typographic features they capture) are stable.

### Pretrained checkpoints

Trained checkpoints for AttriMIL 8x8, AttriMIL 16x16, ConvNeXt full-page (held-out-book and page-random), and Swin S3 cutout (held-out-book and page-random) — five folds per configuration — will be deposited on the Zenodo record [`10.5281/zenodo.20015055`](https://doi.org/10.5281/zenodo.20015055) as a new version once the article is in print.
The deposit will also include the precomputed 8x8 patch embeddings and HDBSCAN cluster assignments used for the interpretability analysis.
A download helper will live at `scripts/release/download_weights.py` once the DOI is finalized.

See `release/ZENODO_ASSETS.md` for the full file manifest with sizes and SHA-256 checksums.

### Interpretability artifacts

The 8x8 patch embeddings and cluster assignments used for the paper's UMAP and HDBSCAN analysis are staged as Zenodo-only assets because they exceed normal GitHub file limits. See `release/ZENODO_ASSETS.md` for filenames, sizes, and checksums.

## Documentation

- [docs/PREPROCESSING_PIPELINE.md](docs/PREPROCESSING_PIPELINE.md) - Data preprocessing and augmentation
- [docs/EVALUATION_USAGE.md](docs/EVALUATION_USAGE.md) - Evaluation scripts guide

## License

The source code in this repository is released under the **MIT License**.
See [LICENSE](LICENSE) for details.

The Zenodo deposit uses dual licensing: the source code archive remains under MIT, and all other artifacts (dataset metadata, fold definitions, trained model weights, evaluation results, interpretability artifacts) are released under **CC-BY-4.0**.

The page-image corpus is not redistributed in this repository or on Zenodo and remains governed by the terms of the Hyundam Mun'go collection.
The only images included are the ten sample pages in `sample_data/`.

## Citation

If you use this software, please cite both the article and the Zenodo record:

```
van de Pol, A., Prokic, J., & Mol, A. (2026).
Interpretable visual feature discovery using multiple instance learning: A case study of colonial Korean print.
Computational Humanities Research, 2, e22. https://doi.org/10.1017/chr.2026.10036
```

The Zenodo concept DOI [`10.5281/zenodo.20015055`](https://doi.org/10.5281/zenodo.20015055) is the citable identifier for the software and accompanying data artifacts.
