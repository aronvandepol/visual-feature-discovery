# Evaluation Guide

## Evaluate a Single Fold

```bash
python scripts/evaluation/evaluate_mil.py \
    --checkpoint_dir checkpoints/ \
    --parquet_file /path/to/dataset.parquet \
    --base_dir /path/to/dataset/ \
    --fold 0 \
    --patch_size 16 \
    --stride 8 \
    --use_gpu
```

## Evaluate All Folds

```bash
python scripts/evaluation/evaluate_mil.py \
    --checkpoint_dir checkpoints/ \
    --parquet_file /path/to/dataset.parquet \
    --base_dir /path/to/dataset/ \
    --evaluate_all_folds \
    --patch_size 16 \
    --stride 8 \
    --n_splits 5 \
    --use_gpu
```

## Evaluate Baselines

```bash
python scripts/evaluation/evaluate_baseline.py \
    --model_name convnextv2_base \
    --model_path models/convnextv2_base.pth \
    --data_dir /path/to/dataset/
```

## Parameters

### Required
- `--checkpoint_dir`: Directory containing fold checkpoint subdirectories
- `--parquet_file`: Path to dataset parquet file

### Model (must match training)
- `--patch_size`: Patch size used during training (default: 8)
- `--stride`: Stride used during training (default: 4)
- `--n_splits`: Number of cross-validation folds (default: 5)
- `--random_state`: Random seed used during training (default: 42)

### Optional
- `--fold`: Specific fold to evaluate (0 to n_splits-1)
- `--evaluate_all_folds`: Evaluate all folds
- `--batch_size`: Batch size for evaluation (default: 16)
- `--use_gpu`: Use GPU acceleration

## Output

Results are saved to a timestamped directory:

```
evaluation_results_AMIL_16_8_20241211_143022/
├── AMIL_16_8_fold0_metrics.txt
├── AMIL_16_8_fold0_confusion_matrix.png
├── AMIL_16_8_fold1_metrics.txt
├── AMIL_16_8_fold1_confusion_matrix.png
├── ...
└── AMIL_16_8_cv_summary.txt
```

Each fold produces a metrics report (accuracy, precision, recall, F1 per class and macro/weighted) and a confusion matrix plot. When evaluating all folds, a cross-validation summary with mean and standard deviation across folds is also generated.
