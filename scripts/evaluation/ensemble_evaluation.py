# ensemble_evaluation.py

import argparse
import os
import sys

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.parquet_datamodule import ParquetPrintshopDataModule  # noqa: E402
from src.models.lightning_attrimil import LightningAttriMIL  # noqa: E402

def load_model(model_path, num_classes, class_names, patch_size, stride):
    model = LightningAttriMIL(
        num_classes=num_classes,
        class_names=class_names,
        patch_size=patch_size,
        stride=stride
    )

    # Load model weights
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        try:
            # Try loading as a checkpoint
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                raise ValueError("Checkpoint does not contain state_dict")
        except Exception as e2:
            raise ValueError(f"Failed to load model: {e2}")

    model.eval()
    return model

def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Load the data module to get the test set
    data_module = ParquetPrintshopDataModule(
        parquet_file=args.parquet_file,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        stride=args.stride,
        fold=0,  # Fold doesn't matter for the test set
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_state,
        fold_file=args.fold_file
    )

    # Prepare the data module to load the test set
    data_module.prepare_data()
    data_module.setup('test')

    # Get class names
    class_names = data_module.class_names
    num_classes = len(class_names)
    print(f"Dataset has {num_classes} classes: {class_names}")

    # Create a test dataloader
    test_loader = data_module.test_dataloader()
    print(f"Test set size: {len(data_module.test_dataset)} samples")

    # Find model files for each fold
    model_files = []
    if args.model_dir:
        # If model_dir is specified, look for models there
        for fold in range(args.n_splits):
            fold_dir = os.path.join(args.model_dir, f"fold{fold}")
            if os.path.exists(fold_dir):
                # Find the best model for this fold
                model_files_in_fold = [f for f in os.listdir(fold_dir) if f.endswith('.pth')]
                if model_files_in_fold:
                    # If multiple models found, prefer ones with 'final' in the name
                    final_models = [f for f in model_files_in_fold if 'final' in f.lower()]
                    if final_models:
                        model_file = os.path.join(fold_dir, final_models[0])
                    else:
                        model_file = os.path.join(fold_dir, model_files_in_fold[0])
                    model_files.append((fold, model_file))
                    print(f"Found model for fold {fold}: {model_file}")
    else:
        # Use explicitly provided model paths
        if args.model_paths:
            for idx, path in enumerate(args.model_paths):
                if os.path.exists(path):
                    model_files.append((idx, path))
                    print(f"Using model path: {path}")

    if not model_files:
        raise ValueError("No model files found! Please specify either --model_dir or --model_paths.")

    # Load models
    models = []
    for fold, model_path in model_files:
        print(f"Loading model for fold {fold} from {model_path}...")
        model = load_model(model_path, num_classes, class_names, args.patch_size, args.stride)
        model = model.to(device)
        models.append(model)

    print(f"Loaded {len(models)} models for ensemble evaluation")

    # Run evaluation
    all_true_labels = []
    all_ensemble_preds = []
    all_fold_preds = [[] for _ in range(len(models))]

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x, y = batch
            x = x.to(device)

            # Collect predictions from all folds
            ensemble_probs = torch.zeros((len(y), num_classes), device=device)

            for i, model in enumerate(models):
                y_prob, _, _, _ = model(x)
                ensemble_probs += y_prob

                # Store individual fold predictions
                fold_preds = y_prob.argmax(dim=1).cpu().numpy()
                all_fold_preds[i].extend(fold_preds)

            # Average predictions across folds
            ensemble_probs /= len(models)
            ensemble_preds = ensemble_probs.argmax(dim=1).cpu().numpy()

            # Store true labels and ensemble predictions
            all_true_labels.extend(y.cpu().numpy())
            all_ensemble_preds.extend(ensemble_preds)

    # Calculate metrics for ensemble
    accuracy = accuracy_score(all_true_labels, all_ensemble_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_ensemble_preds, average='macro')
    cm = confusion_matrix(all_true_labels, all_ensemble_preds)

    print("\nEnsemble Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Calculate metrics for individual folds
    print("\nIndividual Fold Results:")
    fold_metrics = []
    for i, fold_preds in enumerate(all_fold_preds):
        fold_acc = accuracy_score(all_true_labels, fold_preds)
        fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(all_true_labels, fold_preds, average='macro')
        fold_metrics.append((fold_acc, fold_prec, fold_rec, fold_f1))
        print(f"Fold {i}: Accuracy = {fold_acc:.4f}, F1 = {fold_f1:.4f}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': ['Ensemble'] + [f'Fold_{i}' for i in range(len(models))],
        'Accuracy': [accuracy] + [metrics[0] for metrics in fold_metrics],
        'Precision': [precision] + [metrics[1] for metrics in fold_metrics],
        'Recall': [recall] + [metrics[2] for metrics in fold_metrics],
        'F1_Score': [f1] + [metrics[3] for metrics in fold_metrics]
    })

    results_path = os.path.join(args.output_dir, 'ensemble_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Plot and save confusion matrix
    cm_path = os.path.join(args.output_dir, 'ensemble_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble evaluation of k-fold models")
    parser.add_argument('--parquet_file', type=str, required=True,
                        help='Path to the parquet file')
    parser.add_argument('--base_dir', type=str, default="",
                        help='Base directory for image paths if they are relative')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing model checkpoints for each fold')
    parser.add_argument('--model_paths', type=str, nargs='+',
                        help='Explicit paths to model files for each fold')
    parser.add_argument('--output_dir', type=str, default="ensemble_results",
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Size of image patches')
    parser.add_argument('--stride', type=int, default=4,
                        help='Stride for patch extraction')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Proportion of the dataset used as test set')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--fold_file', type=str, default=None,
                        help='Path to a pre-generated fold assignments file')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if GPU is available')

    args = parser.parse_args()
    main(args)
