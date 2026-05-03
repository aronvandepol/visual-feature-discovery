"""
Unified inference script for document classification models.

Supports 3 input modes:
  --image         Single image file
  --image_dir     Directory of images
  --parquet_file  Parquet dataset (with --base_dir for image root)

Supports model variants:
  --model_type attrimil          AttriMIL (returns 5 values)
  --model_type attrimil_features Pre-extracted features (returns 4 values)

Uses the same preprocessing as training (grayscale + CLAHE + single-channel normalize).
"""

import argparse
import glob as glob_module
import json
import os
import sys

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


# ── Model loading ─────────────────────────────────────────────────────

MODEL_CLASSES = {
    "attrimil": "src.models.lightning_attrimil.LightningAttriMIL",
    "attrimil_features": "src.models.lightning_attrimil_features.LightningAttriMILFeatures",
}


def load_model(checkpoint_path, model_type, device):
    """Load a Lightning model from checkpoint."""
    if model_type == "attrimil":
        from src.models.lightning_attrimil import LightningAttriMIL as ModelClass
    elif model_type == "attrimil_features":
        from src.models.lightning_attrimil_features import LightningAttriMILFeatures as ModelClass
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = ModelClass.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    return model


# ── Preprocessing ─────────────────────────────────────────────────────

def get_val_transform(image_size=896):
    """Same transform used during validation/test in training."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.ToGray(p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])


def preprocess_image(image_path, transform, patch_size, stride):
    """
    Load an image, apply the transform, and extract patches.

    Returns:
        patches: Tensor (1, n_h, n_w, C, patch_size, patch_size) ready for model
        img_tensor: The full transformed image tensor (C, H, W)
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    augmented = transform(image=img)
    img_tensor = augmented["image"]  # (C, H, W) — single channel after ToGray

    # Extract patches via unfold
    patches = img_tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    # patches shape: (C, n_h, n_w, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    # patches shape: (n_h, n_w, C, patch_size, patch_size)

    return patches.unsqueeze(0), img_tensor  # add batch dim


# ── Inference ─────────────────────────────────────────────────────────

def run_inference_single(model, patches, model_type, device):
    """
    Run model forward pass and return probabilities and attention weights.

    Returns:
        y_prob: (num_classes,) tensor of class probabilities
        A: attention weights tensor (shape varies by model)
    """
    patches = patches.to(device)
    with torch.no_grad():
        if model_type == "attrimil":
            y_prob, y_hat, A, H, extra_info = model(patches)
        else:
            # attrimil_features returns 4 values
            y_prob, y_hat, A, H = model(patches)

    return y_prob[0], A[0] if A is not None else None


def decode_prediction(y_prob, class_names, top_k=3):
    """Decode model output into prediction dict."""
    probs = torch.softmax(y_prob, dim=0)
    predicted_idx = probs.argmax().item()
    confidence = probs[predicted_idx].item()

    top_k = min(top_k, len(class_names))
    topk_vals, topk_idxs = torch.topk(probs, top_k)

    top_classes = []
    for val, idx in zip(topk_vals, topk_idxs):
        top_classes.append({
            "class": class_names[idx.item()],
            "probability": round(val.item(), 4),
        })

    return {
        "predicted_class": class_names[predicted_idx],
        "confidence": round(confidence, 4),
        "top_k": top_classes,
    }


# ── Visualization ─────────────────────────────────────────────────────

def save_attention_visualization(patches, A, y_prob, class_names, model_type,
                                  patch_size, stride, output_path):
    """Save attention visualization PNG for a single image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Get 2D patch grid dimensions from the input patches (without batch dim)
    if patches.ndim == 6:
        patches = patches[0]  # remove batch dim
    n_h, n_w = patches.shape[0], patches.shape[1]

    from src.visualization.flash_attention_vis import visualize_flash_attention
    A_2d = A.view(n_h, n_w)
    y_true = torch.tensor(0)  # placeholder for inference
    fig = visualize_flash_attention(patches, A_2d, y_prob, y_true, class_names,
                                    patch_size=patch_size, stride=stride,
                                    model_type=model_type)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Input collection ──────────────────────────────────────────────────

def collect_image_paths(args):
    """Collect image paths from the chosen input mode."""
    if args.image:
        return [args.image]
    elif args.image_dir:
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
        paths = []
        for ext in extensions:
            paths.extend(glob_module.glob(os.path.join(args.image_dir, ext)))
            paths.extend(glob_module.glob(os.path.join(args.image_dir, ext.upper())))
        paths = sorted(set(paths))
        if not paths:
            raise FileNotFoundError(f"No images found in {args.image_dir}")
        return paths
    elif args.parquet_file:
        df = pd.read_parquet(args.parquet_file)
        base = args.base_dir or ""
        paths = [os.path.join(base, p) for p in df["image_path"]]
        return paths
    else:
        raise ValueError("Provide one of --image, --image_dir, or --parquet_file")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained document classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python inference.py --checkpoint model.ckpt --image page.jpg

  # Directory of images
  python inference.py --checkpoint model.ckpt --image_dir ./pages/

  # Parquet dataset
  python inference.py --checkpoint model.ckpt --parquet_file data.parquet --base_dir ./images/

  # With attention visualization
  python inference.py --checkpoint model.ckpt --image page.jpg --visualize --output_dir results/
        """,
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to a single image")
    input_group.add_argument("--image_dir", type=str, help="Directory of images")
    input_group.add_argument("--parquet_file", type=str, help="Path to parquet dataset")

    # Parquet-specific
    parser.add_argument("--base_dir", type=str, default="",
                        help="Base directory for image paths in parquet file")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--model_type", type=str, default="attrimil",
                        choices=["attrimil", "attrimil_features"],
                        help="Model architecture (default: attrimil)")

    # Preprocessing
    parser.add_argument("--image_size", type=int, default=896,
                        help="Image resize dimension (default: 896)")
    parser.add_argument("--patch_size", type=int, default=8,
                        help="Patch size for patch extraction (default: 8)")
    parser.add_argument("--stride", type=int, default=4,
                        help="Stride for patch extraction (default: 4)")

    # Output
    parser.add_argument("--output_dir", type=str, default="inference_output",
                        help="Directory for output files (default: inference_output)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save attention visualization PNGs")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available, else cpu)")

    args = parser.parse_args()

    # Resolve device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    print(f"Loading {args.model_type} model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.model_type, device)
    class_names = model.class_names
    print(f"Classes ({len(class_names)}): {class_names}")

    # Prepare transform
    transform = get_val_transform(args.image_size)

    # Collect images
    image_paths = collect_image_paths(args)
    print(f"Processing {len(image_paths)} image(s)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    all_results = []
    for i, img_path in enumerate(image_paths):
        try:
            patches, img_tensor = preprocess_image(
                img_path, transform, args.patch_size, args.stride
            )
            y_prob, A = run_inference_single(model, patches, args.model_type, device)
            prediction = decode_prediction(y_prob, class_names)
            prediction["image"] = img_path

            all_results.append(prediction)

            print(f"  [{i+1}/{len(image_paths)}] {os.path.basename(img_path)}: "
                  f"{prediction['predicted_class']} ({prediction['confidence']:.3f})")

            # Optional attention visualization
            if args.visualize and A is not None and args.model_type != "attrimil_features":
                vis_name = os.path.splitext(os.path.basename(img_path))[0] + "_attention.png"
                vis_path = os.path.join(args.output_dir, vis_name)
                save_attention_visualization(
                    patches.cpu(), A.cpu(), y_prob.cpu(), class_names,
                    args.model_type, args.patch_size, args.stride, vis_path
                )

        except Exception as e:
            print(f"  [{i+1}/{len(image_paths)}] ERROR {os.path.basename(img_path)}: {e}")
            all_results.append({"image": img_path, "error": str(e)})

    # Save JSON results
    results_path = os.path.join(args.output_dir, "predictions.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # Print summary
    successful = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]
    print(f"\nSummary: {len(successful)} successful, {len(errors)} errors")
    if successful:
        from collections import Counter
        class_counts = Counter(r["predicted_class"] for r in successful)
        print("Class distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
