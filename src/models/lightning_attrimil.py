# src/models/lightning_attrimil.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score
from src.visualization.flash_attention_vis import visualize_flash_attention
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pytorch_lightning.loggers import CometLogger
import os
import io
import numpy as np
from PIL import Image
import warnings
import logging

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='Glyph*missing from font*')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from .attrimil import AttriMIL


class LightningAttriMIL(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        class_names,
        learning_rate=1e-4,
        alpha=0.1,
        beta=0.01,
        patch_size=64,
        stride=32,
        ranking_strategy="hybrid",
        empty_patch_threshold=None,
        compile_backbone=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model with configurable parameters
        self.model = AttriMIL(
            num_classes=num_classes,
            ranking_strategy=ranking_strategy,
            empty_patch_threshold=empty_patch_threshold,
            compile_backbone=compile_backbone,
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.alpha = alpha  # weight for spatial loss
        self.beta = beta  # weight for ranking loss
        self.class_names = class_names
        self.max_viz_samples = 50
        self.viz_samples_count = 0

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.val_recall = Recall(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.val_f1 = F1Score(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.test_recall = Recall(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.test_f1 = F1Score(
            task="multiclass", average="macro", num_classes=num_classes
        )

        # Confusion matrix tracking for test phase
        self.test_predictions = []
        self.test_true_labels = []
        self.test_images = []
        self.test_metadata = []
        self.max_confusion_samples_per_cell = 10

        # Fixed validation samples for consistent attention visualization
        self.fixed_sample_indices = []
        self.fixed_viz_samples_collected = False
        self.current_val_samples = []

        # Track logged steps to prevent duplicates
        self.logged_steps = {}

    def forward(self, x):
        return self.model(x)  # Returns y_prob, y_hat, A, H, extra_info

    def _common_step(self, batch, batch_idx, step_type):
        x, y = batch
        y_prob, y_hat, A, H, extra_info = self(x)  # Fixed: Now gets 5 return values

        cls_loss = self.criterion(y_prob, y)
        spatial_loss = self.model.calculate_spatial_loss(A)
        ranking_loss = self.model.calculate_ranking_loss(
            A, H, y, extra_info
        )  # Fixed: Pass extra_info

        loss = cls_loss + self.alpha * spatial_loss + self.beta * ranking_loss

        self.log(
            f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            f"{step_type}_cls_loss", cls_loss, on_step=True, on_epoch=True, sync_dist=True
        )
        self.log(
            f"{step_type}_spatial_loss",
            spatial_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}_ranking_loss",
            ranking_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss, y_prob, y, A

    def training_step(self, batch, batch_idx):
        loss, y_prob, y, _ = self._common_step(batch, batch_idx, "train")
        self.train_accuracy(y_prob, y)
        self.log(
            "train_accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def on_train_start(self):
        """Log model configuration at start of training"""
        if isinstance(self.logger, CometLogger):
            self.logger.experiment.log_parameter("ranking_strategy", self.model.ranking_strategy)
            self.logger.experiment.log_parameter(
                "empty_patch_thresholds", self.model.empty_patch_threshold
            )
            self.logger.experiment.log_parameter(
                "compile_backbone", self.hparams.compile_backbone
            )
            print(f"Training with ranking strategy: {self.model.ranking_strategy}")

    def on_train_end(self):
        # Save the final model
        if (
            self.logger is not None
            and hasattr(self.logger, "save_dir")
            and self.logger.save_dir is not None
        ):
            model_path = os.path.join(self.logger.save_dir, "final_model.pth")
        else:
            model_path = "final_model.pth"

        torch.save(self.state_dict(), model_path)

        if isinstance(self.logger, CometLogger):
            self.logger.experiment.log_model("final_model", model_path)
        else:
            print(f"Final model saved to {model_path}")

        # Log final confusion matrix if we have predictions
        if self.test_predictions and isinstance(self.logger, CometLogger):
            try:
                # Log final confusion matrix with images
                self.logger.experiment.log_confusion_matrix(
                    y_true=self.test_true_labels,
                    y_predicted=self.test_predictions,
                    images=self.test_images,
                    labels=self.class_names,
                    title="Final Validation Confusion Matrix",
                    max_examples_per_cell=self.max_confusion_samples_per_cell,
                )

                print(
                    f"Logged final confusion matrix with {len(self.test_images)} images to Comet.ml"
                )

            except Exception as e:
                print(f"Failed to log final confusion matrix to Comet: {e}")

    def validation_step(self, batch, batch_idx):
        loss, y_prob, y, A = self._common_step(batch, batch_idx, "val")
        self.val_accuracy(y_prob, y)
        self.val_precision(y_prob, y)
        self.val_recall(y_prob, y)
        self.val_f1(y_prob, y)
        self.log("val_accuracy", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_epoch=True)
        self.log("val_recall", self.val_recall, on_epoch=True)
        self.log("val_f1", self.val_f1, on_epoch=True)

        # Collect current validation samples (temporarily during this epoch)
        self.collect_current_val_samples(batch, y_prob, y, A, batch_idx)

        return loss

    def test_step(self, batch, batch_idx):
        loss, y_prob, y, A = self._common_step(batch, batch_idx, "test")
        self.test_accuracy(y_prob, y)
        self.test_precision(y_prob, y)
        self.test_recall(y_prob, y)
        self.test_f1(y_prob, y)

        self.log("test_accuracy", self.test_accuracy, on_epoch=True)
        self.log("test_precision", self.test_precision, on_epoch=True)
        self.log("test_recall", self.test_recall, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)

        # Collect data for confusion matrix
        self.collect_confusion_matrix_data(batch, y_prob, y, A)

        # Visualize attention for a limited number of samples
        if self.viz_samples_count < self.max_viz_samples:
            self.visualize_attention(batch, y_prob, y, A, batch_idx)

        return loss

    def on_test_epoch_start(self):
        self.viz_samples_count = 0

    def visualize_attention(self, batch, y_prob, y, A, batch_idx):
        x, _ = batch
        batch_size, num_patches_h, num_patches_w, _, _, _ = x.shape

        for i in range(batch_size):
            if self.viz_samples_count >= self.max_viz_samples:
                break

            # Reshape attention for visualization
            attention_2d = A[i].view(num_patches_h, num_patches_w)

            fig = visualize_flash_attention(
                x[i],
                attention_2d,
                y_prob[i],
                y[i],
                self.class_names,
                patch_size=self.hparams.patch_size,
                stride=self.hparams.stride,
                model_type="attrimil",
            )

            # Convert figure to image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)

            # Prepare metadata
            predicted_class = self.class_names[y_prob[i].argmax().item()]
            true_class = self.class_names[y[i].item()]
            is_correct = predicted_class == true_class
            confidence = y_prob[i].max().item()

            # Add attention metrics
            attention_stats = {
                'max_attention': attention_2d.max().item(),
                'min_attention': attention_2d.min().item(),
                'mean_attention': attention_2d.mean().item(),
                'std_attention': attention_2d.std().item(),
                'attention_entropy': self._calculate_attention_entropy(attention_2d),
                'attention_sparsity': self._calculate_attention_sparsity(attention_2d),
            }

            # Log the image to Comet.ml with metadata
            self.logger.experiment.log_image(
                image_data=img_array,
                name=f"attention_vis_test_sample_{self.viz_samples_count}",
                step=self.global_step,
                metadata={
                    "sample_idx": self.viz_samples_count,
                    "predicted_class": predicted_class,
                    "true_class": true_class,
                    "is_correct": str(is_correct),
                    "confidence": float(confidence),
                    **{f'attention_{k}': float(v) for k, v in attention_stats.items()},
                },
            )

            # Close the figure and buffer to free up memory
            plt.close(fig)
            buf.close()

            self.viz_samples_count += 1

    def reconstruct_image_from_patches(self, patches):
        """
        Reconstruct the full image from patches
        Args:
            patches: tensor of shape (num_patches_h, num_patches_w, C, patch_size, patch_size)
        Returns:
            PIL Image of the reconstructed image
        """
        num_patches_h, num_patches_w, C, patch_size, _ = patches.shape

        # Calculate output image dimensions
        # Note: This assumes patches tile perfectly without overlap for reconstruction
        img_h = num_patches_h * patch_size
        img_w = num_patches_w * patch_size

        # Initialize output image tensor
        reconstructed = torch.zeros(C, img_h, img_w)

        # Place each patch in its correct position
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y_start = i * patch_size
                y_end = y_start + patch_size
                x_start = j * patch_size
                x_end = x_start + patch_size

                reconstructed[:, y_start:y_end, x_start:x_end] = patches[i, j]

        # Convert to PIL Image
        # Denormalize (reverse the normalization applied in preprocessing)
        reconstructed = reconstructed * 0.229 + 0.485  # Reverse normalization
        reconstructed = torch.clamp(reconstructed, 0, 1)

        # Convert to numpy and then PIL
        if C == 1:  # Grayscale
            img_np = reconstructed.squeeze(0).numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), mode="L")
        else:  # RGB
            img_np = reconstructed.permute(1, 2, 0).numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), mode="RGB")

        return img_pil

    def collect_current_val_samples(self, batch, y_prob, y, A, batch_idx):
        """
        Collect current validation samples temporarily (will be discarded after epoch)
        Only stores tensor data for samples we actually need to visualize
        """
        x, _ = batch
        batch_size = len(y)

        if not self.fixed_viz_samples_collected:
            # First epoch: collect first N samples to establish fixed indices
            samples_needed = self.max_viz_samples - len(self.current_val_samples)
            if samples_needed <= 0:
                return  # Already have enough samples

            # Only store as many samples as we still need
            samples_to_store = min(batch_size, samples_needed)

            for i in range(samples_to_store):
                sample_data = {
                    "x": x[i].clone().detach(),  # Keep on GPU for immediate use
                    "y_prob": y_prob[i].clone().detach(),
                    "y": y[i].clone().detach(),
                    "A": A[i].clone().detach(),
                    "batch_idx": batch_idx,
                    "sample_in_batch": i,
                    "global_idx": len(self.current_val_samples),
                }
                self.current_val_samples.append(sample_data)

            # If we now have enough samples, establish fixed indices
            if len(self.current_val_samples) >= self.max_viz_samples:
                for i in range(self.max_viz_samples):
                    sample = self.current_val_samples[i]
                    index_data = {
                        "batch_idx": sample["batch_idx"],
                        "sample_in_batch": sample["sample_in_batch"],
                        "true_label": sample["y"].item(),
                        "visualization_idx": i,
                    }
                    self.fixed_sample_indices.append(index_data)

                self.fixed_viz_samples_collected = True
                print(
                    f"Selected {len(self.fixed_sample_indices)} fixed sample indices for consistent visualization"
                )

        else:
            # Subsequent epochs: only store data for samples matching our fixed indices
            for i in range(batch_size):
                # Check if this sample matches any of our fixed indices
                for index_data in self.fixed_sample_indices:
                    if index_data["batch_idx"] == batch_idx and index_data[
                        "sample_in_batch"
                    ] == i:
                        sample_data = {
                            "x": x[i].clone().detach(),
                            "y_prob": y_prob[i].clone().detach(),
                            "y": y[i].clone().detach(),
                            "A": A[i].clone().detach(),
                            "batch_idx": batch_idx,
                            "sample_in_batch": i,
                            "global_idx": len(self.current_val_samples),
                        }
                        self.current_val_samples.append(sample_data)
                        break  # Found match, no need to check other indices

    def collect_confusion_matrix_data(self, batch, y_prob, y, A):
        """
        Collect data needed for confusion matrix logging
        """
        x, _ = batch
        batch_size = len(y)

        # Get predictions
        predictions = y_prob.argmax(dim=1).cpu().numpy()
        true_labels = y.cpu().numpy()
        probabilities = torch.softmax(y_prob, dim=1).cpu().numpy()

        for i in range(batch_size):
            pred_idx = predictions[i]
            true_idx = true_labels[i]
            confidence = probabilities[i][pred_idx]

            # Reconstruct image from patches
            try:
                reconstructed_img = self.reconstruct_image_from_patches(x[i].cpu())

                # Convert PIL image to numpy array for Comet
                img_array = np.array(reconstructed_img)

                # Store the data
                self.test_predictions.append(pred_idx)
                self.test_true_labels.append(true_idx)
                self.test_images.append(img_array)

                # Store metadata
                metadata = {
                    "predicted_class": self.class_names[pred_idx],
                    "true_class": self.class_names[true_idx],
                    "confidence": float(confidence),
                    "is_correct": pred_idx == true_idx,
                    "sample_index": len(self.test_predictions) - 1,
                }
                self.test_metadata.append(metadata)

            except Exception as e:
                print(f"Warning: Failed to reconstruct image for confusion matrix: {e}")
                continue

    def on_validation_epoch_start(self):
        """Reset visualization counter and clear temporary samples"""
        self.viz_samples_count = 0
        self.current_val_samples = []  # Clear temporary samples from previous epoch

    def on_validation_epoch_end(self):
        """
        Log fixed attention visualizations to Comet at the end of validation epoch
        """
        if not isinstance(self.logger, CometLogger):
            return

        # Skip logging during sanity check (initial validation before training)
        if hasattr(self.trainer, "sanity_checking") and self.trainer.sanity_checking:
            print(f"Skipping attention visualization - sanity checking mode")
            return

        # Log fixed attention visualizations using current epoch data
        if self.fixed_viz_samples_collected and self.current_val_samples:
            self.visualize_fixed_samples()

        # Clear temporary validation samples to free memory
        self.current_val_samples = []

    def visualize_fixed_samples(self):
        """
        Visualize attention for the fixed validation samples using current epoch data
        """
        if not self.fixed_sample_indices or not isinstance(self.logger, CometLogger):
            return

        print(
            f"Visualizing attention for {len(self.fixed_sample_indices)} fixed samples - Epoch {self.current_epoch}"
        )

        # Find the samples from current validation epoch that match our fixed indices
        for index_data in self.fixed_sample_indices:
            try:
                target_batch_idx = index_data["batch_idx"]
                target_sample_in_batch = index_data["sample_in_batch"]
                viz_idx = index_data["visualization_idx"]

                # Find matching sample in current validation samples
                matching_sample = None
                for sample in self.current_val_samples:
                    if (
                        sample["batch_idx"] == target_batch_idx
                        and sample["sample_in_batch"] == target_sample_in_batch
                    ):
                        matching_sample = sample
                        break

                if matching_sample is None:
                    print(f"Warning: Could not find matching sample for viz_idx {viz_idx}")
                    continue

                # Get current sample data (already on GPU)
                x = matching_sample["x"]
                y_prob = matching_sample["y_prob"]
                y = matching_sample["y"]
                A = matching_sample["A"]

                # Check if we already logged this image at this step (prevent duplicates)
                image_name = f"attention_vis_sample_{viz_idx:02d}"
                current_step = self.current_epoch

                if image_name in self.logged_steps and current_step in self.logged_steps[
                    image_name
                ]:
                    print(f"Skipping duplicate logging for {image_name} at step {current_step}")
                    continue

                # Get patch dimensions
                num_patches_h = int(np.sqrt(A.numel()))
                num_patches_w = num_patches_h

                # Generate visualization with suppressed warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig = visualize_flash_attention(
                        x,  # Single image (no batch dimension)
                        A.view(num_patches_h, num_patches_w),
                        y_prob,
                        y,  # Use tensor directly
                        self.class_names,
                        patch_size=self.hparams.patch_size,
                        stride=self.hparams.stride,
                        model_type="attrimil",
                    )

                # Convert figure to image
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                img = Image.open(buf)
                img_array = np.array(img)

                # Prepare metadata
                predicted_class = self.class_names[y_prob.argmax().item()]
                true_class = self.class_names[y.item()]
                is_correct = predicted_class == true_class
                confidence = y_prob.max().item()

                # Log with consistent naming and step for time tracking
                print(
                    f"Logging image: {image_name}, step: {current_step}, epoch: {self.current_epoch}"
                )

                self.logger.experiment.log_image(
                    image_data=img_array,
                    name=image_name,  # Consistent naming
                    step=current_step,  # Use epoch as step for time progression
                    overwrite=False,  # Don't overwrite previous steps
                    metadata={
                        "epoch": self.current_epoch,
                        "sample_idx": viz_idx,
                        "predicted_class": predicted_class,
                        "true_class": true_class,
                        "is_correct": str(is_correct),
                        "confidence": float(confidence),
                    },
                )

                # Track that we logged this step
                if image_name not in self.logged_steps:
                    self.logged_steps[image_name] = set()
                self.logged_steps[image_name].add(current_step)

                # Close the figure and buffer to free up memory
                plt.close(fig)
                buf.close()

            except Exception as e:
                print(f"Failed to visualize attention for sample {viz_idx}: {e}")
                continue

        # Clear GPU cache after visualization to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Completed attention visualization for epoch {self.current_epoch}")

    def on_test_epoch_end(self):
        """
        Log confusion matrix with images to Comet at the end of test epoch
        """
        if not self.test_predictions or not isinstance(self.logger, CometLogger):
            return

        try:
            # Log confusion matrix with images
            self.logger.experiment.log_confusion_matrix(
                y_true=self.test_true_labels,
                y_predicted=self.test_predictions,
                images=self.test_images,
                labels=self.class_names,
                title="Test Set Confusion Matrix",
                max_examples_per_cell=self.max_confusion_samples_per_cell,
            )

            print(
                f"Logged test confusion matrix with {len(self.test_images)} images to Comet.ml"
            )

        except Exception as e:
            print(f"Failed to log confusion matrix to Comet: {e}")

    def _calculate_attention_entropy(self, attention_weights):
        """Calculate entropy of attention distribution."""
        attention_flat = attention_weights.flatten()
        probs = torch.softmax(attention_flat, dim=0)
        probs = probs + 1e-8
        entropy = -torch.sum(probs * torch.log(probs))
        return entropy.item()

    def _calculate_attention_sparsity(self, attention_weights, threshold=0.01):
        """Calculate sparsity (percentage of weights below threshold)."""
        below_threshold = (attention_weights < threshold).float().mean()
        return below_threshold.item()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # optimizer = optim.Shampoo(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }