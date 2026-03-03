"""
Lightning module wrapping AttriMILFeatures for pre-extracted feature training.

Nearly identical structure to lightning_attrimil.py:
- Same _common_step, training_step, validation_step, test_step
- Same metrics (accuracy, precision, recall, F1)
- Same optimizer config (Adam + ReduceLROnPlateau)
- Same Comet logging
- Simplified: no patch reconstruction visualization (no raw pixels available)
"""

import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1Score
from pytorch_lightning.loggers import CometLogger

from .attrimil_features import AttriMILFeatures


class LightningAttriMILFeatures(pl.LightningModule):
    """Lightning module for feature-based AttriMIL."""

    def __init__(self, num_classes, class_names, feature_dim=256, learning_rate=1e-4,
                 alpha=0.1, beta=0.01):
        super().__init__()
        self.save_hyperparameters()

        self.model = AttriMILFeatures(
            num_classes=num_classes,
            feature_dim=feature_dim,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta
        self.class_names = class_names

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", average='macro', num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", average='macro', num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", average='macro', num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", average='macro', num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", average='macro', num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", average='macro', num_classes=num_classes)

        # Confusion matrix tracking for test
        self.test_predictions = []
        self.test_true_labels = []

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, step_type):
        features, y = batch
        y_prob, y_hat, A, H = self(features)

        cls_loss = self.criterion(y_prob, y)
        spatial_loss = self.model.calculate_spatial_loss(A)
        ranking_loss = self.model.calculate_ranking_loss(A, H, y)

        loss = cls_loss + self.alpha * spatial_loss + self.beta * ranking_loss

        self.log(f'{step_type}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{step_type}_cls_loss', cls_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{step_type}_spatial_loss', spatial_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{step_type}_ranking_loss', ranking_loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss, y_prob, y, A

    def training_step(self, batch, batch_idx):
        loss, y_prob, y, _ = self._common_step(batch, batch_idx, "train")
        self.train_accuracy(y_prob, y)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_end(self):
        if self.logger is not None and hasattr(self.logger, 'save_dir') and self.logger.save_dir is not None:
            model_path = os.path.join(self.logger.save_dir, 'final_model_features.pth')
        else:
            model_path = 'final_model_features.pth'

        torch.save(self.state_dict(), model_path)

        if isinstance(self.logger, CometLogger):
            self.logger.experiment.log_model('final_model_features', model_path)
        else:
            print(f"Final feature model saved to {model_path}")

    def validation_step(self, batch, batch_idx):
        loss, y_prob, y, _ = self._common_step(batch, batch_idx, "val")
        self.val_accuracy(y_prob, y)
        self.val_precision(y_prob, y)
        self.val_recall(y_prob, y)
        self.val_f1(y_prob, y)
        self.log('val_accuracy', self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_epoch=True)
        self.log('val_recall', self.val_recall, on_epoch=True)
        self.log('val_f1', self.val_f1, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_prob, y, A = self._common_step(batch, batch_idx, "test")
        self.test_accuracy(y_prob, y)
        self.test_precision(y_prob, y)
        self.test_recall(y_prob, y)
        self.test_f1(y_prob, y)

        self.log('test_accuracy', self.test_accuracy, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)

        # Collect predictions for confusion matrix
        predictions = y_prob.argmax(dim=1).cpu().numpy()
        true_labels = y.cpu().numpy()
        self.test_predictions.extend(predictions.tolist())
        self.test_true_labels.extend(true_labels.tolist())

        return loss

    def on_test_epoch_end(self):
        if not self.test_predictions or not isinstance(self.logger, CometLogger):
            return

        try:
            self.logger.experiment.log_confusion_matrix(
                y_true=self.test_true_labels,
                y_predicted=self.test_predictions,
                labels=self.class_names,
                title="Feature AttriMIL Test Confusion Matrix",
            )
            print(f"Logged confusion matrix with {len(self.test_predictions)} samples")
        except Exception as e:
            print(f"Failed to log confusion matrix: {e}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
