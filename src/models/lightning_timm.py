# src/models/lightning_timm.py

import pytorch_lightning as pl
import torch
import timm
from torchmetrics import Accuracy, Precision, Recall, F1Score

class LightningTIMM(pl.LightningModule):
    def __init__(self, num_classes, model_name='convnext_base', learning_rate=1e-4, class_names=None):
        super().__init__()
        self.save_hyperparameters()
        
        # Store class names
        self.class_names = class_names if class_names else [str(i) for i in range(num_classes)]
        
        # Initialize model from timm
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", average='macro', num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", average='macro', num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", average='macro', num_classes=num_classes)
        
        # Store test predictions
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, step_type):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log(f'{step_type}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx, "train")
        self.train_accuracy(logits, y)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx, "val")
        self.val_accuracy(logits, y)
        self.log('val_accuracy', self.val_accuracy, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx, "test")
        
        # Calculate metrics
        self.test_accuracy(logits, y)
        self.test_precision(logits, y)
        self.test_recall(logits, y)
        self.test_f1(logits, y)
        
        # Store predictions for confusion matrix
        preds = torch.argmax(logits, dim=1)
        self.test_preds.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        
        # Log metrics
        self.log('test_accuracy', self.test_accuracy, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        
        return loss

    def on_test_epoch_end(self):
        # Log confusion matrix to Comet
        if self.logger and hasattr(self.logger.experiment, 'log_confusion_matrix'):
            self.logger.experiment.log_confusion_matrix(
                y_true=self.test_targets,
                y_predicted=self.test_preds,
                labels=self.class_names,
                title=f"Confusion Matrix - {self.hparams.model_name}"
            )
        
        # Clear stored predictions
        self.test_preds = []
        self.test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }