"""
Simplified AttriMIL model that takes pre-computed backbone features as input.

Instead of: patches -> frozen ResNet18 -> adapters -> 512-dim -> attention -> classify
This does:  256-dim features -> projection(256->512) -> attention -> classify

Reuses OptimizedAttention from flash_attention.py.
"""

import torch
import torch.nn as nn
from .flash_attention import OptimizedAttention


class AttriMILFeatures(nn.Module):
    """AttriMIL operating on pre-extracted backbone features."""

    def __init__(self, num_classes, feature_dim=256, L=512, D=128):
        super(AttriMILFeatures, self).__init__()
        self.L = L
        self.D = D
        self.feature_dim = feature_dim

        # Project backbone features to attention dimension
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, L),
            nn.ReLU(),
        )

        # Attention (same as AttriMILFlash)
        self.patch_attention = OptimizedAttention(L, D)

        # Classifier (same as AttriMILFlash)
        self.classifier = nn.Linear(L, num_classes)

    def forward(self, features):
        """
        Args:
            features: (B, N, feature_dim) pre-extracted backbone features

        Returns:
            Y_prob: (B, num_classes) class logits
            Y_hat: (B,) predicted classes
            A: (B, 1, N) attention weights
            H: (B, N, L) projected features
        """
        # Project features
        H = self.projection(features)  # (B, N, L)

        # Attention
        A = self.patch_attention(H)  # (B, 1, N)

        # Weighted aggregation
        M = torch.bmm(A, H)  # (B, 1, L)

        # Classify
        Y_prob = self.classifier(M.squeeze(1))  # (B, num_classes)
        Y_hat = torch.argmax(Y_prob, dim=1)

        return Y_prob, Y_hat, A, H

    def calculate_spatial_loss(self, A):
        """Same spatial loss as AttriMILFlash."""
        batch_size, _, num_patches = A.size()
        num_patches_h = int(num_patches ** 0.5)
        num_patches_w = num_patches_h
        A = A.view(batch_size, 1, num_patches_h, num_patches_w)

        diff_h = torch.abs(A[:, :, :-1, :] - A[:, :, 1:, :])
        diff_v = torch.abs(A[:, :, :, :-1] - A[:, :, :, 1:])

        spatial_loss = (diff_h.sum() + diff_v.sum()) / (batch_size * num_patches)
        return spatial_loss

    def calculate_ranking_loss(self, A, H, labels):
        """Same ranking loss as AttriMILFlash."""
        batch_size, _, num_patches = A.size()
        A = A.squeeze(1)  # B x N

        attribute_scores = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # B x L

        num_classes = self.classifier.out_features
        loss = 0
        for c in range(num_classes):
            pos_mask = (labels == c).float()
            neg_mask = (labels != c).float()

            pos_max = (attribute_scores * pos_mask.unsqueeze(1)).max(dim=0)[0]
            neg_max = (attribute_scores * neg_mask.unsqueeze(1)).max(dim=0)[0]

            loss += torch.relu(-pos_max + neg_max + 0.1).mean()

        return loss / num_classes
