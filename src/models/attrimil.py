# src/models/attrimil.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .flash_attention import OptimizedAttention

class Adapter(nn.Module):
    def __init__(self, in_channels, reduction_factor=4):
        super(Adapter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_factor, 1)
        self.conv2 = nn.Conv2d(in_channels // reduction_factor, in_channels, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return identity + x


class AttriMIL(nn.Module):
    def __init__(self, num_classes, L=512, D=128, ranking_strategy='hybrid',
                 empty_patch_threshold=None, compile_backbone=True):
        super(AttriMIL, self).__init__()
        self.L = L
        self.D = D
        self.ranking_strategy = ranking_strategy  # 'heuristic', 'cross_page', 'hybrid', or 'none'

        # Default thresholds for empty patch detection
        # Note: These thresholds might need re-tuning now that calculations are on GPU
        # and possibly using different image value ranges (e.g., [0,1] vs [0,255])
        self.empty_patch_threshold = empty_patch_threshold or {
            'variance': 0.004,              # User-optimized: < 0.004 is empty (uniform patches)
            'brightness': 0.5125,           # User-optimized: > 0.5125 is empty (brighter = empty)
            'edge_magnitude_min': 0.1335,   # User-optimized: range 0.1335-0.1620 is empty
            'edge_magnitude_max': 0.1620,   # User-optimized: range 0.1335-0.1620 is empty
            'pixel_range': 0.1              # Keep original (doesn't help much)
        }

        # Feature extractor: ResNet18 backbone split into 3 frozen stages.
        # Why split? torch.compile cannot handle a single large frozen module
        # efficiently -- it retraces the entire graph on shape changes, which
        # happens constantly with variable patch counts. Splitting into stages
        # lets each compiled unit see a fixed-shape subgraph. The learnable
        # adapters between stages also give gradients a path through the frozen
        # backbone without unfreezing it.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        children = list(resnet.children())[:-2]  # remove avgpool + fc

        self.backbone_stage1 = nn.Sequential(*children[:5])   # conv1,bn1,relu,maxpool,layer1
        self.backbone_stage2 = nn.Sequential(*children[5:6])  # layer2
        self.backbone_stage3 = nn.Sequential(*children[6:7])  # layer3

        # Freeze all backbone parameters
        for stage in [self.backbone_stage1, self.backbone_stage2, self.backbone_stage3]:
            for param in stage.parameters():
                param.requires_grad = False

        # Optionally compile each frozen stage for faster inference
        if compile_backbone:
            self.backbone_stage1 = torch.compile(self.backbone_stage1)
            self.backbone_stage2 = torch.compile(self.backbone_stage2)
            self.backbone_stage3 = torch.compile(self.backbone_stage3)

        # Get the number of channels at each stage
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            f1 = self.backbone_stage1(dummy_input)
            f2 = self.backbone_stage2(f1)
            f3 = self.backbone_stage3(f2)

        # Adapters
        self.adapter1 = Adapter(f1.shape[1])
        self.adapter2 = Adapter(f2.shape[1])
        self.adapter3 = Adapter(f3.shape[1])

        # Adapt the output of ResNet to match L
        self.final_adapter = nn.Conv2d(f3.shape[1], L, kernel_size=1)

        # Patch-level attention (optimized SDPA/Flash attention)
        self.patch_attention = OptimizedAttention(L, D)

        # Classifier
        self.classifier = nn.Linear(L, num_classes)

        # Define Sobel kernels for edge detection on GPU
        # Kernels for single channel (grayscale) images
        self.sobel_x_kernel = torch.tensor([[-1., 0., 1.],
                                            [-2., 0., 2.],
                                            [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y_kernel = torch.tensor([[-1., -2., -1.],
                                            [0., 0., 0.],
                                            [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)

    def _rgb_to_grayscale_tensor(self, img_batch):
        # img_batch: (B*N, C, H, W)
        # Assuming C=3 (RGB). Weights for standard luminance method
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=img_batch.device, dtype=img_batch.dtype)
        # Reshape weights to (1, C, 1, 1) for broadcasting across batch and spatial dims
        weights = weights.view(1, img_batch.shape[1], 1, 1)
        # Sum along the channel dimension after multiplying
        grayscale_batch = (img_batch * weights).sum(dim=1, keepdim=True)
        return grayscale_batch


    def detect_empty_patches(self, x):
        """
        Detect empty patches using heuristics purely on GPU with PyTorch operations.
        x: (batch_size, num_patches_h, num_patches_w, channels, height, width)
        Returns: (batch_size, num_patches_h * num_patches_w) boolean tensor
        """
        batch_size, num_patches_h, num_patches_w, channels, height, width = x.shape
        total_patches = num_patches_h * num_patches_w

        # Reshape all patches into a single batch for processing (B*N, C, H, W)
        patches_flat = x.view(batch_size * total_patches, channels, height, width)

        # Ensure kernels are on the same device as input patches
        sobel_x_kernel = self.sobel_x_kernel.to(x.device)
        sobel_y_kernel = self.sobel_y_kernel.to(x.device)

        # Convert to grayscale if necessary
        if channels == 3:
            patches_gray = self._rgb_to_grayscale_tensor(patches_flat)
        else:
            # Assume it's already grayscale if not 3 channels
            patches_gray = patches_flat


        # --- Heuristic 1: Low Variance (uniform regions) ---
        # Calculate variance across height and width for each patch
        # Keepdim=False to get (B*N, C) or (B*N, 1) if grayscale
        variance = torch.var(patches_gray, dim=(-1, -2), keepdim=False, unbiased=True)
        if variance.ndim > 1: # If color, take mean variance across channels
            variance = variance.mean(dim=-1) # (B*N)
        is_low_variance = (variance < self.empty_patch_threshold['variance'])


        # --- Heuristic 2: High Brightness (white space) ---
        # Calculate mean brightness across height and width for each patch
        mean_brightness = torch.mean(patches_gray, dim=(-1, -2))
        if mean_brightness.ndim > 1:
            mean_brightness = mean_brightness.mean(dim=-1)
        is_high_brightness = (mean_brightness > self.empty_patch_threshold['brightness'])


        # --- Heuristic 3: Low Edge Magnitude (proxy for low edge density) ---
        # Apply Sobel filters to get gradients
        # Use F.conv2d for batch processing. Ensure appropriate padding.
        # Gradients are calculated on the single-channel (grayscale) patches.
        grad_x = F.conv2d(patches_gray, sobel_x_kernel, padding='same') # 'same' padding available in PyTorch 1.10+
        grad_y = F.conv2d(patches_gray, sobel_y_kernel, padding='same')

        # Calculate magnitude of the gradients
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        # Sum of gradient magnitudes for each patch (proxy for edge density)
        # The sum needs to be normalized by patch size to be comparable
        # Sum over H and W, then average over channels if still present (should be 1 for grayscale)
        sum_gradient_magnitude = gradient_magnitude.sum(dim=(-1, -2, -3)) # (B*N)
        # Normalize by total pixels in a patch. This gives an average magnitude per pixel.
        avg_gradient_magnitude = sum_gradient_magnitude / (width * height)

        # Range-based edge magnitude detection (user-optimized)
        edge_min = self.empty_patch_threshold['edge_magnitude_min']
        edge_max = self.empty_patch_threshold['edge_magnitude_max']
        is_mid_edge_magnitude = (avg_gradient_magnitude >= edge_min) & (avg_gradient_magnitude <= edge_max)


        # --- Heuristic 4: Low Pixel Range (proxy for low entropy/information) ---
        # Calculate min and max pixel values per patch
        min_vals = patches_gray.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0] # (B*N, 1, 1, 1)
        max_vals = patches_gray.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] # (B*N, 1, 1, 1)
        pixel_range = (max_vals - min_vals).squeeze(-1).squeeze(-1).squeeze(-1) # (B*N)

        is_low_pixel_range = (pixel_range < self.empty_patch_threshold['pixel_range'])


        # Combine all heuristics for the final empty_mask
        # All 'is_X_Y' tensors should now be (B*N) shape
        empty_mask_flat = (
            is_low_variance
            | is_high_brightness
            | is_mid_edge_magnitude
            | is_low_pixel_range
        )

        # Reshape back to (batch_size, total_patches)
        empty_mask = empty_mask_flat.view(batch_size, total_patches)

        return empty_mask


    def forward(self, x):
        # x shape: (batch_size, num_patches_h, num_patches_w, channels, height, width)
        batch_size, num_patches_h, num_patches_w, channels, height, width = x.shape

        # Store original patches for empty detection if needed
        # We need to ensure original_patches is a clone if it's used for empty detection,
        # as x will be reshaped.
        original_patches = x.clone() if self.ranking_strategy in ['heuristic', 'hybrid'] else None

        # Reshape to process all patches
        x = x.view(batch_size * num_patches_h * num_patches_w, channels, height, width)

        # Extract features through compiled backbone stages + trainable adapters
        with torch.no_grad():
            x = self.backbone_stage1(x)
        x = self.adapter1(x)
        with torch.no_grad():
            x = self.backbone_stage2(x)
        x = self.adapter2(x)
        with torch.no_grad():
            x = self.backbone_stage3(x)
        x = self.adapter3(x)

        H = self.final_adapter(x)
        H = F.adaptive_avg_pool2d(H, (1, 1)).squeeze(-1).squeeze(-1)
        H = H.view(batch_size, num_patches_h * num_patches_w, self.L)

        # Patch-level attention (optimized SDPA/Flash attention)
        A = self.patch_attention(H)  # B x 1 x (num_patches_h * num_patches_w)

        # Attribute scoring
        M = torch.bmm(A, H)  # B x 1 x L
        Y_prob = self.classifier(M.squeeze(1))  # B x num_classes
        Y_hat = torch.argmax(Y_prob, dim=1)

        # Store additional info for loss calculation
        extra_info = {
            'original_patches': original_patches,
            'patch_features': H,
            'attention_weights': A
        }

        return Y_prob, Y_hat, A, H, extra_info

    def calculate_spatial_loss(self, A):
        batch_size, _, num_patches = A.size()
        num_patches_h = int(num_patches ** 0.5)
        num_patches_w = num_patches_h
        A = A.view(batch_size, 1, num_patches_h, num_patches_w)

        diff_h = torch.abs(A[:, :, :-1, :] - A[:, :, 1:, :])
        diff_v = torch.abs(A[:, :, :, :-1] - A[:, :, :, 1:])

        spatial_loss = (diff_h.sum() + diff_v.sum()) / (batch_size * num_patches)
        return spatial_loss

    def calculate_patch_attribute_scores(self, A, H):
        """
        Calculate patch-level attribute scores: s_i = attention_i * logit_i
        A: B x 1 x N (attention weights)
        H: B x N x L (patch features)
        Returns: B x N (patch-level attribute scores)
        """
        batch_size, _, num_patches = A.shape
        A_squeezed = A.squeeze(1)  # B x N

        # Get classifier weights for computing logits
        classifier_weights = self.classifier.weight  # num_classes x L

        # Compute patch-level logits: H * classifier_weights^T
        patch_logits = torch.matmul(H, classifier_weights.T)  # B x N x num_classes

        # For simplicity, use the maximum logit across classes for each patch
        # Alternative: use logit for the predicted class
        max_logits, _ = torch.max(patch_logits, dim=2)  # B x N

        # Compute attribute scores: s_i = attention_i * max_logit_i
        attribute_scores = A_squeezed * max_logits  # B x N

        return attribute_scores

    def calculate_ranking_loss(self, A, H, labels, extra_info):
        """
        Calculate ranking loss based on the selected strategy
        """
        if self.ranking_strategy == 'none':
            return torch.tensor(0.0, device=A.device)

        # Calculate patch-level attribute scores
        patch_scores = self.calculate_patch_attribute_scores(A, H)  # B x N

        total_loss = torch.tensor(0.0, device=A.device)
        valid_samples = 0

        if self.ranking_strategy in ['heuristic', 'hybrid']:
            # Within-page ranking: informative > empty
            within_page_loss = self._calculate_within_page_loss(patch_scores, extra_info['original_patches'])
            total_loss += within_page_loss
            valid_samples += 1

        if self.ranking_strategy in ['cross_page', 'hybrid']:
            # Cross-page ranking: our shop > other shops
            cross_page_loss = self._calculate_cross_page_loss(patch_scores, labels)
            if cross_page_loss is not None:  # Only add if we have multiple shops in batch
                total_loss += cross_page_loss
                valid_samples += 1

        return total_loss / max(valid_samples, 1)

    def _calculate_within_page_loss(self, patch_scores, original_patches):
        """
        Calculate within-page ranking loss: informative patches > empty patches
        """
        if original_patches is None:
            return torch.tensor(0.0, device=patch_scores.device)

        batch_size = patch_scores.shape[0]
        empty_mask = self.detect_empty_patches(original_patches)  # B x N

        total_loss = torch.tensor(0.0, device=patch_scores.device)
        valid_bags = 0

        for b in range(batch_size):
            empty_indices = empty_mask[b]
            inform_indices = ~empty_indices

            if empty_indices.sum() > 0 and inform_indices.sum() > 0:
                max_empty_score = patch_scores[b][empty_indices].max()
                max_inform_score = patch_scores[b][inform_indices].max()

                # Ensure informative patches score higher than empty patches
                margin_loss = F.relu(-max_inform_score + max_empty_score + 0.1)

                # Regularization: encourage positive scores for informative, negative for empty
                pos_reg = F.relu(-max_inform_score)  # Penalty if max informative is negative
                neg_reg = F.relu(max_empty_score)    # Penalty if max empty is positive

                total_loss += margin_loss + pos_reg + neg_reg
                valid_bags += 1

        return total_loss / max(valid_bags, 1)

    def _calculate_cross_page_loss(self, patch_scores, labels):
        """
        Calculate cross-page ranking loss: our shop patches > other shop patches
        """
        unique_labels = torch.unique(labels)

        # Skip if batch has only one type of print shop
        if len(unique_labels) <= 1:
            return None

        total_loss = torch.tensor(0.0, device=patch_scores.device)
        valid_comparisons = 0

        for label in unique_labels:
            pos_mask = (labels == label)  # Bags from this print shop
            neg_mask = (labels != label)  # Bags from other print shops

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                # Get all patch scores for positive and negative bags
                pos_scores = patch_scores[pos_mask].flatten()  # All patches from positive bags
                neg_scores = patch_scores[neg_mask].flatten()  # All patches from negative bags

                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    max_pos_score = pos_scores.max()
                    max_neg_score = neg_scores.max()

                    # Ensure patches from correct shop score higher
                    margin_loss = F.relu(-max_pos_score + max_neg_score + 0.1)
                    total_loss += margin_loss
                    valid_comparisons += 1

        return total_loss / max(valid_comparisons, 1)
