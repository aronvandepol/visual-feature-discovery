import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def denormalize_image(img_tensor, mean=[0.485], std=[0.229]):
    """
    Denormalize image tensor for visualization.

    Args:
        img_tensor: Normalized image tensor (C,H,W) or (1,C,H,W) or (B,C,H,W)
        mean: Normalization mean values
        std: Normalization std values

    Returns:
        Denormalized image as numpy array (H,W,C) or (H,W) for grayscale
    """
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.clone()
        # Squeeze leading batch dimensions to get (C,H,W)
        while img_tensor.ndim > 3:
            img_tensor = img_tensor.squeeze(0)
        for t, m, s in zip(img_tensor, mean, std):
            t.mul_(s).add_(m)
        img_tensor = torch.clamp(img_tensor, 0, 1)
        return img_tensor.permute(1, 2, 0).cpu().numpy()
    return img_tensor


def reconstruct_image_from_patches(patches, stride):
    """
    Reconstruct image from patches with proper overlap handling.
    
    Args:
        patches: Tensor of shape (num_patches_h, num_patches_w, channels, patch_h, patch_w)
        stride: Stride used for patch extraction
        
    Returns:
        Reconstructed image as numpy array
    """
    num_patches_h, num_patches_w, channels, patch_h, patch_w = patches.shape
    
    # Calculate output image dimensions
    img_h = (num_patches_h - 1) * stride + patch_h
    img_w = (num_patches_w - 1) * stride + patch_w
    
    # Initialize output arrays
    reconstructed = np.zeros((img_h, img_w, channels))
    weight_mask = np.zeros((img_h, img_w, channels))
    
    # Place patches with overlap handling
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * stride
            y_end = y_start + patch_h
            x_start = j * stride
            x_end = x_start + patch_w
            
            # Get patch and convert to numpy
            patch = patches[i, j].permute(1, 2, 0).cpu().numpy()
            
            # Add to reconstruction
            reconstructed[y_start:y_end, x_start:x_end] += patch
            weight_mask[y_start:y_end, x_start:x_end] += 1
    
    # Average overlapping regions
    weight_mask[weight_mask == 0] = 1  # Avoid division by zero
    reconstructed = reconstructed / weight_mask
    
    # Denormalize for display
    if channels == 1:
        reconstructed = denormalize_image(torch.from_numpy(reconstructed).permute(2, 0, 1))
    else:
        reconstructed = denormalize_image(torch.from_numpy(reconstructed).permute(2, 0, 1))
    
    return reconstructed


def visualize_flash_attention(patches, attention_weights, y_prob, y_true, class_names, 
                             patch_size=64, stride=32, model_type="flash"):
    """
    Visualize Flash Attention weights overlaid on the reconstructed image.
    Compatible with both Flash AttriMIL and original AttriMIL models.
    
    Args:
        patches: Tensor of shape (num_patches_h, num_patches_w, channels, patch_height, patch_width)
        attention_weights: Tensor of shape (num_patches_h, num_patches_w) OR (1, num_patches_total) containing attention weights
        y_prob: Tensor of class probabilities
        y_true: True class label (integer)
        class_names: List of class names
        patch_size: Size of each patch
        stride: Stride used for patch extraction
        model_type: "flash" or "original" to handle different attention tensor shapes
    
    Returns:
        matplotlib figure
    """
    
    # Convert tensors to numpy and move to CPU
    if isinstance(patches, torch.Tensor):
        patches = patches.detach().cpu()
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu()
    
    num_patches_h, num_patches_w, channels, patch_h, patch_w = patches.shape
    
    # Handle different attention weight formats
    if len(attention_weights.shape) == 3 and attention_weights.shape[0] == 1:
        # Flash attention format: (1, 1, num_patches_total) -> (num_patches_h, num_patches_w)
        attention_weights = attention_weights.squeeze(0).squeeze(0)
        attention_weights = attention_weights.view(num_patches_h, num_patches_w)
    elif len(attention_weights.shape) == 2 and attention_weights.shape[0] == 1:
        # Alternative format: (1, num_patches_total) -> (num_patches_h, num_patches_w)
        attention_weights = attention_weights.squeeze(0)
        attention_weights = attention_weights.view(num_patches_h, num_patches_w)
    elif len(attention_weights.shape) == 1:
        # Flattened format: (num_patches_total,) -> (num_patches_h, num_patches_w)
        attention_weights = attention_weights.view(num_patches_h, num_patches_w)
    # If already (num_patches_h, num_patches_w), keep as is
    
    # Reconstruct the image from patches
    reconstructed_img = reconstruct_image_from_patches(patches, stride)
    
    # Create figure with subplots - enhanced layout for Flash Attention
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Original reconstructed image
    ax = axes[0, 0]
    if channels == 1:
        ax.imshow(reconstructed_img.squeeze(), cmap='gray')
    else:
        img_display = denormalize_image(reconstructed_img)
        ax.imshow(img_display)
    ax.set_title('Original Image')
    ax.axis('off')
    
    # Attention heatmap
    ax = axes[0, 1]
    im = ax.imshow(attention_weights.numpy(), cmap='hot', interpolation='nearest')
    ax.set_title(f'Attention Weights ({model_type.title()})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Attention statistics
    ax = axes[0, 2]
    attention_stats = get_attention_statistics(attention_weights)
    plot_attention_statistics(ax, attention_stats, model_type)
    
    # Attention overlay on image
    ax = axes[1, 0]
    if channels == 1:
        ax.imshow(reconstructed_img.squeeze(), cmap='gray', alpha=0.7)
    else:
        img_display = denormalize_image(reconstructed_img)
        ax.imshow(img_display, alpha=0.7)
    
    # Resize attention weights to match image size
    img_h, img_w = reconstructed_img.shape[:2] if channels == 1 else reconstructed_img.shape[1:3]
    attention_resized = torch.nn.functional.interpolate(
        attention_weights.unsqueeze(0).unsqueeze(0).float(),
        size=(img_h, img_w),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    
    ax.imshow(attention_resized, cmap='hot', alpha=0.4, interpolation='bilinear')
    ax.set_title('Attention Overlay')
    ax.axis('off')
    
    # High attention regions (top 10%)
    ax = axes[1, 1]
    plot_high_attention_regions(ax, patches, attention_weights, top_k_percent=10)
    
    # Class prediction info
    ax = axes[1, 2]
    ax.axis('off')
    
    # Get prediction info
    predicted_class_idx = torch.argmax(y_prob).item()
    predicted_class = class_names[predicted_class_idx]
    true_class = class_names[y_true.item() if isinstance(y_true, torch.Tensor) else y_true]
    confidence = torch.softmax(y_prob, dim=0)[predicted_class_idx].item()
    
    # Create enhanced text info for Flash Attention
    info_text = f"Model: {model_type.title()} AttriMIL\n"
    info_text += f"True Class: {true_class}\n"
    info_text += f"Predicted: {predicted_class}\n"
    info_text += f"Confidence: {confidence:.3f}\n"
    info_text += f"Correct: {'✓' if predicted_class == true_class else '✗'}\n\n"
    
    # Add attention info
    info_text += f"Attention Info:\n"
    info_text += f"Max attention: {attention_weights.max().item():.4f}\n"
    info_text += f"Min attention: {attention_weights.min().item():.4f}\n"
    info_text += f"Mean attention: {attention_weights.mean().item():.4f}\n"
    info_text += f"Std attention: {attention_weights.std().item():.4f}\n\n"
    
    # Add top-3 predictions
    probs = torch.softmax(y_prob, dim=0)
    top3_indices = torch.topk(probs, min(3, len(probs))).indices
    info_text += "Top 3 Predictions:\n"
    for i, idx in enumerate(top3_indices):
        class_name = class_names[idx.item()]
        prob = probs[idx].item()
        info_text += f"{i+1}. {class_name}: {prob:.3f}\n"
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig


def get_attention_statistics(attention_weights):
    """Calculate comprehensive attention statistics."""
    attention_flat = attention_weights.flatten()
    
    stats = {
        'mean': attention_flat.mean().item(),
        'std': attention_flat.std().item(),
        'min': attention_flat.min().item(),
        'max': attention_flat.max().item(),
        'median': attention_flat.median().item(),
        'q25': torch.quantile(attention_flat, 0.25).item(),
        'q75': torch.quantile(attention_flat, 0.75).item(),
        'entropy': calculate_attention_entropy(attention_flat),
        'sparsity': calculate_attention_sparsity(attention_flat),
        'concentration': calculate_attention_concentration(attention_weights)
    }
    
    return stats


def calculate_attention_entropy(attention_weights):
    """Calculate entropy of attention distribution."""
    # Normalize to probabilities
    probs = torch.softmax(attention_weights.flatten(), dim=0)
    # Add small epsilon to avoid log(0)
    probs = probs + 1e-8
    entropy = -torch.sum(probs * torch.log(probs)).item()
    return entropy


def calculate_attention_sparsity(attention_weights, threshold=0.01):
    """Calculate sparsity (percentage of weights below threshold)."""
    below_threshold = (attention_weights < threshold).float().mean()
    return below_threshold.item()


def calculate_attention_concentration(attention_weights, top_k_percent=10):
    """Calculate how concentrated attention is in top k% of patches."""
    attention_flat = attention_weights.flatten()
    total_attention = attention_flat.sum()
    
    # Get top k% indices
    k = max(1, int(len(attention_flat) * top_k_percent / 100))
    top_k_values = torch.topk(attention_flat, k).values
    top_k_sum = top_k_values.sum()
    
    concentration = (top_k_sum / total_attention).item()
    return concentration


def plot_attention_statistics(ax, stats, model_type):
    """Plot attention statistics as a bar chart."""
    ax.axis('off')
    
    # Create a table-like visualization
    stat_names = ['Mean', 'Std', 'Min', 'Max', 'Entropy', 'Sparsity', 'Concentration']
    stat_values = [
        f"{stats['mean']:.4f}",
        f"{stats['std']:.4f}",
        f"{stats['min']:.4f}",
        f"{stats['max']:.4f}",
        f"{stats['entropy']:.3f}",
        f"{stats['sparsity']:.3f}",
        f"{stats['concentration']:.3f}"
    ]
    
    # Create text display
    stats_text = f"Attention Statistics ({model_type.title()}):\n\n"
    for name, value in zip(stat_names, stat_values):
        stats_text += f"{name:>12}: {value}\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))


def plot_high_attention_regions(ax, patches, attention_weights, top_k_percent=10):
    """Plot the patches with highest attention weights."""
    num_patches_h, num_patches_w, channels, patch_h, patch_w = patches.shape
    
    # Get top-k attention indices
    attention_flat = attention_weights.flatten()
    k = max(1, int(len(attention_flat) * top_k_percent / 100))
    top_k_indices = torch.topk(attention_flat, k).indices
    
    # Convert flat indices to 2D coordinates
    top_patches = []
    for idx in top_k_indices:
        i = idx // num_patches_w
        j = idx % num_patches_w
        patch = patches[i, j]
        attention_val = attention_weights[i, j].item()
        top_patches.append((patch, attention_val, i.item(), j.item()))
    
    # Create a grid of top patches
    if len(top_patches) > 0:
        # Limit to maximum displayable patches
        display_patches = top_patches[:min(16, len(top_patches))]
        
        # Arrange patches in a grid
        grid_size = int(np.ceil(np.sqrt(len(display_patches))))
        
        # Create subplot grid
        ax.axis('off')
        ax.set_title(f'Top {top_k_percent}% Attention Patches')
        
        # Create patches tensor for grid display
        patches_for_grid = []
        for patch, attn_val, i, j in display_patches:
            # Denormalize patch for display: returns (H,W,C) or (H,W) numpy
            patch_display = denormalize_image(patch)
            patches_for_grid.append(torch.from_numpy(patch_display))
        
        if patches_for_grid:
            # Stack patches and create grid
            patches_tensor = torch.stack([p.unsqueeze(0) if len(p.shape) == 2 else p for p in patches_for_grid])
            
            # Handle different channel formats
            if len(patches_tensor.shape) == 3:  # Grayscale patches
                patches_tensor = patches_tensor.unsqueeze(1)  # Add channel dimension
            elif len(patches_tensor.shape) == 4 and patches_tensor.shape[1] == 3:  # RGB patches
                pass  # Already correct format
            else:
                patches_tensor = patches_tensor.unsqueeze(1)  # Default to single channel
            
            grid = make_grid(patches_tensor, nrow=grid_size, padding=2, normalize=False)
            
            if grid.shape[0] == 1:  # Grayscale
                ax.imshow(grid.squeeze(), cmap='gray')
            else:  # RGB
                ax.imshow(grid.permute(1, 2, 0))
        
        # Add attention values as text
        info_text = "Attention values:\n"
        for i, (_, attn_val, patch_i, patch_j) in enumerate(display_patches[:8]):  # Show first 8
            info_text += f"Patch ({patch_i},{patch_j}): {attn_val:.4f}\n"
        
        ax.text(1.02, 1.0, info_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace')
    else:
        ax.text(0.5, 0.5, 'No high attention\nregions found', 
                ha='center', va='center', transform=ax.transAxes)


def compare_attention_models(patches, attention_original, attention_flash, 
                           y_prob, y_true, class_names, patch_size=64, stride=32):
    """
    Create a side-by-side comparison of original vs Flash Attention visualization.
    
    Args:
        patches: Input patches
        attention_original: Attention weights from original model
        attention_flash: Attention weights from Flash Attention model
        y_prob: Class probabilities
        y_true: True class label
        class_names: List of class names
        patch_size: Size of patches
        stride: Stride used
        
    Returns:
        matplotlib figure with comparison
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Process both attention types
    for idx, (attention, model_name) in enumerate([(attention_original, "Original"), 
                                                   (attention_flash, "Flash")]):
        row = idx
        
        # Convert and reshape attention if needed
        if isinstance(attention, torch.Tensor):
            attention = attention.detach().cpu()
        
        # Handle different shapes
        if len(attention.shape) > 2:
            attention = attention.squeeze()
        if len(attention.shape) == 1:
            num_patches_h, num_patches_w = patches.shape[:2]
            attention = attention.view(num_patches_h, num_patches_w)
        
        # Reconstruct image (same for both)
        if idx == 0:
            reconstructed_img = reconstruct_image_from_patches(patches, stride)
        
        # Original image
        ax = axes[row, 0]
        if patches.shape[2] == 1:
            ax.imshow(reconstructed_img.squeeze(), cmap='gray')
        else:
            img_display = denormalize_image(reconstructed_img)
            ax.imshow(img_display)
        ax.set_title(f'{model_name} - Original')
        ax.axis('off')
        
        # Attention heatmap
        ax = axes[row, 1]
        im = ax.imshow(attention.numpy(), cmap='hot', interpolation='nearest')
        ax.set_title(f'{model_name} - Attention')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Attention overlay
        ax = axes[row, 2]
        if patches.shape[2] == 1:
            ax.imshow(reconstructed_img.squeeze(), cmap='gray', alpha=0.7)
        else:
            img_display = denormalize_image(reconstructed_img)
            ax.imshow(img_display, alpha=0.7)
        
        # Resize attention for overlay
        img_h, img_w = reconstructed_img.shape[:2] if patches.shape[2] == 1 else reconstructed_img.shape[1:3]
        attention_resized = torch.nn.functional.interpolate(
            attention.unsqueeze(0).unsqueeze(0).float(),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        ax.imshow(attention_resized, cmap='hot', alpha=0.4)
        ax.set_title(f'{model_name} - Overlay')
        ax.axis('off')
        
        # Statistics
        ax = axes[row, 3]
        stats = get_attention_statistics(attention)
        plot_attention_statistics(ax, stats, model_name.lower())
    
    plt.tight_layout()
    return fig


# Convenience function for easy integration
def visualize_attention_flash_compatible(patches, attention_weights, y_prob, y_true, class_names, 
                                       patch_size=64, stride=32):
    """
    Auto-detecting visualization function that works with both model types.
    """
    # Detect model type based on attention tensor shape
    if isinstance(attention_weights, torch.Tensor):
        if len(attention_weights.shape) == 3 and attention_weights.shape[0] == 1:
            model_type = "flash"
        else:
            model_type = "original"
    else:
        model_type = "unknown"
    
    return visualize_flash_attention(patches, attention_weights, y_prob, y_true, class_names,
                                   patch_size, stride, model_type)