# src/visualization/attention_vis.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import numpy as np
import os
import matplotlib.cm as cm
import warnings

# Suppress matplotlib font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

def ensure_korean_fonts():
    """Ensure Korean/CJK fonts are available for plots with Korean text"""
    try:
        # Try to register the font file if it exists
        noto_font_path = "NotoSansCJK.ttc"
        if os.path.exists(noto_font_path):
            fm.fontManager.addfont(noto_font_path)
        
        # Set font with fallbacks - suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans', 'Arial', 'sans-serif']
    except Exception:
        # Silently fall back to default fonts
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

def reconstruct_image_from_patches(page_patches, num_patches_h, num_patches_w, patch_size, stride):
    """
    Reconstruct the original image from patches.

    Args:
    page_patches (torch.Tensor): Patches tensor of shape (num_patches_h, num_patches_w, channels, height, width)
    num_patches_h (int): Number of patches along the height
    num_patches_w (int): Number of patches along the width
    patch_size (int): Size of each patch
    stride (int): Stride used during patch extraction

    Returns:
    np.ndarray: Reconstructed image of shape (height, width, channels)
    """
    channels = page_patches.shape[2]
    reconstructed_height = (num_patches_h - 1) * stride + patch_size
    reconstructed_width = (num_patches_w - 1) * stride + patch_size
    reconstructed_image = np.zeros((reconstructed_height, reconstructed_width, channels))
    weight_mask = np.zeros((reconstructed_height, reconstructed_width, channels))

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = page_patches[i, j].permute(1, 2, 0).cpu().numpy()
            # Normalize patch to [0,1]
            patch = (patch - patch.min()) / (patch.max() - patch.min())
            y = i * stride
            x = j * stride
            reconstructed_image[y:y+patch_size, x:x+patch_size] += patch
            weight_mask[y:y+patch_size, x:x+patch_size] += 1

    # Avoid division by zero
    weight_mask[weight_mask == 0] = 1
    reconstructed_image /= weight_mask

    # Clip pixel values to [0,1]
    reconstructed_image = np.clip(reconstructed_image, 0, 1)

    return reconstructed_image

def visualize_attention(page_patches, patch_attention, prediction, true_label, class_names, patch_size, stride):
    """
    Visualize attention by reconstructing the image and overlaying the attention heatmap.
    
    Args:
    page_patches (torch.Tensor): Patches tensor of shape (num_patches_h, num_patches_w, channels, height, width)
    patch_attention (torch.Tensor): Attention weights tensor of shape (num_patches_h, num_patches_w)
    prediction (torch.Tensor): Model's prediction tensor
    true_label (torch.Tensor): True label tensor
    class_names (list): List of class names
    patch_size (int): Size of the patches
    stride (int): Stride used during patch extraction
    
    Returns:
    matplotlib.figure.Figure: The figure object containing the visualization
    """
    # Ensure Korean fonts are available for class names
    ensure_korean_fonts()
    
    num_patches_h, num_patches_w, channels, height, width = page_patches.shape

    # Reconstruct the image from patches
    reconstructed_image = reconstruct_image_from_patches(page_patches, num_patches_h, num_patches_w, patch_size, stride)

    # Normalize attention weights to [0,1]
    attention_min = patch_attention.min()
    attention_max = patch_attention.max()
    patch_attention_norm = (patch_attention - attention_min) / (attention_max - attention_min)

    # Create attention heatmap matching the reconstructed image size
    attention_heatmap = np.zeros((reconstructed_image.shape[0], reconstructed_image.shape[1]))
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y = i * stride
            x = j * stride
            attention_value = patch_attention_norm[i, j].item()
            attention_heatmap[y:y+patch_size, x:x+patch_size] += attention_value

    # Normalize the attention heatmap
    attention_heatmap = attention_heatmap / np.max(attention_heatmap)

    # Apply colormap to the attention heatmap
    cmap = cm.get_cmap('jet')
    attention_colormap = cmap(attention_heatmap)[:, :, :3]  # Ignore alpha channel

    # Blend the attention colormap with the reconstructed image
    alpha = 0.5  # Adjust blending strength
    blended_image = (1 - alpha) * reconstructed_image + alpha * attention_colormap
    blended_image = np.clip(blended_image, 0, 1)  # Ensure pixel values are in [0,1]

    # Plot the blended image
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(blended_image)
    plt.axis('off')
    pred_label = class_names[prediction.argmax().item()]
    true_label_name = class_names[true_label.item()]
    plt.title(f'Predicted: {pred_label}, True: {true_label_name}', fontsize=16)

    # Return the figure object
    return fig