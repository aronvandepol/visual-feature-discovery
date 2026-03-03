# src/visualization/gradcam_vis.py

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os

_font = None

def _get_korean_font():
    """Lazy-load Korean font for matplotlib labels."""
    global _font
    if _font is None:
        try:
            from pyfonts import load_font
            _font = load_font(
                font_url="https://github.com/google/fonts/blob/main/ofl/nanumgothic/NanumGothic-Regular.ttf?raw=true"
            )
        except Exception:
            _font = None
    return _font

def reconstruct_image_from_patches(page_patches, num_patches_h, num_patches_w, patch_size, stride):
    """
    Reconstruct the original image from patches with safe normalization.
    """
    channels = page_patches.shape[2]
    reconstructed_height = (num_patches_h - 1) * stride + patch_size
    reconstructed_width = (num_patches_w - 1) * stride + patch_size
    reconstructed_image = np.zeros((reconstructed_height, reconstructed_width, channels))
    weight_mask = np.zeros((reconstructed_height, reconstructed_width, channels))

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = page_patches[i, j].permute(1, 2, 0).cpu().numpy()
            # Safe normalization
            patch_min = patch.min()
            patch_max = patch.max()
            if patch_max > patch_min:  # Only normalize if there's a range
                patch = (patch - patch_min) / (patch_max - patch_min)
            else:
                patch = np.zeros_like(patch)  # Set to zeros if constant patch

            y = i * stride
            x = j * stride
            reconstructed_image[y:y+patch_size, x:x+patch_size] += patch
            weight_mask[y:y+patch_size, x:x+patch_size] += 1

    # Avoid division by zero
    weight_mask[weight_mask == 0] = 1
    reconstructed_image /= weight_mask

    # Denormalize the image (assuming normalization was applied with mean=0.485, std=0.229)
    reconstructed_image = reconstructed_image * 0.229 + 0.485
    reconstructed_image = np.clip(reconstructed_image, 0, 1)

    # If the image is grayscale (all channels are identical), keep it as RGB but with equal channels
    if np.allclose(reconstructed_image[:,:,0], reconstructed_image[:,:,1]) and np.allclose(reconstructed_image[:,:,0], reconstructed_image[:,:,2]):
        gray_channel = reconstructed_image[:,:,0]
        reconstructed_image = np.stack([gray_channel, gray_channel, gray_channel], axis=-1)

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

    num_patches_h, num_patches_w, channels, height, width = page_patches.shape

    # Reconstruct the image from patches
    reconstructed_image = reconstruct_image_from_patches(page_patches, num_patches_h, num_patches_w, patch_size, stride)

    # Ensure proper normalization for grayscale images
    if len(reconstructed_image.shape) == 2:
        reconstructed_image = np.stack([reconstructed_image] * 3, axis=-1)
    elif reconstructed_image.shape[2] == 1:
        reconstructed_image = np.concatenate([reconstructed_image] * 3, axis=2)

    # Normalize to [0, 1] range if needed
    if reconstructed_image.max() > 1.0:
        reconstructed_image = reconstructed_image / 255.0

    # Normalize attention weights to [0,1]
    attention_min = patch_attention.min()
    attention_max = patch_attention.max()
    patch_attention_norm = (patch_attention - attention_min) / (attention_max - attention_min)

    # Create attention heatmap matching the reconstructed image size
    attention_heatmap = np.zeros((reconstructed_image.shape[0], reconstructed_image.shape[1]))

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * stride
            x_start = j * stride
            y_end = min(y_start + patch_size, attention_heatmap.shape[0])
            x_end = min(x_start + patch_size, attention_heatmap.shape[1])

            attention_value = patch_attention_norm[i, j].item()
            attention_heatmap[y_start:y_end, x_start:x_end] += attention_value

    # Smooth the heatmap with a gaussian filter
    from scipy.ndimage import gaussian_filter
    attention_heatmap = gaussian_filter(attention_heatmap, sigma=2)

    # Normalize the heatmap
    attention_max = np.max(attention_heatmap)
    if attention_max > 0:
        attention_heatmap = attention_heatmap / attention_max

    # Apply colormap to the attention heatmap
    cmap = cm.get_cmap('jet')
    attention_colormap = cmap(attention_heatmap)[:, :, :3]

    # Create a more transparent blend for better visibility
    alpha = 0.4
    blended_image = (1 - alpha * attention_heatmap[:, :, np.newaxis]) * reconstructed_image + \
                    alpha * attention_heatmap[:, :, np.newaxis] * attention_colormap

    blended_image = np.clip(blended_image, 0, 1)

    # Plot the blended image
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(blended_image)
    plt.axis('off')

    # Get class names
    pred_label = class_names[prediction.argmax().item()]
    true_label_name = class_names[true_label.item()]
    pred_idx = prediction.argmax().item()
    true_idx = true_label.item()

    font = _get_korean_font()
    title_kwargs = {"fontsize": 16}
    if font is not None:
        title_kwargs["font"] = font
    plt.title(f'Predicted: {pred_label} [Index: {pred_idx}] | True: {true_label_name} [Index: {true_idx}]', **title_kwargs)

    return fig
