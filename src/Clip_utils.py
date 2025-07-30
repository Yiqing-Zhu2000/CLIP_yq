import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip
from PIL import Image
import numpy as np

def to_numpy_array(lst):
    """
    Convert a list of tensors or floats to a NumPy array.
    Works for both scalar and batch tensors, on CPU or GPU.

    Note: If the input list contains tensors with inconsistent shapes,
    the returned NumPy array will have dtype=object, which may not behave as expected in downstream operations
    Args:
        lst (list): List of torch.Tensors or floats.
    Returns:
        np.ndarray: Converted NumPy array.
    """
    out = []
    for x in lst:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            out.append(x.item() if x.numel() == 1 else x.numpy())
        else:
            out.append(x)
    return np.array(out)

# ========= get similarities of pacthes vs. target labels, For ONE image =========
def get_patches_vs_targets_simMatrix_Clip(Clip_model, preprocess, text_features, 
                                          patches, device):
    """
    Computes similarity scores between image patches and text features using CLIP.

    Args:
        Clip_model: The CLIP model used to encode image patches.
        preprocess: Function to preprocess image patches for CLIP.
        text_features (torch.Tensor): Text feature vectors from CLIP, shape [N_text, D].
        patches (List[PIL.Image]): List of image patches to compare.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: Similarity matrix of shape [N_patches, N_text], with scores ×100.
    """
    # ========= input patches as CLIP inputs ==============
    patch_tensors = torch.stack([preprocess(p) for p in patches]).to(device)
    # image_features and compute logits_per_image
    with torch.no_grad():
        image_features = Clip_model.encode_image(patch_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    similarity_matrix = 100.0 * image_features @ text_features.T  # [N_patch, N_text]
    # Tips: similarity_matrix[:, 0]   # get tensor sims of the only one text for N_patches
    return similarity_matrix       # tensor matrix

# ========= functions for geometric cropping method ==============
def extract_grid_patches_and_boxes(image, grid_sizes):
    """
    Extract patches and corresponding bounding boxes from an image based on multiple grid sizes.

    Args:
        image (PIL.Image): Input image.
        grid_sizes (List[int]): List of grid splits, e.g. [1, 2, 3, 4].

    Returns:
        boxes (List[List[int]]): List of [x1, y1, x2, y2] boxes for all patches.
        patches (List[PIL.Image]): List of cropped patch images.
        grids (List[int]): Grid size corresponding to each patch.
        positions (List[Tuple[int, int]]): Row, col position in grid for each patch.
    """
    W, H = image.size
    boxes = []
    patches = []
    grids = []
    positions = []

    for grid in grid_sizes:
        patch_width = W // grid
        patch_height = H // grid

        for row in range(grid):
            for col in range(grid):
                x1 = col * patch_width
                y1 = row * patch_height
                x2 = (col + 1) * patch_width if col < grid - 1 else W
                y2 = (row + 1) * patch_height if row < grid - 1 else H

                patch = image.crop((x1, y1, x2, y2))
                boxes.append([x1, y1, x2, y2])
                patches.append(patch)
                grids.append(grid)
                positions.append((row, col))

    return boxes, patches, grids, positions

