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


