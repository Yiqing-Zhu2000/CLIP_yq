import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip
from PIL import Image
import numpy as np

# ========= CONFIG ==============
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root="clip/models")
image_path = "houses/house1.jpg"

target_texts = ["a house", "a cat","vegetable", "a window", "pumpkins", "a bicycle", "a car","vehicle", "potted plant"]

#target_texts = ["a house", "a cat", "vegetable", "a window","pumpkins","a bicycle", "a car", "door", "mailbox"]
threshold = 25.0  # as I use cosine sim * 100
# threshold > 30.0, generally most exists. > 25.0, maybe match. (maybe use 26 or 27 would be better. )
grid_sizes = [1,2,3,4,5]
print("the target_texts :", target_texts)

# load image, deal with text_tokens.
image = Image.open(image_path).convert("RGB")
W, H = image.size
text_tokens = clip.tokenize(target_texts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    

# for the grid_sizes each patch
all_image_features = []
all_positions = []
all_grids = []

for grid in grid_sizes:
    print(f"\n====== {grid}x{grid} Grid Split ======")

    patch_width = W // grid
    patch_height = H // grid
    patches = []
    positions = []

    for row in range(grid):
        for col in range(grid):
            left = col * patch_width
            top = row * patch_height
            right = (col + 1) * patch_width if col < grid - 1 else W
            bottom = (row + 1) * patch_height if row < grid - 1 else H
            patch = image.crop((left, top, right, bottom))
            patches.append(patch)
            positions.append((row, col))

    patch_tensors = torch.stack([preprocess(p) for p in patches]).to(device)

    # image_features and compute logits_per_image
    with torch.no_grad():
        image_features = model.encode_image(patch_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    all_image_features.append(image_features)  # [num_patches, 512]
    all_positions.extend(positions)
    all_grids.extend([grid] * len(patches))


# gather all patches image features (after L2 norm): shape [total_patches, 512]
# change list of tensor[1,512] patch to tensor shape [N_pachtes, 512]
all_image_features = torch.cat(all_image_features, dim=0)  # combine all patches
similarity_matrix = 100.0 * all_image_features @ text_features.T  # [N_patch, N_text]
# print("the similarity matrix:\n", similarity_matrix)

# check whether each text exists in the image. 
detected_labels = []
for t_idx, text in enumerate(target_texts):
    sims = similarity_matrix[:, t_idx]
    #print("sims:", sims)
    max_sim, max_idx = sims.max(0)
    max_pos = all_positions[max_idx]
    max_grid = all_grids[max_idx]

    if max_sim > threshold:
        print(f"✅ Detected '{text}' at patch {max_pos} (grid={max_grid}), score = {max_sim.item():.3f}")
        detected_labels.append(text)
    else:
        print(f"❌ '{text}' not found. Max similarity = {max_sim.item():.3f} patch{max_pos} (grid = {max_grid})")

# output results of whether "text" exsits in image
if detected_labels!=[]:
    print("The speculated exsisting text in image are: \n", detected_labels)
else:
    print("NONE of them detected from this image.")

