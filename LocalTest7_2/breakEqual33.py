import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip
from PIL import Image
import numpy as np
import os


# choose device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root="clip/models")

# load image
image_path = "houses/house1.jpg"
image = Image.open(image_path).convert("RGB")
W, H = image.size

# patch config: split into 3 x 3 grid
rows, cols = 3, 3
patch_width = W // cols
patch_height = H // rows

# output folder for patches
save_dir = "patches_output_3x3"
os.makedirs(save_dir, exist_ok=True)

# split into 3×3 patches
patches = []
for row in range(rows):
    for col in range(cols):
        left = col * patch_width
        top = row * patch_height
        right = (col + 1) * patch_width if col < cols - 1 else W
        bottom = (row + 1) * patch_height if row < rows - 1 else H
        patch = image.crop((left, top, right, bottom))
        patch_path = os.path.join(save_dir, f"patch_{row}_{col}.jpg")
        patch.save(patch_path)
        patches.append(patch)

# define text labels
labels = ["a house", "a tree", "bicycle", "pumpkins", "nothing"]
text_tokens = clip.tokenize(labels).to(device)

# preprocess all patches
patch_tensors = torch.stack([preprocess(p) for p in patches]).to(device)

# run through CLIP
with torch.no_grad():
    image_features = model.encode_image(patch_tensors)
    text_features = model.encode_text(text_tokens)
    logits_per_image, _ = model(patch_tensors, text_tokens)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# output results
print("logits_per_image", logits_per_image)
print("Labels:", labels)
for idx, prob in enumerate(probs):
    row, col = divmod(idx, cols)
    top_label = labels[np.argmax(prob)]
    print(f"Patch ({row}, {col}) → {top_label} (confidence = {prob.max():.3f})")