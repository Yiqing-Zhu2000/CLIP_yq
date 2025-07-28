# based on our discussion, we need non-overlap crop patches as input for geo methods.
# This file would stop updating for aftering get the setted thresholds.
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
target_texts = ["a house", "a cat","vegetable", "a window", "pumpkins", "a bicycle", "a car","vehicle","potted plant"]
# target_texts = ["a house", "a cat", "vegetable", "a window","pumpkins","a bicycle", "a car", "door", "mailbox"]
# target_texts = ["a mailbox near the door", "mailbox"]

print("the target_texts :", target_texts)

# as I use cosine sim * 100
threshold = 25.0 
# threshold > 30.0, generally most exists. > 25.0, maybe match. 

# load image, deal with text_tokens.
image = Image.open(image_path).convert("RGB")
W, H = image.size
min_side = min(W, H)
text_tokens = clip.tokenize(target_texts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# patch sizes based on shortest side
scales = [1, 2, 3, 4, 5]     
patch_sizes = [min_side // s for s in scales]    # [3448, 1724, 1149, 862, 689]

# scale use slidng windows
all_image_features = []
all_positions = []
all_scales = []
for s_idx, patch_size in enumerate(patch_sizes):
    # set the sliding stride is :
    stride = patch_size // 2
    print(f"\n====== Square Split scale {s_idx+1} with stride {stride} ======")

    for top in range(0, H, stride):
        for left in range(0, W, stride):
            # Ensure patch stays within bounds
            if left + patch_size > W:
                left = W - patch_size
            if top + patch_size > H:
                top = H - patch_size
            if left < 0 or top < 0:
                continue

            # compute embedding for each cropped square patch. 
            patch = image.crop((left, top, left + patch_size, top + patch_size))
            patch_tensor = preprocess(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feat = model.encode_image(patch_tensor)
                image_feat /= image_feat.norm(dim=-1, keepdim=True)

            all_image_features.append(image_feat)
            all_positions.append((scales[s_idx], top, left))
            all_scales.append(scales[s_idx])

# merge all patch features [N, 512]
all_image_features = torch.cat(all_image_features, dim=0)
similarity_matrix = 100.0 * all_image_features @ text_features.T  # [N_patch, N_text]

# check whether each text exists in the image. 
detected_labels = []
for t_idx, text in enumerate(target_texts):
    sims = similarity_matrix[:, t_idx]
    #print("sims:", sims)
    max_sim, max_idx = sims.max(0)
    max_pos = all_positions[max_idx]
    scale = all_scales[max_idx]

    if max_sim > threshold:
        print(f"✅ Detected '{text}' at patch {max_pos} (scale=1/{scale}), score = {max_sim.item():.3f}")
        detected_labels.append(text)
    else:
        # patch = image.crop((left, top, left + patch_size, top + patch_size))
        # patch.save(os.path.join("saved_max_patches", f"NOT_Found_maxsim_patch.jpg"))

        print(f"❌ '{text}' not found. Max similarity = {max_sim.item():.3f}")

# output results of whether "text" exsits in image
if detected_labels!=[]:
    print("The speculated exsisting text in image are: \n", detected_labels)
else:
    print("NONE of them detected from this image.")

# ========== store the max sim patch for detected text =======================
save_dir = "saved_max_patches"
os.makedirs(save_dir, exist_ok=True)

for t_idx, text in enumerate(target_texts):
    sims = similarity_matrix[:, t_idx]
    max_sim, max_idx = sims.max(0)

    # # store image for other second top sims
    # top2_values, top2_indices = sims.topk(k=3)
    # second_score = top2_values[0]
    # print("third_score", second_score)
    # max_idx = top2_indices[0]


    scale, top, left = all_positions[max_idx]
    patch_size = min_side // scale

    # Only save if detected
    if text in detected_labels:
        patch = image.crop((left, top, left + patch_size, top + patch_size))
        clean_text = text.replace(" ", "_").replace("/", "_")  # safe file name
        patch.save(os.path.join(save_dir, f"{clean_text}_patch.jpg"))