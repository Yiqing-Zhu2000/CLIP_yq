import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import sys
import numpy as np
import pandas as pd
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip
from src.YOLO_utils import *
from src.Clip_utils import * 
# most of grid crop functions are in Clip_utils 

# ========= CONFIG ==============
# Paths
COCO_img_dir = "images/"
thresholdsGEO_path = "output/category_GEOClip_thresholds.csv"
output_dir = "output/"

img_task = "bottle"
img_name = "000000009527.jpg"
img_path = os.path.join(COCO_img_dir, img_task, img_name)
# img_path = os.path.join('houses', 'house1.jpg')

# Load model, read the thresholds file
os.makedirs(output_dir, exist_ok=True)
grid_sizes = [1,2,3,4,5]
device = "cuda" if torch.cuda.is_available() else "cpu"
Clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root="clip/models")
thresholds_df = pd.read_csv(thresholdsGEO_path)


# ============ Prepare ======================
# ======= target word, similarity threshold === vegetable, pumpkin, vehicle
# if want to test many target words, 1. check whether in 18 categories, 2. set relative thred.
target_word = "bowl"
# refer to YOLOGlove pipeline how to get target relative threshold
if target_word in thresholds_df["category"].values:
    print(target_word, " is in our 18 categories, has trained threshold.")
    threshold = thresholds_df[thresholds_df["category"] == target_word]["midpoint_threshold"].values[0]
else:
    # no analyzed threshold for this target word 
    threshold = 0
print("Threshold used here is:", threshold)

# ========== GET (GEO) grid-split patches from ONE image ==============     
img = Image.open(img_path).convert("RGB")
# Extract all patches and boxes   (grids: grid idx (List[int]), positions: List[Tuple[int, int]])
boxes, geo_patches, grids, positions = extract_grid_patches_and_boxes(img, grid_sizes)

# ======== show + store square patches on original img (for checking)=============== 
square_box_img = draw_square_boxes_on_image(img, boxes)
output_path = "output/GEO_grid_boxed.jpg"
square_box_img.save(output_path)

# =============== Input GEO gripd-split patches to CLIP ======================
target_labels = [target_word]    # only the target task. Here we just use one input target object checking
# ============== get text features by CLIP ======================
text_tokens = clip.tokenize(target_labels).to(device)
with torch.no_grad():
    text_features = Clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ========= get similarities of pacthes vs. target labels, For ONE image =========
patches_text_simMatrix = get_patches_vs_targets_simMatrix_Clip(Clip_model, preprocess, text_features, 
                                          geo_patches, device)
patches_sims = patches_text_simMatrix[:,0] # get tensor sims of the only one text for N_patches

patches_sims = patches_sims.tolist()

# ========= find the index that sim >= threshold, and store idx in list
overThred_idx = [i for i, val in enumerate(patches_sims) if val >= threshold]

# ======== store patches that with sim >= threshold ========= [for checking]
output_matchPatches_dir = "output/Oneimg_GEOmatched_patches/"
os.makedirs(output_matchPatches_dir, exist_ok=True)
# Assume: square_patches is a list of PIL.Image objects
for idx in overThred_idx:
    patch = geo_patches[idx]
    sim_score = patches_sims[idx]
    patch.save(os.path.join(output_matchPatches_dir, f"patch_{idx}_sim{sim_score:.2f}.png"))

# ======= output result ======
print("\n=== Final Judgment ===")
if overThred_idx!=[]:
    print(f"✅ image contains the target word. As there is at least one patches over the threshold.")
    print("Target matched patches stored in folder: ", output_matchPatches_dir)
else:
    print("❌ NO target word object in this image.")
