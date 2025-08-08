import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import sys
import numpy as np
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip
from src.YOLO_utils import *
from src.Clip_utils import * 
# most of grid crop functions are in Clip_utils 
# Equal Error Rate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd
import csv

# Use geometric crop (grid split) original image then use CLIP to analyze
# Paths
COCO_img_dir = "images/"
json_path = "new_jsonFile/coco18_train_split1_deduplicated_task_name_bbox.json"
output_dir = "output/"

# Prepare, load clip model, read files
os.makedirs(output_dir, exist_ok=True)
grid_sizes = [1,2,3,4,5]
device = "cuda" if torch.cuda.is_available() else "cpu"
Clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root="clip/models")

with open(json_path, "r") as f:
    deduplicated_data = json.load(f)

# Group items by task
task_to_items = group_by_task(deduplicated_data)
# print(task_to_items)

# Process each task
thresholds_result = {}
for task, items in task_to_items.items():

    # items = items[:3]  # ← only use the first 3 samples of each category JUST for local test
    
    # for this task, collect singal vs noise sims.
    signal_sims, noise_sims = [], []

    # ======= get text features for the task by CLIP ==========
    target_labels = [task]    # only the target task.
    # get clip text features
    text_tokens = clip.tokenize(target_labels).to(device)
    with torch.no_grad():
        text_features = Clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # ===== for training images for this task, get signal/noise distributions=======
    for item in items:
        img_name = item["name"]
        bbox = item["bbox"]
        img_path = os.path.join(COCO_img_dir, task, img_name)

        # One of images' path:
        img = Image.open(img_path).convert("RGB")
        # # run yolo for each img
        # try:
        #     pred_boxes, pred_labels = YOLO_detect_labels_boxes(img_path, "noneed", YOLO_model)
        # except Exception as e:
        #     print(f"Error on image {img_name}: {e}")
        #     continue

        ##############################################
        #
        # Step 1: Extract all patches and boxes
        boxes, geo_patches, grids, positions = extract_grid_patches_and_boxes(img, grid_sizes)

        # Q: grid boxes also would may have overlap cases, so how to decisde signal cases? 
        # Q: if use non-overlapping grip split -> only one grid_size, but Q->which size would be suitable?
        # As we don't know how large/where is the target object in img. 

        # for each img for this task, based on yolo pred_boxes to match with ground truth bbox for img
        # get the most matching yolo box by iou, the corresponding yolo square box is for signal sim.
        signal_sim, list_noise_sims = get_OneSignal_N_noiseSims_CLIP(bbox, boxes, Clip_model, preprocess,
                                       geo_patches, text_features, device)
        
        signal_sims.append(signal_sim)
        noise_sims.extend(list_noise_sims)

    # ======== compute threshold for this task =============
    signal_distri = to_numpy_array(signal_sims)
    noise_distri = to_numpy_array(noise_sims)
    mu_signal = np.mean(signal_distri)
    mu_noise = np.mean(noise_distri)
    midpoint_threshold = (mu_signal + mu_noise) / 2

    # EER Threshold
    y_true = np.concatenate([np.ones_like(signal_distri), np.zeros_like(noise_distri)])
    y_scores = np.concatenate([signal_distri, noise_distri])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer_threshold = thresholds[eer_idx]

    thresholds_result[task] = {
        "midpoint_threshold": float(midpoint_threshold),
        "eer_threshold": float(eer_threshold),
        "signal_size": len(signal_distri),
        "noise_size": len(noise_distri)
    }

    # Plot and save figure
    plot_path = os.path.join(output_dir,"signal_vs_noise_thresGEO")
    plot_signal_vs_noise(signal_distri, noise_distri, midpoint_threshold, 
                         save_path=plot_path, filename = f"{task}_threshold_plot.png")
    
# Save thresholds to CSV file
csv_path = os.path.join(output_dir, "category_GEOClip_thresholds.csv")
# Convert the thresholds dictionary to a DataFrame
df = pd.DataFrame.from_dict(thresholds_result, orient="index")
df.index.name = "category"  # set the row index name
df.reset_index(inplace=True)  # move category back to a column

# Save as CSV with rounded float precision
df.to_csv(csv_path, index=False)   # float_format="%.4f"

print("\nThreshold computation complete. Results saved to:", csv_path)
