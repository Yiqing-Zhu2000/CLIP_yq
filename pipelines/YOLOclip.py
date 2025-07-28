import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import sys
import numpy as np
import pandas as pd
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip
from src.YOLO_utils import * 

# Paths
COCO_img_dir = "D:/019_2025summer/Datasets/COCOSearch18-images-TP/images/"
glove_path = "glove6B/glove.6B.300d.word2vec.txt"
thresholds_path = "output/category_thresholds.csv"  # change for YOLOclip

img_task = "bottle"
img_name = "000000009527.jpg"
img_path = os.path.join(COCO_img_dir, img_task, img_name)
# img_path = os.path.join('houses', 'house1.jpg')

output_dir = "output/"

# Load model, read the thresholds file
os.makedirs(output_dir, exist_ok=True)
YOLO_model = YOLO('yolov8x.pt')
device = "cuda" if torch.cuda.is_available() else "cpu"
Clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root="clip/models")

# thresholds_df = pd.read_csv(thresholds_path)

# ============ Prepare ======================
# ======= target word, similarity threshold === vegetable, pumpkin, vehicle
# if want to test many target words, 1. check whether in 18 categories, 2. set relative thred.
target_word = "bowl"
threshold = 0  # temp here！
## refer to YOLOGlove pipeline how to get target relative threshold
# if target_word in thresholds_df["category"].values:
#     print(target_word, " is in our 18 categories, has trained threshold.")
#     threshold = thresholds_df[thresholds_df["category"] == target_word]["midpoint_threshold"].values[0]
# else:
#     # no analyzed threshold for this target word 
#     threshold = 0
# print("Threshold used here is:", threshold)

# ========== Prepare ==========
img = Image.open(img_path).convert("RGB")
names = YOLO_model.names

# ========== RUN YOLO model on ONE image ==============
# verbose=False -> to close the print for YOLO log output infor.                         
YOLOresutls = YOLO_model(img_path, imgsz=1024, augment=True, verbose=False)
boxes = YOLOresutls[0].boxes.xyxy.cpu().numpy()

# ======== save patches ============
# save_YOLO_patches(img, boxes)

# # draw YOLO boxes
# draw_YOLOboxes_on_Img(img_path, boxes, names, YOLOresutls)

# ======= YOLO patches to min. square patches ==========
yolo_square_patches = []
img_width, img_height = img.size
for idx, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)

    # center of the box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    # width and height of the box
    box_w = x2 - x1
    box_h = y2 - y1
    # square size
    square_size = max(box_w, box_h)
    half_size = square_size // 2
    # intended crop coordinates
    new_x1 = center_x - half_size
    new_y1 = center_y - half_size
    new_x2 = center_x + half_size
    new_y2 = center_y + half_size
    # amount of padding needed (left, top, right, bottom)
    pad_left = max(0, -new_x1)
    pad_top = max(0, -new_y1)
    pad_right = max(0, new_x2 - img_width)
    pad_bottom = max(0, new_y2 - img_height)
    # clip coordinates to image boundaries
    crop_x1 = max(0, new_x1)
    crop_y1 = max(0, new_y1)
    crop_x2 = min(img_width, new_x2)
    crop_y2 = min(img_height, new_y2)

    # crop actual image
    cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # check if padding is needed
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        # apply gray padding only when needed
        padded_img = ImageOps.expand(cropped_img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(128, 128, 128))
    else:
        padded_img = cropped_img  # no padding needed

    # confirm output is square
    assert padded_img.size[0] == padded_img.size[1], f"Patch not square: {padded_img.size}"

    yolo_square_patches.append(padded_img)

# ================= check square patches resutls =============== 
square_box_img = draw_square_boxes_on_image(img, boxes)
output_path = "output/yolo_square_boxed.jpg"
square_box_img.save(output_path)


#####################################
names = model.names   # important! also used in later process
draw_YOLOboxes_on_Img(img_path, boxes, names, results)



# =============== YOLO detected labels + target labels ==============================
# YOLO output labels:
# extract detected index.
YOLOclass_ids = results[0].boxes.cls.cpu().numpy().astype(int)
YOLO_labels = [names[i] for i in YOLOclass_ids]   # change index -> text labels
YOLO_labels_unique = list(set(YOLO_labels))     # unique labels. 
print("YOLO detected unique labels", YOLO_labels_unique)

target = ["pumpkins","bicycle","vegetable", "car", "vehicle","a potted plant behind the fence"]
target.extend(YOLO_labels_unique)
traget_labels = list(set(target))
print("Combine YOLO detected labels with our target:", traget_labels)

# =============== input YOLO detected square images to CLIP ======================
threshold = 25.0 

# get text features
text_tokens = clip.tokenize(traget_labels).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ================== add a pad to square function =======================
# ================ Not suitable for our goal, I will not use it =========
# def pad_to_square(img):
#     w, h = img.size
#     if w != h:
#         # find the longer side
#         side = max(w, h)
#         # calculate padding (left, top, right, bottom)
#         pad_left = (side - w) // 2
#         pad_top = (side - h) // 2
#         pad_right = side - w - pad_left
#         pad_bottom = side - h - pad_top
#         padding = (pad_left, pad_top, pad_right, pad_bottom)
#         img = ImageOps.expand(img, padding, fill=(128, 128, 128))  # gray padding 
#     return img


# # ================ read all patches under yoloOutput folder ====
# patch_folder = 'yoloOutput'
# yolo_patches = []
# for fname in sorted(os.listdir(patch_folder)):
#     if fname.endswith('.jpg'):
#         patch_path = os.path.join(patch_folder, fname)
#         patch_img = Image.open(patch_path)
#         # add the gray padding for rectangular yolo detected image.
#         # if dont' want to add padding (comment it out)
#         # print("======= add the padding to square ========")
#         # patch_img =  pad_to_square(patch_img)

#         yolo_patches.append(patch_img)

# patch_tensors = torch.stack([preprocess(p) for p in yolo_patches]).to(device)


# ===== use direct YOLO detected patches as CLIP inputs ========
def get_yolo_box_patch_tensors(img, results, preprocess, device='cuda'):
    """
    Crop YOLO-detected bounding box regions (no square), preprocess them,
    and stack into a tensor batch.

    Parameters:
        img (PIL.Image): Original image.
        results: YOLO results object.
        preprocess (callable): e.g. CLIP's preprocess function.
        device (str): 'cuda' or 'cpu'

    Returns:
        torch.Tensor: (N, C, H, W) stacked tensor of patches
    """
    boxes = results[0].boxes.xyxy.cpu().numpy()
    patch_tensors = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # ensure coordinates are valid
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)

        # crop the rectangular patch
        patch = img.crop((x1, y1, x2, y2))

        # apply preprocess (e.g. resize + normalize for CLIP)
        tensor = preprocess(patch).to(device)
        patch_tensors.append(tensor)

    if patch_tensors:
        return torch.stack(patch_tensors)
    else:
        return torch.empty(0)

# patch_tensors = get_yolo_box_patch_tensors(img, results, preprocess, device=device)
# print(patch_tensors.shape)  # (N, 3, 224, 224)


# ========= use the YOLO square patches as CLIP inputs ==============
patch_tensors = torch.stack([preprocess(p) for p in yolo_square_patches]).to(device)


# image_features and compute logits_per_image
with torch.no_grad():
    image_features = model.encode_image(patch_tensors)
    image_features /= image_features.norm(dim=-1, keepdim=True)

similarity_matrix = 100.0 * image_features @ text_features.T  # [N_patch, N_text]

# check whether each text exists in the image. 
detected_labels = []
for t_idx, text in enumerate(traget_labels):
    sims = similarity_matrix[:, t_idx]
    #print("sims:", sims)
    max_sim, max_idx = sims.max(0)

    if max_sim > threshold:
        print(f"✅ Detected '{text}' at YOLO patch {max_idx}, score = {max_sim.item():.3f}")
        detected_labels.append(text)
    else:
        print(f"❌ '{text}' not found. Max similarity = {max_sim.item():.3f} at YOLO patch {max_idx}")

# output results of whether "text" exsits in image
if detected_labels!=[]:
    print("The speculated exsisting text in image are: \n", detected_labels)
else:
    print("NONE of them detected from this image.")
