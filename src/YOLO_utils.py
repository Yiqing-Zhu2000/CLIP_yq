import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def save_YOLO_patches(img, boxes):
    """
    img: the opened original img
    boxes: boxes from YOLO results[0].boxes.xyxy.cpu().numpy()
    """
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        patch = img.crop((x1, y1, x2, y2))
        os.makedirs('yoloOutput', exist_ok=True)
        patch.save(f'yoloOutput/patch_{idx}.jpg')
        # print(f"patch_{idx}.jpg saved")
    return 

# ======= draw boxes =======
# ======== draw boxes on the original img and store new img to "output/yolo_boxed.jpg" ======
def draw_YOLOboxes_on_Img(img_path, boxes, mod_names, YOLOresutls):
    """
    boxes: can be square boxes modified based on detected boxes
    mod_names: use model.names
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box, cls, conf in zip(
        boxes,
        YOLOresutls[0].boxes.cls.cpu().numpy(),
        YOLOresutls[0].boxes.conf.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)
        label = f"{mod_names[int(cls)]} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="cyan", width=4)
        draw.text((x1, y1 - 20), label, fill="cyan", font=font)
    
    os.makedirs("output", exist_ok=True)
    output_path = "output/yolo_boxed.jpg"
    img.save(output_path)
    print(f"Saved the boxed image to {output_path}")
    return 

# ================= check square patches resutls =============== 
def draw_square_boxes_on_image(img, boxes, outline_color='red', width=3):
    """
    Draw square boxes (centered on YOLO-detected boxes) on the image.
    
    Parameters:
        img (PIL.Image): The original image.
        boxes (List of [x1, y1, x2, y2]): YOLO output bounding boxes.
        outline_color (str or tuple): Color of the box outlines.
        width (int): Line width of the boxes.

    Returns:
        PIL.Image: Image with drawn square boxes.
    """

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    img_width, img_height = img.size

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # center and square size
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        square_size = max(x2 - x1, y2 - y1)
        half_size = square_size // 2

        # square coordinates
        sx1 = max(center_x - half_size, 0)
        sy1 = max(center_y - half_size, 0)
        sx2 = min(center_x + half_size, img_width)
        sy2 = min(center_y + half_size, img_height)

        # draw rectangle
        draw.rectangle([sx1, sy1, sx2, sy2], outline=outline_color, width=width)

        # draw index number near top-left corner
        text_position = (sx1 + 4, sy1 + 2)
        draw.text(text_position, str(idx), fill="red", font=font)

    return img_copy

# ============ functions used in set_threshold =========
def group_by_task(deduplicated_data):
    """
    based on task to classify, and under each task, 
    store in form:  'task': [{'name': 'name1.jpg', 'bbox': [x, y, w, h]}, 
                             {'name': 'name2.jpg', 'bbox': [x2, y2, w2, h2]},...]
    and return this 

    Args:
        deduplicated_data (List[Dict]): [{task, name, bbox}, ...]
        read from new json file
    """
    grouped = defaultdict(list)

    for item in deduplicated_data:
        grouped[item["task"]].append({
            "name": item["name"],
            "bbox": item["bbox"]
        })

    return dict(grouped)


# ============ yolo detects for one image =========
def YOLO_detect_labels_boxes(img_path, image_name, YOLO_model):
    # run YOLO model
    # verbose=False -> to close the print for YOLO log output infor.  
    results = YOLO_model(img_path, imgsz=1024, augment=True, verbose=False) 
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
    pred_labels = [YOLO_model.names[int(i)] for i in results[0].boxes.cls.cpu().numpy()]   # match with pred_boxes
    return pred_boxes, pred_labels

# =============== compute IOU ==================
def compute_iou(boxA, boxB):
    # compute the intersection or
    # boxA: yolo box [x1, y1, x2, y2]
    # boxB: COCO bbox [x, y, w, h] target label's bbox (ground truth). 
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# ======= YOLO patches to min. square patches ==========
def get_YOLOsquare_pacthes(img, boxes):
    yolo_square_patches = []    # list of image patches
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
    return yolo_square_patches

# ============= for ONE of training images (YOLO square-> CLIP) ============
def get_OneSignal_N_noiseSims_YOLOCLIP(bbox, pred_boxes,pred_labels, Clip_model, preprocess,
                                       yolo_square_patches, text_features, device):
    """
    Process one image to compute signal and noise similarities using YOLO and CLIP.
    Args:
        bbox: the bbox for the target object
        pred_boxes: yolo boxes [x1, y1, x2, y2]
        pred_labels: yolo detected labels corresponding to yolo boxes
        Clip_model, preprocess: loaded model and preprocess funciton
    """
    signal_sim = 0.0
    l_noise_sims = []

    # Get the Most matched box index
    matched_box_idx = -1
    max_iou = 0
    # Attenton! Here still use yolo pred_boxes to check IOU, instead of the boxes of squared_patches 
    for i, (pred_box, label) in enumerate(zip(pred_boxes, pred_labels)):
        iou = compute_iou(pred_box, bbox)
        # and iou >= 0.5
        if iou > max_iou: # this is make sure: detected patch and target patch have at least 50% overlap
            matched_box_idx = i
            max_iou = iou
    
    # ========= use the YOLO square patches as CLIP inputs ==============
    patch_tensors = torch.stack([preprocess(p) for p in yolo_square_patches]).to(device)
    # image_features and compute logits_per_image
    with torch.no_grad():
        image_features = Clip_model.encode_image(patch_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    similarity_matrix = 100.0 * image_features @ text_features.T  # [N_patch, N_text]
    sims = similarity_matrix[:, 0]   # only one text for target label here
    #print("sims:", sims)
    # the most similar one with this target 
    max_sim, max_idx = sims.max(0)      # why clip here sims is tensor??? check??
    signal_sim = max_sim
    l_noise_sims = sims.tolist()
    l_noise_sims.pop(max_idx)  # remove max_sim from original sims list. 
    
    return signal_sim, l_noise_sims
# def get_OneSingal_N_noise_sim(target_vec, bbox, pred_boxes,pred_labels,glove_model):
#     """
#     target_vec: glove vec computed for the task label
#     bbox: the bbox for the target object
#     pred_boxes: yolo boxes [x1, y1, x2, y2]
#     pred_labels: yolo detected labels corresponding to yolo boxes
#     glove_model: the loaded glove model 

#     return: one signal sim, the list of noise sims.
#     """
#     signal_sim = 0.0
#     l_noise_sims = []

#     # Get the Most matched box index
#     matched_box_idx = -1
#     max_iou = 0
#     for i, (pred_box, label) in enumerate(zip(pred_boxes, pred_labels)):
#         iou = compute_iou(pred_box, bbox)
#         # and iou >= 0.5   # check whether YOLO pred_box and target bbox have at least 50% overlap
#         if iou > max_iou:
#             matched_box_idx = i
#             max_iou = iou

#     if matched_box_idx != -1:   # find the most matched yolo box + label
#         signal_label = pred_labels[matched_box_idx]
#         label_vec = get_glove_vector(signal_label, glove_model)
#         signal_sim = np.dot(label_vec, target_vec) / (np.linalg.norm(label_vec) * np.linalg.norm(target_vec))
       
#         # removing duplicates using set(), and then keeping only those labels that are not equal to the signal_label
#         uni_noisy_labels = [l for l in set(pred_labels) if l != signal_label]
#         for l in uni_noisy_labels:
#             l_vec = get_glove_vector(l, glove_model)
#             noisy_sim = np.dot(l_vec, target_vec) / (np.linalg.norm(l_vec) * np.linalg.norm(target_vec))
#             l_noise_sims.append(noisy_sim)

#     return signal_sim, l_noise_sims

# ================ save distribution plot =============
def plot_signal_vs_noise(signal_distri, noise_distri, threshold, save_path=None, filename='signal_vs_noise.png'):
    """
    Plots histogram comparing signal and noise cosine similarity distributions, and optionally saves the plot.

    Parameters:
    - signal_distri (list or array): Cosine similarities for signal.
    - noise_distri (list or array): Cosine similarities for noise.
    - threshold (float): Decision threshold to be shown as a vertical line.
    - save_path (str or None): Directory to save the plot. If None, the plot won't be saved. eg. "output/"
    - filename (str): Name of the file to save the plot as (default: 'signal_vs_noise.png').
    """
    plt.figure(figsize=(8, 6))
    plt.hist(noise_distri, bins=30, alpha=0.5, label='Noise', color='red')
    plt.hist(signal_distri, bins=30, alpha=0.5, label='Signal', color='green')
    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
    plt.legend()
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Signal vs Noise Distribution")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)
        print(f"Plot saved to {full_path}")

    # plt.show()
    return 

# =============== get similarities of labels vs. target word, For ONE image =========
def labels_vs_target_similarity_glove(glove_model, YOLO_labels_unique, target_word):
    """
    This is for one image
    Args:
        glove_model: A pretrained GloVe model loaded with KeyedVectors.
        YOLO_labels_unique: A list of unique predicted labels from YOLO for ONE image
        target_word: The target category word to compare against.
    Return:
        found_overThred: A list of predicted labels whose cosine similarity with the target word.
    """
    # === for each label, compute similarity ===
    labels_sims = []
    target_vec = get_glove_vector(target_word, glove_model)
    for label in YOLO_labels_unique:
        label_vec = get_glove_vector(label, glove_model)
        if label_vec is None:
            print(f"'{label}' not in GloVe vocabulary, skipped.")
            continue
        # calculate cosine similarity 
        sim = np.dot(label_vec, target_vec) / (np.linalg.norm(label_vec) * np.linalg.norm(target_vec))
        print(f"{label} vs {target_word} similarity = {sim:.3f}")

        labels_sims.append(sim)
    return labels_sims

