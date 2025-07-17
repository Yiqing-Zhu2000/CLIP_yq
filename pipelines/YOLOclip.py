import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from gensim.models import KeyedVectors
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip

# load model
model = YOLO('yolov8x.pt')

# image path
img_path = os.path.join('houses', 'house1.jpg')

# run model
results = model(img_path, imgsz=1024, augment=True)

# open image
img = Image.open(img_path).convert("RGB")
os.makedirs('yoloOutput', exist_ok=True)

# save patches
boxes = results[0].boxes.xyxy.cpu().numpy()
print(f"Number of box detected: {len(boxes)} ")

for idx, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    patch = img.crop((x1, y1, x2, y2))
    patch.save(f'yoloOutput/patch_{idx}.jpg')
    # print(f"patch_{idx}.jpg saved")

# draw boxes
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

names = model.names
os.makedirs("output", exist_ok=True)

for box, cls, conf in zip(
    boxes,
    results[0].boxes.cls.cpu().numpy(),
    results[0].boxes.conf.cpu().numpy()
):
    x1, y1, x2, y2 = map(int, box)
    label = f"{names[int(cls)]} {conf:.2f}"
    draw.rectangle([x1, y1, x2, y2], outline="cyan", width=4)
    draw.text((x1, y1 - 20), label, fill="cyan", font=font)

output_path = "output/yolo_boxed.jpg"
img.save(output_path)
print(f"Saved the boxed image to {output_path}")

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

# =============== input YOLO detected images to CLIP ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root="clip/models")
threshold = 25.0 

# get text features
text_tokens = clip.tokenize(traget_labels).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ================== add a pad to square function =======================
def pad_to_square(img):
    w, h = img.size
    if w != h:
        # find the longer side
        side = max(w, h)
        # calculate padding (left, top, right, bottom)
        pad_left = (side - w) // 2
        pad_top = (side - h) // 2
        pad_right = side - w - pad_left
        pad_bottom = side - h - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        img = ImageOps.expand(img, padding, fill=(128, 128, 128))  # gray padding 
    return img


# ================ read all patches under yoloOutput folder ====
patch_folder = 'yoloOutput'
yolo_patches = []
for fname in sorted(os.listdir(patch_folder)):
    if fname.endswith('.jpg'):
        patch_path = os.path.join(patch_folder, fname)
        patch_img = Image.open(patch_path)
        # add the gray padding for rectangular yolo detected image.
        # if dont' want to add padding (comment it out)
        # print("======= add the padding to square ========")
        # patch_img =  pad_to_square(patch_img)

        yolo_patches.append(patch_img)

patch_tensors = torch.stack([preprocess(p) for p in yolo_patches]).to(device)

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
