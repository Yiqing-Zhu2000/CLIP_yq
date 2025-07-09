# Can deal with batch of patches 
import torch
import clip
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import numpy as np
import os

# choose device (local I use cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"
# from models folder to find that model.pt
model, preprocess = clip.load("ViT-B/32", device=device, download_root="clip/models")

# load image
# image = Image.open("houses/house1.jpg").convert("RGB")
image = Image.open("houses/house1.jpg").convert("RGB")

# define patch（can change size）
save_dir = "patches_output"
os.makedirs(save_dir, exist_ok=True)

patches = [
    image.crop((0, 0, 1292, 900)),              # top-left
    image.crop((1292, 862, 3876, 2586)),          # center patch
    image.crop((450, 250, 674, 474)),          # bottom-right
]
# save patches images
for i, patch in enumerate(patches):
    patch_path = os.path.join(save_dir, f"patch_{i}.jpg")
    patch.save(patch_path)

# text label
# labels = ["a house", "a tree", "a dog", "nothing"]
# labels = ["a house", "a bicycle", "a dog", "pumpkins","rail fence"]

# how about no correct/suitable answer in label:
labels = ["a cat", "a dog", "nothing"]
text_tokens = clip.tokenize(labels).to(device)

# labelsUsed = ["a house", "a bicycle", "a dog", "pumpkins","rail fence"]
# text_tokens = clip.tokenize(labelsUsed).to(device)
# print("here")
# print("text I used:", labelsUsed)

# !! special! direct use the original image, can be commented out
# patches = [image]

# deal with patch
patch_tensors = torch.stack([preprocess(p) for p in patches]).to(device)

# gogogo!
with torch.no_grad():
    # embedding value
    image_features = model.encode_image(patch_tensors)
    print("image_feature size", image_features.shape)
    text_features = model.encode_text(text_tokens)
    print("text_features size", text_features.shape)
    logits_per_image, logits_per_text = model(patch_tensors, text_tokens)
    print("after compare with text embedding:\n", logits_per_image)
    # print("logit_per_image shape", logits_per_image.shape)
    # print("After comp, logits_per_text", logits_per_text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# output
print("Labels", labels)
# print("Label probs:", probs)
for i, prob in enumerate(probs):
    # print("for each label prob", sum(prob))
    print("Label probs:", prob)  
    top_label = labels[np.argmax(prob)]
    print(f"Patch {i}: most likely -> {top_label} (confidence = {prob.max():.3f})")


