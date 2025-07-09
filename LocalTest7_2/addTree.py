import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip
from PIL import Image
import numpy as np
import os

# choose device (local I use cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"
# from models folder to find that model.pt
model, preprocess = clip.load("ViT-B/32", device=device, download_root="clip/models")

# load image
image1 = Image.open("houses/house1.jpg").convert("RGB")
image2 = Image.open("houses/house_tree.jpg").convert("RGB")
image3 = Image.open("houses/house_largeTree.jpg").convert("RGB")


# text labe
# labels = ["a house", "a bicycle", "tree","a dog", "pumpkins","rail fence"]
labels = ["a house", "a bicycle", "tree","a tree in front of the house"]
print("labels:", labels)
text_tokens = clip.tokenize(labels).to(device)

# labelsUsed = ["a house", "a bicycle", "a dog", "pumpkins","rail fence"]
# text_tokens = clip.tokenize(labelsUsed).to(device)
# print("here")
# print("text I used:", labelsUsed)

# !! special! direct use the original image, can be commented out
patches = [image1, image2,image3]

# deal with patch
patch_tensors = torch.stack([preprocess(p) for p in patches]).to(device)

# gogogo!
with torch.no_grad():
    image_features = model.encode_image(patch_tensors)
    text_features = model.encode_text(text_tokens)
    logits_per_image, _ = model(patch_tensors, text_tokens)
    print("logits_per_image (cosine similarities):\n", logits_per_image)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# output
# print("Label probs:", probs)
for i, prob in enumerate(probs):
    print("Label prob:", prob)
    top_label = labels[np.argmax(prob)]
    print(f"Patch {i}: most likely -> {top_label} (confidence = {prob.max():.3f})")

