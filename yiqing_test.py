# the usage of example show in CLIP
# Only test for ONE image.
import torch
import clip
import numpy as np
from PIL import Image

# choose device (local I use cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"
# from models folder to find that model.pt
model, preprocess = clip.load("ViT-B/32", device=device, download_root="clip/models")
# print("the model visual input resolution:", model.visual.input_resolution)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
image = preprocess(Image.open("yoloOutput2/bicycle1.jpg")).unsqueeze(0).to(device)
# labels= ["a diagram", "a dog", "a cat"]

# test diff. labels. 
"house", "tree", "rail fence",
labels=["house", "tree", "a bicycle", "rail fence", "handrail"]
print("labels:", labels)
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    print("logits_per_image", logits_per_image)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]] 
top_label = labels[np.argmax(probs[0])]
print(f"The image most likely -> {top_label} (confidence = {probs[0].max():.3f})")