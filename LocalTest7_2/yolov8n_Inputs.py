import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip
from PIL import Image
import os
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# text category 
labels = ["a house", "a bicycle", "a dog", "pumpkins","rail fence"]
text = clip.tokenize(labels).to(device)
print("Labels:", labels)

# read all patches under the yolov8nOutput folder 
patch_folder = 'yolov8nOutput'
for fname in sorted(os.listdir(patch_folder)):
    if fname.endswith('.jpg'):
        patch_path = os.path.join(patch_folder, fname)
        patch_img = Image.open(patch_path)
        image_input = preprocess(patch_img).unsqueeze(0).to(device)

        # here I read patch one by one, 
        # so deal with one image with all texts once here:
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            logits_per_image, logits_per_text = model(image_input, text)
            print("logits_per_image:\n", logits_per_image)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        for i, prob in enumerate(probs): # only one row here actually 
            print("Label prob:", prob)
            top_label = labels[np.argmax(prob)]
            print(f"{fname}: most likely -> {top_label} (confidence = {prob.max():.3f})")

