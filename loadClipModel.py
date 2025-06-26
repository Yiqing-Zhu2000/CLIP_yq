import clip
import torch
# here I just load model .pt to my local place
# since we need to upload and use it when submit job to ComputeCanada
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, 
                              download_root="clip/models")
