import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip
from PIL import Image
import numpy as np
import os

# check token 
tokens = clip.tokenize(["your text here"])
print("Token count:", (tokens != 0).sum(dim=1))  # Token count: tensor([5])


import clip
import torch

# 加载 CLIP 模型（只需要 tokenizer 和 encode_text）
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

# 构造超长文本
long_text = (
    "This is a very long sentence that should exceed the maximum length CLIP allows for text input. "
    "We are continuing to write more and more just to test what happens when the number of tokens goes beyond seventy-seven. "
    "This final part is very likely to be cut off entirely."
)

# # Tokenize：CLIP 会自动截断到 77 tokens
# tokens = clip.tokenize([long_text])  # shape: [1, 77]

# # 送入 encode_text 得到 embedding
# with torch.no_grad():
#     text_features = model.encode_text(tokens.to(device))  # shape: [1, 512]

# # 打印有效 token 数量（排除 padding）
# token_ids = tokens[0].tolist()
# nonzero_ids = [t for t in token_ids if t != 0]

# print(f"⚠️ Token count (after truncation): {len(nonzero_ids)} / 77")
# print(f"📏 Embedding shape: {text_features.shape}")



# check whether my long long text has been cut:
original = clip.tokenize(long_text)
print("Token shape:", original.shape)
token_count = (original != 0).sum(dim=1).item()

if token_count == 77:
    print("⚠️ Text likely got truncated (too long)")
else:
    print("✅ Text is within limit, padded to 77")
