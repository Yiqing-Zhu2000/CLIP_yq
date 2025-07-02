# check image size
from PIL import Image

img1 = Image.open("houses/house1.jpg")
img2 = Image.open("houses/house2.jpg")

print("Image 1 size:", img1.size)  # 输出 (宽, 高)
print("Image 2 size:", img2.size)
