# check image size
from PIL import Image

img1 = Image.open("houses/house1.jpg")   # (5168, 3448)
img2 = Image.open("patches_output/cifar100_3637.png")   # (32, 32)
img3 = Image.open("houses/house_tree.jpg")  # (5168, 3448)
img4 = Image.open("houses/house_largeTree.jpg")


print("Image house1 size:", img1.size)  # 输出 (宽, 高)
print("Image patches_output/cifar100 size:", img2.size)
print("house_tree", img3.size)
print("house_largetree:", img4.size)


