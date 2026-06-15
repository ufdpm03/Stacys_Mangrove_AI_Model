from PIL import Image
import os
import math

Image.MAX_IMAGE_PIXELS = None

files = [f for f in os.listdir(".") if f.lower().endswith((".tif",".tiff"))]

print(files)

img = Image.open(files[0])

tile_size = 512

print("Image:", files[0])
print("Width:", img.width)
print("Height:", img.height)

print("Tiles across:", math.ceil(img.width / tile_size))
print("Tiles down:", math.ceil(img.height / tile_size))
print("Total tiles:", math.ceil(img.width / tile_size) * math.ceil(img.height / tile_size))