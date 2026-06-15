from PIL import Image
import os
import math

Image.MAX_IMAGE_PIXELS = None

input_image = "input_orthomosaic.tif"
tile_folder = "predicted_tiles"
output_file = "mangrove_overlay_full.png"

tile_size = 512

# Get uploaded image size
original = Image.open(input_image)
original_width, original_height = original.size

tiles_across = math.ceil(original_width / tile_size)
tiles_down = math.ceil(original_height / tile_size)

print("Original size:", original_width, original_height)
print("Tiles across:", tiles_across)
print("Tiles down:", tiles_down)

files = []

for f in os.listdir(tile_folder):
    if f.endswith(".png"):
        number = int(f.replace("tile_", "").replace(".png", ""))
        files.append((number, f))

files.sort()

print("Found", len(files), "tiles")

full = Image.new(
    "RGB",
    (
        tiles_across * tile_size,
        tiles_down * tile_size
    )
)

for idx, (num, filename) in enumerate(files):
    row = num // tiles_across
    col = num % tiles_across

    img = Image.open(
        os.path.join(tile_folder, filename)
    ).convert("RGB")

    full.paste(
        img,
        (
            col * tile_size,
            row * tile_size
        )
    )

    if idx % 50 == 0:
        print("stitched", idx)

# crop back to original image size
full = full.crop((0, 0, original_width, original_height))


full.save(output_file)

print("DONE")
print("Saved:", output_file)


# -----------------------------
# CREATE BLACK/WHITE AI MASK
# -----------------------------

import numpy as np

mask_file = "mangrove_mask_full.png"

arr = np.array(full)

# extract the green AI overlay
mask = (
    (arr[:,:,1] > arr[:,:,0] * 1.2) &
    (arr[:,:,1] > arr[:,:,2] * 1.2)
)

mask_img = Image.fromarray(
    (mask * 255).astype("uint8")
)

mask_img.save(mask_file)

print("Saved:", mask_file)