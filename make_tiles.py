from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None

input_image = "input_orthomosaic.tif"

output_folder = "tiles"

os.makedirs(output_folder, exist_ok=True)

img = Image.open(input_image)

tile_size = 512

count = 0

for y in range(0, img.height, tile_size):
    for x in range(0, img.width, tile_size):

        tile = img.crop(
            (x, y, x+tile_size, y+tile_size)
        )

        tile.save(
            f"{output_folder}/tile_{count}.png"
        )

        count += 1

print("Created", count, "tiles")
