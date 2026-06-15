import os
from PIL import Image
import numpy as np

img_dir = "images"
mask_dir = "masks"

os.makedirs(mask_dir, exist_ok=True)

existing_masks = set()

for m in os.listdir(mask_dir):
    existing_masks.add(os.path.splitext(m)[0].lower())

created = 0

for f in os.listdir(img_dir):

    if f.lower().endswith((".jpg",".jpeg",".png")):

        base = os.path.splitext(f)[0].lower()

        if base not in existing_masks:

            img = Image.open(os.path.join(img_dir, f))

            blank = np.zeros(
                (img.height, img.width),
                dtype=np.uint8
            )

            Image.fromarray(blank).save(
                os.path.join(
                    mask_dir,
                    os.path.splitext(f)[0] + ".png"
                )
            )

            print("created", f)

            created += 1

print("DONE")
print("Created", created, "blank masks")