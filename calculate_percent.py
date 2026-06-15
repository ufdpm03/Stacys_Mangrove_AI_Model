from PIL import Image
import os
import numpy as np
import pandas as pd

Image.MAX_IMAGE_PIXELS = None

folder = "predicted_tiles"
output_csv = "mangrove_results.csv"

mangrove_pixels = 0
total_pixels = 0

for file in os.listdir(folder):

    if not file.endswith(".png"):
        continue

    img = Image.open(
        os.path.join(folder, file)
    ).convert("RGB")

    arr = np.array(img)

    # remove black no-data areas
    valid = (
        (arr[:, :, 0] > 5) |
        (arr[:, :, 1] > 5) |
        (arr[:, :, 2] > 5)
    )

    # find green overlay pixels
    mangrove = (
        (arr[:, :, 1] > arr[:, :, 0] * 1.2) &
        (arr[:, :, 1] > arr[:, :, 2] * 1.2) &
        valid
    )

    mangrove_pixels += int(np.sum(mangrove))
    total_pixels += int(np.sum(valid))


percent = (mangrove_pixels / total_pixels) * 100

print("----------------------")
print("Mangrove pixels:", mangrove_pixels)
print("Total pixels:", total_pixels)
print("Mangrove cover:", round(percent, 3), "%")
print("----------------------")

results = pd.DataFrame([{
    "mangrove_pixels": mangrove_pixels,
    "total_valid_pixels": total_pixels,
    "mangrove_cover_percent": round(percent, 3)
}])

results.to_csv(output_csv, index=False)

print("Saved:", output_csv)