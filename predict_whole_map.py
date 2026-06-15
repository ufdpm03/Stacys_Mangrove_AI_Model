import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IMG_SIZE = 256
TILE_SIZE = 512
THRESHOLD = 0.5

ORTHO = "Pine-Island-Drive-12-22-2025-orthophoto.tif"
MODEL = "mangrove_unet_model.h5"

OUT_MASK = "whole_mangrove_prediction.png"
OUT_OVERLAY = "whole_mangrove_overlay.png"

model = tf.keras.models.load_model(MODEL)

img = Image.open(ORTHO).convert("RGB")
w, h = img.size

full_mask = np.zeros((h, w), dtype=np.uint8)
valid_pixels = np.zeros((h, w), dtype=np.uint8)

count = 0

for y in range(0, h, TILE_SIZE):
    for x in range(0, w, TILE_SIZE):

        right = min(x + TILE_SIZE, w)
        bottom = min(y + TILE_SIZE, h)

        tile = img.crop((x, y, right, bottom))
        tile_np = np.array(tile)

        if tile_np.mean() < 10:
            continue

        th, tw = tile_np.shape[:2]

        small = cv2.resize(tile_np, (IMG_SIZE, IMG_SIZE))
        small = small / 255.0
        small = np.expand_dims(small, axis=0)

        pred = model.predict(small, verbose=0)[0]
        pred = cv2.resize(pred, (tw, th))

        mask = (pred > THRESHOLD).astype(np.uint8) * 255

        full_mask[y:bottom, x:right] = mask
        valid_pixels[y:bottom, x:right] = 1

        count += 1
        print("Processed tile", count)

Image.fromarray(full_mask).save(OUT_MASK)

ortho_np = np.array(img)
overlay = ortho_np.copy()

mangrove_area = full_mask > 0
overlay[mangrove_area] = (
    overlay[mangrove_area] * 0.5 + np.array([0, 255, 0]) * 0.5
).astype(np.uint8)

Image.fromarray(overlay).save(OUT_OVERLAY)

mangrove_pixels = np.sum(full_mask > 0)
total_valid_pixels = np.sum(valid_pixels > 0)

percent_cover = (mangrove_pixels / total_valid_pixels) * 100

print("--------------------------------")
print("DONE")
print("Tiles processed:", count)
print("Mangrove pixels:", mangrove_pixels)
print("Valid pixels:", total_valid_pixels)
print("Mangrove cover:", round(percent_cover, 3), "%")
print("Saved:", OUT_MASK)
print("Saved:", OUT_OVERLAY)