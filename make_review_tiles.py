import os
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 256
THRESHOLD = 0.5

model = tf.keras.models.load_model("mangrove_unet_model.h5")

input_folder = "tiles"
output_folder = "review_tiles"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if not file.endswith(".png"):
        continue

    path = os.path.join(input_folder, file)
    img = cv2.imread(path)

    if img is None or img.mean() < 10:
        continue

    h, w, _ = img.shape

    small = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    small = np.expand_dims(small, axis=0)

    pred = model.predict(small, verbose=0)[0]
    pred = cv2.resize(pred, (w, h))

    mask = pred > THRESHOLD

    overlay = img.copy()
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

    out_path = os.path.join(output_folder, file)
    cv2.imwrite(out_path, overlay)

print("Done. Check the review_tiles folder.")