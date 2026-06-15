import cv2
import numpy as np
import tensorflow as tf
import os

IMG_SIZE = 256

model = tf.keras.models.load_model(
    "mangrove_unet_model.h5"
)

input_folder = "tiles"
output_folder = "predicted_tiles"

os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(input_folder):

    if not filename.lower().endswith((".png",".jpg",".jpeg")):
        continue

    print("Predicting:", filename)

    image_path = os.path.join(input_folder, filename)

    img = cv2.imread(image_path)

    original = img.copy()

    h, w, _ = original.shape

    small = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    small = small / 255.0
    small = np.expand_dims(small,0)

    prediction = model.predict(small)[0]

    prediction = cv2.resize(
        prediction,
        (w,h)
    )

    mask = prediction > 0.5


    overlay = original.copy()

    # make detected mangroves green
    overlay[mask] = (
        overlay[mask] * 0.5
        +
        np.array([0,255,0]) * 0.5
    )

    output_path = os.path.join(
        output_folder,
        filename
    )

    cv2.imwrite(output_path, overlay)

print("DONE")