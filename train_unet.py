import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


# ------------------------
# SETTINGS
# ------------------------

IMG_SIZE = 256

IMAGE_DIR = "images"
MASK_DIR = "masks"


# ------------------------
# LOAD IMAGES
# ------------------------

images = []
masks = []

files = os.listdir(IMAGE_DIR)

for file in files:

    img_path = os.path.join(IMAGE_DIR, file)
    mask_path = os.path.join(MASK_DIR, file)

    if not os.path.exists(mask_path):
        continue

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    # anything not black becomes mangrove
    mask = (mask > 0).astype(np.float32)

    mask = np.expand_dims(mask, axis=-1)

    images.append(img)
    masks.append(mask)


X = np.array(images)
Y = np.array(masks)

print("Images loaded:", len(X))


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


# ------------------------
# BUILD U-NET
# ------------------------

def unet():

    inputs = Input((IMG_SIZE, IMG_SIZE, 3))

    c1 = Conv2D(32,3,activation="relu",padding="same")(inputs)
    c1 = Conv2D(32,3,activation="relu",padding="same")(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64,3,activation="relu",padding="same")(p1)
    c2 = Conv2D(64,3,activation="relu",padding="same")(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128,3,activation="relu",padding="same")(p2)
    c3 = Conv2D(128,3,activation="relu",padding="same")(c3)


    u1 = UpSampling2D()(c3)
    u1 = concatenate([u1,c2])

    c4 = Conv2D(64,3,activation="relu",padding="same")(u1)


    u2 = UpSampling2D()(c4)
    u2 = concatenate([u2,c1])

    c5 = Conv2D(32,3,activation="relu",padding="same")(u2)


    outputs = Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c5)

    return Model(inputs, outputs)


model = unet()


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------
# TRAIN
# ------------------------

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test,Y_test),
    epochs=25,
    batch_size=4
)


# SAVE MODEL

model.save("mangrove_unet_model.h5")

print("DONE - model saved!")
