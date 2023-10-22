import numpy as np
import cv2 as cv
import sys
import tensorflow as tf
img = cv.imread("Screenshot 2023-08-18 at 3.28.38 AM.png")
img = cv.resize(img, (30, 30))
model = tf.keras.models.load_model("model1.h5")
print(model.predict(np.array(img).reshape(1,30,30,3)).argmax())