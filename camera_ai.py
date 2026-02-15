import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("model.keras", compile=False)
classes = ["Bird Drop",
    "Clean Panel",
    "Crack",
    "Dusty Panel",
    "Electrical Damage",
    "Snow Covered"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
    img = np.expand_dims(img,0)

    pred = model.predict(img)
    label = classes[np.argmax(pred)]

    cv2.putText(frame,label,(20,50),
    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Solar Panel AI Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()