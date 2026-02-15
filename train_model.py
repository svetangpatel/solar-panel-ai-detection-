import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix

# ============ DATASET PATH ============

dataset_path = "dataset"   # My dataset folder

# ============ IMAGE GENERATOR ============

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=40,
    zoom_range=0.4,
    shear_range=0.3,
    brightness_range=[0.5,1.5],
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
dataset_path,
target_size=(224,224),
batch_size=32,
class_mode='categorical',
subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Classes:", train_data.class_indices)

# ============ MODEL ============

base_model = MobileNetV2(
weights='imagenet',
include_top=False,
input_shape=(224,224,3)
)

for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
optimizer=Adam(learning_rate=0.001),
loss='categorical_crossentropy',
metrics=['accuracy']
)

# ============ TRAIN ============
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

history1 = model.fit(
train_data,
validation_data=val_data,
epochs=5,
class_weight=class_weights
)


for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights
)
# ============ SAVE MODEL ============

model.save("model.keras")
print("MODEL SAVED SUCCESSFULLY")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred = model.predict(val_data)
y_pred = np.argmax(y_pred, axis=1)

print(classification_report(val_data.classes, y_pred))