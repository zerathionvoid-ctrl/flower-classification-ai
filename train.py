import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===========================
# Config
# ===========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"

# ===========================
# Dataset
# ===========================

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

with open("model/class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.map(
    lambda x, y: (preprocess_input(x), y)
).prefetch(AUTOTUNE)

val_dataset = val_dataset.map(
    lambda x, y: (preprocess_input(x), y)
).prefetch(AUTOTUNE)

# ===========================
# Model
# ===========================

base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(len(class_names), activation="softmax")(x)

model = Model(base_model.input, output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ===========================
# Callback
# ===========================

checkpoint = ModelCheckpoint(
    "model/model.keras",
    save_best_only=True,
    monitor="val_accuracy"
)

earlystop = EarlyStopping(
    patience=3,
    restore_best_weights=True
)

# ===========================
# Training
# ===========================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)

# ===========================
# Plot
# ===========================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.legend()

plt.tight_layout()

plt.savefig("model/history.png")

print("\nTraining selesai!")
print("Model disimpan di model/model.keras")