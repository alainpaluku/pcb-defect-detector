#!/usr/bin/env python3
"""
PCB Defect Detection - Kaggle Runner (Simplified)
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("PCB DEFECT DETECTION")
print("=" * 60)

# Setup
os.chdir("/kaggle/working")
if os.path.exists("pcb-defect-detector"):
    sys.path.insert(0, "/kaggle/working/pcb-defect-detector")

# Find dataset
data_path = None
for p in [
    "/kaggle/input/pcb-defects/PCB_DATASET/images",
    "/kaggle/input/pcb-defects/PCB_DATASET",
    "/kaggle/input/pcb-defects",
]:
    if Path(p).exists():
        # Check for class folders
        classes = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper",
                   "missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
        found = [c for c in classes if (Path(p) / c).exists()]
        if len(found) >= 3:
            data_path = Path(p)
            print(f"Dataset: {data_path}")
            print(f"Classes: {found}")
            break

if data_path is None:
    print("ERROR: Dataset not found!")
    print("Add 'akhatova/pcb-defects' via '+ Add Input'")
    sys.exit(1)

# Count images
total = 0
for c in found:
    n = len(list((data_path / c).glob("*.jpg"))) + len(list((data_path / c).glob("*.JPG")))
    print(f"  {c}: {n}")
    total += n
print(f"Total: {total} images")

# Imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f"\nTensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

# Data generators - SIMPLE, no preprocessing in generator
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nTrain: {train_gen.samples}, Val: {val_gen.samples}")
print(f"Classes: {list(train_gen.class_indices.keys())}")
num_classes = len(train_gen.class_indices)

# Model - Simple MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Fine-tune
print("\n" + "=" * 60)
print("FINE-TUNING")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks
)

# Evaluate
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

val_gen.reset()
results = model.evaluate(val_gen)
print(f"\nLoss: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.2%}")

# Save
model.save("/kaggle/working/pcb_model.keras")
print("\nModel saved to /kaggle/working/pcb_model.keras")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Combine histories
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

axes[0].plot(acc, label='Train')
axes[0].plot(val_acc, label='Val')
axes[0].set_title('Accuracy')
axes[0].legend()

axes[1].plot(loss, label='Train')
axes[1].plot(val_loss, label='Val')
axes[1].set_title('Loss')
axes[1].legend()

plt.savefig('/kaggle/working/training_history.png')
print("Plot saved to /kaggle/working/training_history.png")

print("\n" + "=" * 60)
print(f"DONE! Final Accuracy: {results[1]:.2%}")
print("=" * 60)
