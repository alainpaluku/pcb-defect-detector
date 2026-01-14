#!/usr/bin/env python3
"""
PCB Defect Detection - Kaggle One-Click Runner

Usage sur Kaggle (une seule cellule):
    !rm -rf /kaggle/working/pcb-defect-detector
    !git clone https://github.com/alainpaluku/pcb-defect-detector.git
    %cd /kaggle/working/pcb-defect-detector
    !python run_kaggle.py
"""

import os
import sys
import subprocess
from pathlib import Path

# ============================================================
# 1. SETUP ENVIRONMENT
# ============================================================
print("=" * 60)
print("PCB DEFECT DETECTION - AUTO RUNNER")
print("=" * 60)

# S assurer qu on est dans le bon repertoire
if os.path.basename(os.getcwd()) != 'pcb-defect-detector':
    if os.path.exists('/kaggle/working/pcb-defect-detector'):
        os.chdir('/kaggle/working/pcb-defect-detector')
    elif os.path.exists('pcb-defect-detector'):
        os.chdir('pcb-defect-detector')

sys.path.insert(0, os.getcwd())
print(f"Working directory: {os.getcwd()}")

# Installer dependances ONNX
print("\nInstalling dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tf2onnx", "onnx", "onnxruntime"], 
               capture_output=True)

# ============================================================
# 2. TROUVER LE DATASET
# ============================================================
print("\nSearching for dataset...")

data_path = None
search_paths = [
    "/kaggle/input/pcb-defects/PCB_DATASET/images",
    "/kaggle/input/pcb-defects/PCB_DATASET",
    "/kaggle/input/pcb-defects/images",
    "/kaggle/input/pcb-defects",
    "/kaggle/input/pcbdefects/PCB_DATASET",
    "/kaggle/input/pcbdefects",
]

# Classes possibles (lowercase et CamelCase)
class_names = [
    "missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper",
    "Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"
]

for p in search_paths:
    path = Path(p)
    if path.exists():
        classes_found = [c for c in class_names if (path / c).exists()]
        if len(classes_found) >= 3:
            data_path = path
            print(f"   Found: {data_path}")
            print(f"   Classes: {classes_found}")
            break

# Si pas trouve, chercher recursivement
if data_path is None and Path("/kaggle/input").exists():
    print("   Searching recursively...")
    for dataset_dir in Path("/kaggle/input").iterdir():
        if dataset_dir.is_dir():
            for root, dirs, files in os.walk(dataset_dir):
                root_path = Path(root)
                classes_found = [c for c in class_names if (root_path / c).exists()]
                if len(classes_found) >= 3:
                    data_path = root_path
                    print(f"   Found: {data_path}")
                    break
            if data_path:
                break

if data_path is None:
    print("\nERROR: Dataset not found!")
    print("\nPlease add the dataset:")
    print("   1. Click '+ Add Input' on the right panel")
    print("   2. Search for 'akhatova/pcb-defects'")
    print("   3. Click 'Add'")
    print("   4. Re-run this script")
    sys.exit(1)

# Compter les images
total_images = 0
for c in class_names:
    cls_path = data_path / c
    if cls_path.exists():
        count = len(list(cls_path.glob("*.jpg"))) + len(list(cls_path.glob("*.png"))) + len(list(cls_path.glob("*.JPG")))
        if count > 0:
            print(f"      {c}: {count} images")
            total_images += count

print(f"\n   Total: {total_images} images")

if total_images == 0:
    print("\nERROR: No images found!")
    sys.exit(1)

# ============================================================
# 3. IMPORTS
# ============================================================
print("\nLoading libraries...")

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

print(f"   TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
gpu_msg = str(len(gpus)) + " available" if gpus else "Not available"
print(f"   GPU: {gpu_msg}")

# ============================================================
# 4. TRAINING
# ============================================================
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

from src.data_ingestion import DataIngestion
from src.trainer import TrainingManager

# Initialiser avec le chemin trouve
trainer = TrainingManager()
trainer.data = DataIngestion(data_path=data_path)

# Pipeline
trainer.data.analyze_dataset()
trainer.data.compute_class_weights()
trainer.data.create_generators()
trainer.setup_model()
trainer.train()
trainer.fine_tune()

# Evaluation
metrics = trainer.evaluate()

# Visualisations
trainer.plot_training_history()
trainer.plot_confusion_matrix()

# ROC curves
trainer.data.val_generator.reset()
predictions = trainer.model.model.predict(trainer.data.val_generator, verbose=0)
y_true = trainer.data.val_generator.classes
trainer.plot_roc_curves(y_true, predictions)
trainer.generate_classification_report(y_true, np.argmax(predictions, axis=1))

# Sauvegarder
trainer.save_model()

# ============================================================
# 5. RESUME
# ============================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

# Afficher les metriques depuis l'historique si metrics vide
if metrics.get('accuracy', 0) == 0 and trainer.history:
    # Prendre les dernieres valeurs de l'historique
    hist = trainer.history.history
    acc = hist.get('val_accuracy', [0])[-1] if 'val_accuracy' in hist else 0
    prec = hist.get('val_precision', [0])[-1] if 'val_precision' in hist else 0
    rec = hist.get('val_recall', [0])[-1] if 'val_recall' in hist else 0
    f1 = 2 * (prec * rec) / (prec + rec + 1e-7)
else:
    acc = metrics.get('accuracy', 0)
    prec = metrics.get('precision', 0)
    rec = metrics.get('recall', 0)
    f1 = metrics.get('f1_score', 0)

print(f"\nResults:")
print(f"   Accuracy:  {acc:.2%}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print(f"\nOutput: /kaggle/working/")
print("\nDone!")
