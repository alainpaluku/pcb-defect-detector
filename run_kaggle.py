#!/usr/bin/env python3
"""
🔬 PCB Defect Detection - Kaggle One-Click Runner
Exécutez ce script dans une cellule Kaggle pour tout lancer automatiquement.

Usage sur Kaggle:
    !git clone https://github.com/alainpaluku/pcb-defect-detector.git
    %cd pcb-defect-detector
    !python run_kaggle.py
"""

import os
import sys
import subprocess

# ============================================================
# 1. SETUP ENVIRONMENT
# ============================================================
print("=" * 60)
print("🚀 PCB DEFECT DETECTION - AUTO RUNNER")
print("=" * 60)

# Ensure we're in the right directory
if os.path.basename(os.getcwd()) != 'pcb-defect-detector':
    if os.path.exists('pcb-defect-detector'):
        os.chdir('pcb-defect-detector')

# Add to path
sys.path.insert(0, os.getcwd())

# Install ONNX dependencies
print("\n📦 Installing ONNX dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tf2onnx", "onnx", "onnxruntime"], 
               capture_output=True)

# ============================================================
# 2. IMPORTS
# ============================================================
print("\n📦 Loading libraries...")

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')

from src.config import Config
from src.trainer import TrainingManager

# ============================================================
# 3. VERIFY ENVIRONMENT
# ============================================================
print("\n🔍 Checking environment...")
print(f"   TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"   GPUs: {len(gpus)} available")
for gpu in gpus:
    print(f"      - {gpu.name}")

print(f"   Environment: {'Kaggle' if Config.is_kaggle() else 'Local'}")

# ============================================================
# 4. VERIFY DATASET
# ============================================================
print("\n📁 Checking dataset...")

if Config.is_kaggle():
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        datasets = os.listdir(kaggle_input)
        print(f"   Found {len(datasets)} dataset(s) in /kaggle/input:")
        for ds in datasets:
            print(f"      - {ds}")
            ds_path = os.path.join(kaggle_input, ds)
            if os.path.isdir(ds_path):
                contents = os.listdir(ds_path)[:5]
                for c in contents:
                    print(f"         └── {c}")
    else:
        print("   ⚠️  /kaggle/input not found!")
        print("   Please add dataset 'akhatova/pcb-defects' to your notebook")
        sys.exit(1)

# Force correct path detection for Kaggle
from pathlib import Path
possible_paths = [
    Path("/kaggle/input/pcb-defects/PCB_DATASET"),
    Path("/kaggle/input/pcb-defects"),
    Path("/kaggle/input/pcbdefects/PCB_DATASET"),
    Path("/kaggle/input/pcbdefects"),
]

data_path = None
for p in possible_paths:
    if p.exists() and any((p / c).exists() for c in Config.DEFECT_CLASSES):
        data_path = p
        break

if data_path is None:
    data_path = Config.get_data_path()

print(f"\n   Data path: {data_path}")

if not data_path or not data_path.exists():
    print(f"\n❌ ERROR: Dataset not found at {data_path}")
    print("\n👉 Please add the dataset 'akhatova/pcb-defects' to your Kaggle notebook:")
    print("   1. Click '+ Add Input' on the right panel")
    print("   2. Search for 'akhatova/pcb-defects'")
    print("   3. Click 'Add' to add it")
    print("   4. Re-run this script")
    sys.exit(1)

# Check for class folders
classes_found = [c for c in Config.DEFECT_CLASSES if (data_path / c).exists()]
print(f"   Classes found: {len(classes_found)}/{len(Config.DEFECT_CLASSES)}")

if len(classes_found) == 0:
    print(f"\n❌ ERROR: No class folders found in {data_path}")
    sys.exit(1)

# ============================================================
# 5. RUN TRAINING (with correct data path)
# ============================================================
print("\n" + "=" * 60)
print("🏋️ STARTING TRAINING PIPELINE")
print("=" * 60)

# Initialize and run with explicit data path
from src.data_ingestion import DataIngestion
from src.model import PCBClassifier

trainer = TrainingManager()
trainer.data = DataIngestion(data_path=data_path)
trainer.data.analyze_dataset()
trainer.data.compute_class_weights()
trainer.data.create_generators()
trainer.setup_model()
trainer.train()
trainer.fine_tune()
metrics = trainer.evaluate()
trainer.plot_training_history()
trainer.plot_confusion_matrix()
trainer.data.val_generator.reset()
predictions = trainer.model.model.predict(trainer.data.val_generator, verbose=0)
y_true = trainer.data.val_generator.classes
trainer.plot_roc_curves(y_true, predictions)
trainer.generate_classification_report(y_true, np.argmax(predictions, axis=1))
trainer.save_model()

# ============================================================
# 6. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📊 Final Results:")
print(f"   Accuracy:  {metrics.get('accuracy', 0):.2%}")
print(f"   Precision: {metrics.get('precision', 0):.4f}")
print(f"   Recall:    {metrics.get('recall', 0):.4f}")
print(f"   F1 Score:  {metrics.get('f1_score', 0):.4f}")
print(f"   AUC:       {metrics.get('auc', 0):.4f}")

print(f"\n📁 Output files saved to: {Config.get_output_path()}")
print("\n🎉 Done! Check the output folder for model and visualizations.")
