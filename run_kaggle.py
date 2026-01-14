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
from pathlib import Path

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
# 4. FIND DATASET (Kaggle-specific search)
# ============================================================
print("\n📁 Searching for dataset...")

def find_pcb_dataset():
    """Find PCB dataset with class folders."""
    # All possible class names (standard + CamelCase + alternatives)
    all_classes = list(set(Config.DEFECT_CLASSES + Config.DEFECT_CLASSES_ALT))
    
    # Chemins spécifiques connus pour ce dataset
    known_paths = [
        "/kaggle/input/pcb-defects/PCB_DATASET/images",
        "/kaggle/input/pcb-defects/PCB_DATASET",
        "/kaggle/input/pcb-defects",
    ]
    
    # Vérifier d'abord les chemins connus
    for path_str in known_paths:
        path = Path(path_str)
        if path.exists():
            classes_found = [c for c in all_classes if (path / c).exists()]
            if len(classes_found) >= 3:
                print(f"   ✅ Found dataset at: {path}")
                print(f"      Classes: {classes_found}")
                return path
    
    # Si pas trouvé, explorer la structure
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        print("\n   🔍 Exploring /kaggle/input structure:")
        for dataset_dir in kaggle_input.iterdir():
            if dataset_dir.is_dir():
                print(f"      📂 {dataset_dir.name}/")
                explore_dir(dataset_dir, depth=0, max_depth=4, all_classes=all_classes)
    
    # Recherche récursive
    search_paths = []
    if kaggle_input.exists():
        for dataset_dir in kaggle_input.iterdir():
            if dataset_dir.is_dir():
                search_paths.append(dataset_dir)
                for root, dirs, files in os.walk(dataset_dir):
                    for d in dirs:
                        search_paths.append(Path(root) / d)
    
    print(f"\n   🔍 Searching {len(search_paths)} directories for class folders...")
    
    for path in search_paths:
        if path.exists():
            classes_found = [c for c in all_classes if (path / c).exists()]
            if len(classes_found) >= 3:
                print(f"   ✅ Found dataset at: {path}")
                print(f"      Classes: {classes_found}")
                return path
    
    return None

def explore_dir(path, depth=0, max_depth=3, all_classes=None):
    """Recursively explore directory structure."""
    if depth >= max_depth:
        return
    if all_classes is None:
        all_classes = Config.DEFECT_CLASSES
    
    indent = "      " + "   " * depth
    try:
        items = list(path.iterdir())[:10]  # Limit to 10 items
        for item in items:
            if item.is_dir():
                # Check if it's a class folder
                is_class = item.name in all_classes
                marker = "🏷️ " if is_class else "📁 "
                print(f"{indent}└── {marker}{item.name}/")
                explore_dir(item, depth + 1, max_depth, all_classes)
            else:
                if depth < 2:  # Only show files at shallow depths
                    print(f"{indent}└── 📄 {item.name}")
    except PermissionError:
        pass

# Show what's in /kaggle/input
if Config.is_kaggle():
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        print(f"   Contents of /kaggle/input:")
        for item in kaggle_input.iterdir():
            print(f"      📂 {item.name}/")
            if item.is_dir():
                for subitem in list(item.iterdir())[:5]:
                    print(f"         └── {subitem.name}")

# Find the dataset
data_path = find_pcb_dataset()

if data_path is None:
    # Fallback to Config
    data_path = Config.get_data_path()
    if not data_path.exists():
        print(f"\n❌ ERROR: Dataset not found!")
        print("\n👉 Please add the dataset 'akhatova/pcb-defects' to your Kaggle notebook:")
        print("   1. Click '+ Add Input' on the right panel")
        print("   2. Search for 'akhatova/pcb-defects'")
        print("   3. Click 'Add' to add it")
        print("   4. Re-run this script")
        sys.exit(1)

print(f"\n   📁 Using data path: {data_path}")

# Verify classes
classes_found = [c for c in Config.DEFECT_CLASSES if (data_path / c).exists()]
print(f"   Classes found: {len(classes_found)}/{len(Config.DEFECT_CLASSES)}")
for cls in classes_found:
    count = len(list((data_path / cls).glob("*")))
    print(f"      - {cls}: {count} images")

if len(classes_found) == 0:
    print(f"\n❌ ERROR: No class folders found in {data_path}")
    print(f"   Expected folders: {Config.DEFECT_CLASSES}")
    sys.exit(1)

# ============================================================
# 5. RUN TRAINING
# ============================================================
print("\n" + "=" * 60)
print("🏋️ STARTING TRAINING PIPELINE")
print("=" * 60)

# Initialize and run with explicit data path
from src.data_ingestion import DataIngestion

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
