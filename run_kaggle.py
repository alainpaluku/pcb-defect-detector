#!/usr/bin/env python3
"""
PCB Defect Detection with YOLOv8 - Kaggle Runner

Usage on Kaggle:
    !pip install ultralytics -q
    !rm -rf /kaggle/working/pcb-defect-detector
    !git clone https://github.com/alainpaluku/pcb-defect-detector.git
    %cd /kaggle/working/pcb-defect-detector
    !python run_kaggle.py
"""

import os
import sys

# Setup path
if os.path.basename(os.getcwd()) != 'pcb-defect-detector':
    if os.path.exists('/kaggle/working/pcb-defect-detector'):
        os.chdir('/kaggle/working/pcb-defect-detector')

sys.path.insert(0, os.getcwd())

# Install ultralytics
print("Installing ultralytics...")
os.system("pip install ultralytics -q")

# Run training
from src.trainer import TrainingManager

trainer = TrainingManager()
metrics = trainer.run_pipeline(epochs=50)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"mAP@50:     {metrics.get('mAP50', 0):.4f}")
print(f"mAP@50-95:  {metrics.get('mAP50-95', 0):.4f}")
print(f"Precision:  {metrics.get('precision', 0):.4f}")
print(f"Recall:     {metrics.get('recall', 0):.4f}")
print("=" * 60)
