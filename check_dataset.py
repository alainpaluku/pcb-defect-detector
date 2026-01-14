#!/usr/bin/env python3
"""
🔍 Quick Dataset Checker for Kaggle
Run this BEFORE training to verify the dataset is properly loaded.
"""

import os
from pathlib import Path
from src.config import Config

print("=" * 60)
print("🔍 PCB DATASET CHECKER")
print("=" * 60)

# Check environment
print(f"\n📍 Environment: {'Kaggle' if Config.is_kaggle() else 'Local'}")

# Check Kaggle input
if Config.is_kaggle():
    kaggle_input = Path("/kaggle/input")
    print(f"\n📂 Datasets in /kaggle/input:")
    
    if kaggle_input.exists():
        datasets = list(kaggle_input.iterdir())
        if datasets:
            for item in datasets:
                print(f"   ✅ {item.name}")
        else:
            print("   ❌ EMPTY - No datasets found!")
            print("\n👉 Add dataset via '+ Add Input' → 'akhatova/pcb-defects'")
    else:
        print("   ❌ /kaggle/input not found")

# Try to find PCB dataset
print(f"\n🔍 Searching for PCB dataset...")
data_path = Config.get_data_path()

if data_path.exists():
    print(f"   ✅ Found at: {data_path}")
    
    # Check for class folders
    all_classes = list(set(Config.DEFECT_CLASSES + Config.DEFECT_CLASSES_ALT))
    classes_found = [c for c in all_classes if (data_path / c).exists()]
    
    if classes_found:
        print(f"\n🏷️  Classes found: {len(classes_found)}/6")
        total_images = 0
        for cls in sorted(classes_found):
            cls_path = data_path / cls
            images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))
            count = len(images)
            total_images += count
            print(f"   ✅ {cls:20s} : {count:4d} images")
        
        print(f"\n📊 Total images: {total_images}")
        
        if total_images > 0:
            print("\n" + "=" * 60)
            print("✅ DATASET OK - Ready to train!")
            print("=" * 60)
            print("\n👉 Run: python run_kaggle.py")
        else:
            print("\n❌ ERROR: No images found in class folders!")
    else:
        print(f"\n❌ ERROR: No class folders found in {data_path}")
        print(f"\n   Expected folders (any of):")
        for cls in Config.DEFECT_CLASSES:
            print(f"      - {cls}")
        print(f"\n   Or CamelCase versions:")
        print(f"      - Missing_hole, Mouse_bite, Open_circuit, Short, Spur, Spurious_copper")
else:
    print(f"   ❌ NOT FOUND: {data_path}")
    print("\n❌ ERROR: Dataset not found!")
    print("\n👉 On Kaggle:")
    print("   1. Click '+ Add Input' in the right panel")
    print("   2. Search for 'akhatova/pcb-defects'")
    print("   3. Click 'Add'")
    print("   4. Restart kernel and re-run this script")
    print("\n👉 Locally:")
    print("   1. Download from: https://www.kaggle.com/datasets/akhatova/pcb-defects")
    print("   2. Extract to: data/pcb-defects/")

print("\n" + "=" * 60)
