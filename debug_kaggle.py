#!/usr/bin/env python3
"""
🔍 Debug script for Kaggle - Run this FIRST to diagnose issues
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("🔍 KAGGLE DEBUG SCRIPT")
print("=" * 60)

# 1. Check environment
print("\n📍 ENVIRONMENT CHECK")
print("-" * 40)
is_kaggle = os.path.exists("/kaggle")
print(f"   Running on Kaggle: {is_kaggle}")
print(f"   Current directory: {os.getcwd()}")
print(f"   Python version: {sys.version}")

# 2. Check /kaggle/input
print("\n📂 KAGGLE INPUT CHECK")
print("-" * 40)

kaggle_input = Path("/kaggle/input")
if kaggle_input.exists():
    datasets = list(kaggle_input.iterdir())
    if datasets:
        print(f"   ✅ Found {len(datasets)} dataset(s):")
        for ds in datasets:
            print(f"      📁 {ds.name}")
            # Show first level contents
            try:
                for item in list(ds.iterdir())[:5]:
                    print(f"         └── {item.name}")
            except:
                pass
    else:
        print("   ❌ /kaggle/input is EMPTY!")
        print("   👉 You need to add a dataset!")
else:
    print("   ❌ /kaggle/input does not exist")
    print("   👉 Not running on Kaggle or no datasets added")

# 3. Search for PCB dataset
print("\n🔍 SEARCHING FOR PCB DATASET")
print("-" * 40)

# All possible class names
class_names = [
    "missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper",
    "Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"
]

def find_classes_in_path(path):
    """Check if path contains class folders."""
    if not path.exists():
        return []
    return [c for c in class_names if (path / c).exists()]

def count_images_in_folder(folder):
    """Count images in a folder."""
    if not folder.exists():
        return 0
    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        count += len(list(folder.glob(ext)))
    return count

# Search all possible locations
search_paths = []
if kaggle_input.exists():
    for ds in kaggle_input.iterdir():
        if ds.is_dir():
            search_paths.append(ds)
            for sub in ds.iterdir():
                if sub.is_dir():
                    search_paths.append(sub)
                    for subsub in sub.iterdir():
                        if subsub.is_dir():
                            search_paths.append(subsub)

found_dataset = None
for path in search_paths:
    classes = find_classes_in_path(path)
    if len(classes) >= 3:
        print(f"   ✅ FOUND DATASET: {path}")
        print(f"      Classes: {classes}")
        total = 0
        for cls in classes:
            count = count_images_in_folder(path / cls)
            total += count
            print(f"         {cls}: {count} images")
        print(f"      Total images: {total}")
        found_dataset = path
        break

if not found_dataset:
    print("   ❌ NO PCB DATASET FOUND!")
    print("")
    print("   👉 SOLUTION:")
    print("   1. Click '+ Add Input' in the right panel")
    print("   2. Search for: akhatova/pcb-defects")
    print("   3. Click 'Add'")
    print("   4. Restart kernel and re-run this script")

# 4. Test data loading
print("\n🧪 DATA LOADING TEST")
print("-" * 40)

if found_dataset:
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        
        train_gen = datagen.flow_from_directory(
            found_dataset,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        val_gen = datagen.flow_from_directory(
            found_dataset,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        print(f"   ✅ Training samples: {train_gen.samples}")
        print(f"   ✅ Validation samples: {val_gen.samples}")
        print(f"   ✅ Classes: {list(train_gen.class_indices.keys())}")
        
        if train_gen.samples == 0:
            print("\n   ❌ ERROR: 0 training samples!")
            print("   👉 Images might be in wrong format or nested folders")
        else:
            print("\n   ✅ DATA LOADING OK - Ready to train!")
            
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
else:
    print("   ⏭️  Skipped (no dataset found)")

# 5. Summary
print("\n" + "=" * 60)
if found_dataset and 'train_gen' in dir() and train_gen.samples > 0:
    print("✅ ALL CHECKS PASSED - Ready to train!")
    print(f"   Dataset path: {found_dataset}")
    print(f"   Training samples: {train_gen.samples}")
    print("\n   👉 Run: python run_kaggle.py")
else:
    print("❌ ISSUES FOUND - Fix them before training!")
    print("\n   Most common fix:")
    print("   1. Add dataset 'akhatova/pcb-defects' via '+ Add Input'")
    print("   2. Restart kernel")
    print("   3. Re-run this script")
print("=" * 60)
