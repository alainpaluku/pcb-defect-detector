"""Data ingestion module for PCB Defect Detection."""

import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import Config
from src.utils import count_images, SUPPORTED_IMAGE_FORMATS
from src.visualization import Visualizer


class DataIngestion:
    """Handles data loading, preprocessing, and augmentation for PCB defect images."""
    
    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else Config.get_data_path()
        self.img_size = Config.IMG_SIZE
        self.batch_size = Config.BATCH_SIZE
        self.validation_split = Config.VALIDATION_SPLIT
        
        self.train_generator = None
        self.val_generator = None
        
        self.class_names = None
        self.num_classes = None
        self.class_weights = None
        self.dataset_stats = None
    
    def analyze_dataset(self):
        """Analyze dataset structure and class distribution."""
        # Check if data path exists
        if not self.data_path.exists():
            # Try to find it again using Config logic if seemingly missing?
            # Config.get_data_path() already does the search.
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        # Get class directories
        try:
            class_dirs = sorted([
                d for d in self.data_path.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ])
        except PermissionError as e:
            raise PermissionError(f"Cannot access dataset directory: {self.data_path}") from e
        
        if not class_dirs:
            # Maybe we are not at the root with class folders?
            # Config.get_data_path() is supposed to return the folder WITH class folders.
            raise ValueError(f"No class directories found in: {self.data_path}")
        
        # Collect statistics
        stats = {}
        total = 0
        min_count = float('inf')
        max_count = 0
        
        for d in class_dirs:
            try:
                count = count_images(d)
                if count > 0:
                    stats[d.name] = count
                    total += count
                    min_count = min(min_count, count)
                    max_count = max(max_count, count)
            except PermissionError:
                print(f"   ⚠️ Cannot access: {d}")
                continue
        
        if total == 0:
            raise ValueError(f"No images found in: {self.data_path}")
        
        # Calculate imbalance ratio
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        self.dataset_stats = {
            "total": total,
            "classes": len(stats),
            "distribution": stats,
            "min_samples": min_count,
            "max_samples": max_count,
            "imbalance_ratio": imbalance_ratio,
            "avg_samples_per_class": total / len(stats) if len(stats) > 0 else 0
        }
        
        # Print analysis using Visualizer
        Visualizer.print_dataset_analysis(self.dataset_stats, self.data_path)
        
        return self.dataset_stats
    
    def compute_class_weights(self):
        """Compute class weights to handle imbalanced data."""
        class_dirs = sorted([
            d for d in self.data_path.iterdir() 
            if d.is_dir() and count_images(d) > 0
        ])
        self.class_names = [d.name for d in class_dirs]
        
        # Build label array
        labels = []
        for idx, d in enumerate(class_dirs):
            count = count_images(d)
            labels.extend([idx] * count)
        
        # Compute balanced weights
        weights = compute_class_weight(
            'balanced', 
            classes=np.unique(labels), 
            y=labels
        )
        self.class_weights = dict(enumerate(weights))
        
        print("⚖️  Class weights (for imbalance correction):")
        for idx, name in enumerate(self.class_names):
            print(f"   {name}: {self.class_weights[idx]:.3f}")
        print()
        
        return self.class_weights
    
    def create_generators(self):
        """Create training and validation data generators with augmentation."""
        
        # Training augmentation - NO rescale, preprocess_input is in the model
        train_datagen = ImageDataGenerator(
            rotation_range=Config.ROTATION_RANGE,
            width_shift_range=Config.WIDTH_SHIFT_RANGE,
            height_shift_range=Config.HEIGHT_SHIFT_RANGE,
            shear_range=Config.SHEAR_RANGE,
            zoom_range=Config.ZOOM_RANGE,
            horizontal_flip=Config.HORIZONTAL_FLIP,
            vertical_flip=Config.VERTICAL_FLIP,
            brightness_range=Config.BRIGHTNESS_RANGE,
            fill_mode=Config.FILL_MODE,
            validation_split=self.validation_split
        )
        
        # Validation - no augmentation, no rescale
        val_datagen = ImageDataGenerator(
            validation_split=self.validation_split
        )
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=Config.RANDOM_SEED,
            interpolation='bilinear'
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=Config.RANDOM_SEED,
            interpolation='bilinear'
        )
        
        # Update class info
        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print("✅ Data generators created:")
        print(f"   Training samples: {self.train_generator.samples}")
        print(f"   Validation samples: {self.val_generator.samples}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Image size: {self.img_size}")
        print(f"   Classes: {self.class_names}")
        print()
        
        # CRITICAL: Check if we have any samples
        if self.train_generator.samples == 0:
            raise ValueError(
                f"❌ CRITICAL ERROR: No training images found!\n"
                f"   Data path: {self.data_path}\n"
                f"   This usually means:\n"
                f"   1. Dataset not added to Kaggle notebook\n"
                f"   2. Wrong path to images\n"
                f"   3. Images in unsupported format\n\n"
                f"   👉 Solution: Add 'akhatova/pcb-defects' via '+ Add Input'"
            )
        
        if self.val_generator.samples == 0:
            raise ValueError(
                f"❌ CRITICAL ERROR: No validation images found!\n"
                f"   Data path: {self.data_path}\n"
                f"   Training samples: {self.train_generator.samples}"
            )
        
        return self.train_generator, self.val_generator
    
    def get_steps(self):
        """Return steps per epoch for train and validation."""
        from src.utils import calculate_steps_per_epoch
        train_steps = calculate_steps_per_epoch(self.train_generator.samples, self.batch_size)
        val_steps = calculate_steps_per_epoch(self.val_generator.samples, self.batch_size)
        return train_steps, val_steps
    
    def get_sample_batch(self):
        """Get a sample batch for visualization."""
        self.train_generator.reset()
        images, labels = next(self.train_generator)
        return images, labels
    
    def get_class_indices(self):
        """Return mapping of class names to indices."""
        return self.train_generator.class_indices
