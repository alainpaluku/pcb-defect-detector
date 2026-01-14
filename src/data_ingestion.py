"""Data ingestion module for PCB Defect Detection."""

import numpy as np
from pathlib import Path
from collections import Counter
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import Config
from src.utils import count_images, get_all_images, print_section_header, print_subsection


class DataIngestion:
    """Handles data loading, preprocessing, and augmentation for PCB defect images."""
    
    SUPPORTED_FORMATS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
    
    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else Config.get_data_path()
        self.img_size = Config.IMG_SIZE
        self.batch_size = Config.BATCH_SIZE
        self.validation_split = Config.VALIDATION_SPLIT
        
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        
        self.class_names = None
        self.num_classes = None
        self.class_weights = None
        self.dataset_stats = None
    
    def _find_data_root(self):
        """Find the correct data root containing class folders."""
        if Config._has_class_folders(self.data_path):
            return self.data_path
        
        # Search common subdirectories
        for subdir in ["", "images", "PCB_DATASET/images", "PCB_DATASET", "data"]:
            candidate = self.data_path / subdir if subdir else self.data_path
            if candidate.exists() and Config._has_class_folders(candidate):
                return candidate
        
        return self.data_path
    
    def analyze_dataset(self):
        """Analyze dataset structure and class distribution."""
        self.data_path = self._find_data_root()
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        # Get class directories
        class_dirs = sorted([
            d for d in self.data_path.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if not class_dirs:
            raise ValueError(f"No class directories found in: {self.data_path}")
        
        # Collect statistics
        stats = {}
        total = 0
        min_count = float('inf')
        max_count = 0
        
        for d in class_dirs:
            count = count_images(d, self.SUPPORTED_FORMATS)
            if count > 0:
                stats[d.name] = count
                total += count
                min_count = min(min_count, count)
                max_count = max(max_count, count)
        
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
            "avg_samples_per_class": total / len(stats)
        }
        
        # Print analysis
        self._print_dataset_analysis()
        
        return self.dataset_stats
    
    def _print_dataset_analysis(self):
        """Print formatted dataset analysis."""
        stats = self.dataset_stats
        
        print_section_header("📊 DATASET ANALYSIS")
        print(f"📁 Path: {self.data_path}")
        print(f"🖼️  Total images: {stats['total']}")
        print(f"🏷️  Classes: {stats['classes']}")
        print(f"📈 Avg per class: {stats['avg_samples_per_class']:.1f}")
        print(f"⚖️  Imbalance ratio: {stats['imbalance_ratio']:.2f}")
        print_subsection("Class Distribution:")
        
        for name, count in sorted(stats['distribution'].items()):
            pct = count / stats['total'] * 100
            bar_len = int(pct / 100 * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {name:20s} │ {bar} │ {count:4d} ({pct:5.1f}%)")
        
        print("=" * 60 + "\n")
    
    def compute_class_weights(self):
        """Compute class weights to handle imbalanced data."""
        class_dirs = sorted([
            d for d in self.data_path.iterdir() 
            if d.is_dir() and count_images(d, self.SUPPORTED_FORMATS) > 0
        ])
        self.class_names = [d.name for d in class_dirs]
        
        # Build label array
        labels = []
        for idx, d in enumerate(class_dirs):
            count = count_images(d, self.SUPPORTED_FORMATS)
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
        
        # Training augmentation - optimized for PCB images
        train_datagen = ImageDataGenerator(
            rescale=1./255,
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
        
        # Validation - no augmentation, only rescale
        val_datagen = ImageDataGenerator(
            rescale=1./255,
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
    
    def visualize_augmentation(self, num_samples=5):
        """Visualize augmentation effects on sample images."""
        import matplotlib.pyplot as plt
        from src.utils import get_all_images
        
        # Get one image per class
        fig, axes = plt.subplots(self.num_classes, num_samples + 1, 
                                  figsize=(3 * (num_samples + 1), 3 * self.num_classes))
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_path / class_name
            images = get_all_images(class_dir, self.SUPPORTED_FORMATS)
            
            if not images:
                continue
                
            # Load original image
            img = tf.keras.preprocessing.image.load_img(
                images[0], target_size=self.img_size
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Show original
            axes[class_idx, 0].imshow(img_array.astype('uint8'))
            axes[class_idx, 0].set_title(f'{class_name}\n(Original)')
            axes[class_idx, 0].axis('off')
            
            # Show augmented versions
            datagen = ImageDataGenerator(
                rotation_range=Config.ROTATION_RANGE,
                width_shift_range=Config.WIDTH_SHIFT_RANGE,
                height_shift_range=Config.HEIGHT_SHIFT_RANGE,
                zoom_range=Config.ZOOM_RANGE,
                horizontal_flip=Config.HORIZONTAL_FLIP,
                vertical_flip=Config.VERTICAL_FLIP,
                brightness_range=Config.BRIGHTNESS_RANGE,
                fill_mode=Config.FILL_MODE
            )
            
            img_batch = np.expand_dims(img_array, 0)
            aug_iter = datagen.flow(img_batch, batch_size=1)
            
            for i in range(num_samples):
                aug_img = next(aug_iter)[0].astype('uint8')
                axes[class_idx, i + 1].imshow(aug_img)
                axes[class_idx, i + 1].set_title(f'Aug {i + 1}')
                axes[class_idx, i + 1].axis('off')
        
        plt.suptitle('Data Augmentation Preview', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Config.get_output_path() / 'augmentation_preview.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"📸 Augmentation preview saved to: {output_path}")
