"""
Data Ingestion Module for PCB Defect Detection.

Handles dataset loading, preprocessing, class imbalance detection,
and data augmentation for industrial AOI applications.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import Config


class DataIngestion:
    """
    Manages data loading, preprocessing, and augmentation for PCB defect images.
    
    This class handles:
    - Directory-based image loading
    - Class imbalance detection and weight computation
    - Data augmentation to simulate conveyor belt variations
    - Train/validation splitting
    """
    
    def __init__(self, data_path=None):
        """
        Initialize data ingestion pipeline.
        
        Args:
            data_path (Path, optional): Path to dataset. Auto-detects if None.
        """
        self.data_path = data_path or Config.get_data_path()
        self.img_size = Config.IMG_SIZE
        self.batch_size = Config.BATCH_SIZE
        self.validation_split = Config.VALIDATION_SPLIT
        
        self.train_generator = None
        self.val_generator = None
        self.class_names = None
        self.num_classes = None
        self.class_weights = None
        
        print(f"Data path: {self.data_path}")
    
    def analyze_dataset(self):
        """
        Analyze dataset structure and class distribution.
        
        Returns:
            dict: Dataset statistics including class counts and imbalance ratio
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
        
        # Get class directories
        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        
        if not class_dirs:
            raise ValueError(f"No class directories found in: {self.data_path}")
        
        stats = {}
        total_images = 0
        
        for class_dir in class_dirs:
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.jpeg"))
            count = len(image_files)
            stats[class_dir.name] = count
            total_images += count
        
        # Calculate imbalance ratio
        if stats:
            max_count = max(stats.values())
            min_count = min(stats.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        else:
            imbalance_ratio = 1.0
        
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        print(f"Total Images: {total_images}")
        print(f"Number of Classes: {len(stats)}")
        print(f"Imbalance Ratio: {imbalance_ratio:.2f}")
        print("\nClass Distribution:")
        for class_name, count in sorted(stats.items()):
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"  {class_name:20s}: {count:5d} images ({percentage:5.2f}%)")
        print("="*60 + "\n")
        
        return {
            "total_images": total_images,
            "num_classes": len(stats),
            "class_distribution": stats,
            "imbalance_ratio": imbalance_ratio
        }
    
    def compute_class_weights(self):
        """
        Compute class weights to handle imbalanced datasets.
        
        This is critical for PCB defect detection where some defect types
        may be rarer than others in the training data.
        
        Returns:
            dict: Class weights for loss function balancing
        """
        # Get all image paths and labels
        image_paths = []
        labels = []
        
        class_dirs = sorted([d for d in self.data_path.iterdir() if d.is_dir()])
        self.class_names = [d.name for d in class_dirs]
        
        for idx, class_dir in enumerate(class_dirs):
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.jpeg"))
            labels.extend([idx] * len(image_files))
        
        # Compute weights
        labels_array = np.array(labels)
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels_array),
            y=labels_array
        )
        
        self.class_weights = dict(enumerate(class_weights_array))
        
        print("Class Weights (for handling imbalance):")
        for idx, weight in self.class_weights.items():
            print(f"  {self.class_names[idx]:20s}: {weight:.4f}")
        print()
        
        return self.class_weights
    
    def create_data_generators(self, use_tf_data=False):
        """
        Create training and validation data generators with augmentation.
        
        Augmentation simulates real-world variations in PCB positioning
        on conveyor belts and camera angles in industrial settings.
        
        Args:
            use_tf_data (bool): Use tf.data API for better performance
        
        Returns:
            tuple: (train_generator, val_generator)
        """
        if use_tf_data:
            return self._create_tf_data_pipeline()
        
        # Training data augmentation (robust for small datasets)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=Config.ROTATION_RANGE,
            width_shift_range=Config.WIDTH_SHIFT_RANGE,
            height_shift_range=Config.HEIGHT_SHIFT_RANGE,
            zoom_range=Config.ZOOM_RANGE,
            horizontal_flip=Config.HORIZONTAL_FLIP,
            vertical_flip=Config.VERTICAL_FLIP,
            fill_mode='nearest',
            validation_split=self.validation_split
        )
        
        # Validation data (no augmentation, only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split
        )
        
        # Create training generator
        self.train_generator = train_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=Config.RANDOM_SEED
        )
        
        # Create validation generator
        self.val_generator = val_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=Config.RANDOM_SEED
        )
        
        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}\n")
        
        return self.train_generator, self.val_generator
    
    def _create_tf_data_pipeline(self):
        """
        Create optimized tf.data pipeline for better performance.
        
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        # This is a placeholder for tf.data implementation
        # Can be implemented for production use
        raise NotImplementedError("tf.data pipeline not yet implemented")
    
    def get_steps_per_epoch(self):
        """
        Calculate steps per epoch for training and validation.
        
        Returns:
            tuple: (train_steps, val_steps)
        """
        train_steps = self.train_generator.samples // self.batch_size
        val_steps = self.val_generator.samples // self.batch_size
        return train_steps, val_steps
