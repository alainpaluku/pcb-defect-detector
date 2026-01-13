"""
Data Pipeline module for PCB Defect Detector.

This module handles data splitting, augmentation, class weight calculation,
and creation of TensorFlow data generators for training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import DataConfig


class DataPipeline:
    """Handles data preparation and augmentation for model training.
    
    This class creates train/validation/test splits, calculates class weights
    for handling imbalanced data, and provides TensorFlow data generators
    with appropriate augmentation.
    
    Attributes:
        config: Data configuration object.
        class_names: List of class names.
        class_weights: Computed class weights for imbalanced data.
    """
    
    def __init__(self, config: DataConfig, class_names: List[str]) -> None:
        """Initialize DataPipeline with configuration and class names.
        
        Args:
            config: DataConfig object containing data parameters.
            class_names: List of class names in the dataset.
        """
        self.config = config
        self.class_names = class_names
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.class_weights: Optional[Dict[int, float]] = None
        self.train_paths: List[Tuple[Path, int]] = []
        self.val_paths: List[Tuple[Path, int]] = []
        self.test_paths: List[Tuple[Path, int]] = []
        
        self._train_dataset: Optional[tf.data.Dataset] = None
        self._val_dataset: Optional[tf.data.Dataset] = None
        self._test_dataset: Optional[tf.data.Dataset] = None
        
        # Augmentation parameters based on strength
        self._aug_strength = config.augmentation_strength
        
    def prepare_data(self, class_images: Dict[str, List[Path]]) -> None:
        """Prepare data by creating splits and calculating class weights.
        
        Args:
            class_images: Dictionary mapping class names to image paths.
        """
        self.logger.info("Preparing data splits...")
        
        all_paths = []
        all_labels = []
        
        for class_name, images in class_images.items():
            label = self.class_names.index(class_name)
            for img_path in images:
                all_paths.append(img_path)
                all_labels.append(label)
        
        all_paths = np.array(all_paths)
        all_labels = np.array(all_labels)
        
        # First split: separate test set
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_paths, all_labels,
            test_size=self.config.test_split,
            stratify=all_labels,
            random_state=self.config.random_seed
        )
        
        # Second split: separate validation from training
        val_ratio = self.config.validation_split / (1 - self.config.test_split)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_ratio,
            stratify=train_val_labels,
            random_state=self.config.random_seed
        )
        
        self.train_paths = list(zip(train_paths, train_labels))
        self.val_paths = list(zip(val_paths, val_labels))
        self.test_paths = list(zip(test_paths, test_labels))
        
        self.logger.info(f"Train: {len(self.train_paths)}, "
                        f"Val: {len(self.val_paths)}, "
                        f"Test: {len(self.test_paths)}")
        
        self._calculate_class_weights(train_labels)
        self._create_datasets()

    def _calculate_class_weights(self, labels: np.ndarray) -> None:
        """Calculate class weights to handle imbalanced dataset."""
        self.logger.info("Calculating class weights...")
        
        class_counts = Counter(labels)
        self.logger.info("Class distribution:")
        for class_idx, count in sorted(class_counts.items()):
            self.logger.info(f"  {self.class_names[class_idx]}: {count}")
        
        unique_classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
        
        self.class_weights = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
        
        self.logger.info("Class weights:")
        for idx, weight in self.class_weights.items():
            self.logger.info(f"  {self.class_names[idx]}: {weight:.3f}")
    
    def _load_and_preprocess(self, path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Load and preprocess a single image."""
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, self.config.image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    def _augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply data augmentation to training images."""
        s = self._aug_strength
        
        # Geometric augmentations
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Random 90-degree rotation
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        
        # Color augmentations
        image = tf.image.random_brightness(image, max_delta=0.2 * s)
        image = tf.image.random_contrast(image, 1 - 0.2 * s, 1 + 0.2 * s)
        image = tf.image.random_saturation(image, 1 - 0.2 * s, 1 + 0.2 * s)
        image = tf.image.random_hue(image, 0.05 * s)
        
        # Random zoom/crop
        if tf.random.uniform([]) > 0.5:
            scale = tf.random.uniform([], 0.9, 1.1)
            new_h = tf.cast(tf.cast(self.config.image_size[0], tf.float32) * scale, tf.int32)
            new_w = tf.cast(tf.cast(self.config.image_size[1], tf.float32) * scale, tf.int32)
            image = tf.image.resize(image, [new_h, new_w])
            image = tf.image.resize_with_crop_or_pad(image, self.config.image_size[0], self.config.image_size[1])
        
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label
    
    def _create_datasets(self) -> None:
        """Create TensorFlow datasets for train, validation, and test."""
        self.logger.info("Creating TensorFlow datasets...")
        
        # Training dataset
        train_paths = [str(p) for p, _ in self.train_paths]
        train_labels = [l for _, l in self.train_paths]
        
        self._train_dataset = (
            tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
            .shuffle(len(train_paths), seed=self.config.random_seed)
            .map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        # Validation dataset
        val_paths = [str(p) for p, _ in self.val_paths]
        val_labels = [l for _, l in self.val_paths]
        
        self._val_dataset = (
            tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
            .map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        # Test dataset
        test_paths = [str(p) for p, _ in self.test_paths]
        test_labels = [l for _, l in self.test_paths]
        
        self._test_dataset = (
            tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
            .map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    
    def get_train_dataset(self) -> tf.data.Dataset:
        """Get the training dataset."""
        if self._train_dataset is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        return self._train_dataset
    
    def get_val_dataset(self) -> tf.data.Dataset:
        """Get the validation dataset."""
        if self._val_dataset is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        return self._val_dataset
    
    def get_test_dataset(self) -> tf.data.Dataset:
        """Get the test dataset."""
        if self._test_dataset is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        return self._test_dataset
    
    def get_class_weights(self) -> Dict[int, float]:
        """Get the calculated class weights."""
        if self.class_weights is None:
            raise ValueError("Class weights not calculated.")
        return self.class_weights
    
    def get_test_labels(self) -> np.ndarray:
        """Get the true labels for the test set."""
        return np.array([l for _, l in self.test_paths])
    
    def get_test_paths(self) -> List[str]:
        """Get the file paths for the test set."""
        return [str(p) for p, _ in self.test_paths]
    
    def get_num_samples(self) -> Dict[str, int]:
        """Get number of samples in each split."""
        return {
            'train': len(self.train_paths),
            'val': len(self.val_paths),
            'test': len(self.test_paths)
        }
