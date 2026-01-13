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
        logger: Logger instance for this class.
        class_names: List of class names.
        class_weights: Computed class weights for imbalanced data.
        train_generator: Training data generator.
        val_generator: Validation data generator.
        test_generator: Test data generator.
    
    Example:
        >>> config = DataConfig()
        >>> pipeline = DataPipeline(config, class_names=['defect1', 'defect2'])
        >>> pipeline.prepare_data(class_images)
        >>> train_gen = pipeline.get_train_generator()
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
        
    def prepare_data(self, class_images: Dict[str, List[Path]]) -> None:
        """Prepare data by creating splits and calculating class weights.
        
        Args:
            class_images: Dictionary mapping class names to image paths.
        """
        self.logger.info("Preparing data splits...")
        
        # Create labeled dataset
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
        
        # Store as tuples of (path, label)
        self.train_paths = list(zip(train_paths, train_labels))
        self.val_paths = list(zip(val_paths, val_labels))
        self.test_paths = list(zip(test_paths, test_labels))
        
        self.logger.info(f"Train: {len(self.train_paths)}, "
                        f"Val: {len(self.val_paths)}, "
                        f"Test: {len(self.test_paths)}")
        
        # Calculate class weights
        self._calculate_class_weights(train_labels)
        
        # Create datasets
        self._create_datasets()

    def _calculate_class_weights(self, labels: np.ndarray) -> None:
        """Calculate class weights to handle imbalanced dataset.
        
        Uses sklearn's compute_class_weight with 'balanced' strategy to
        automatically calculate weights inversely proportional to class
        frequencies.
        
        Args:
            labels: Array of training labels.
        """
        self.logger.info("Calculating class weights for imbalanced data...")
        
        # Count samples per class
        class_counts = Counter(labels)
        self.logger.info("Class distribution:")
        for class_idx, count in sorted(class_counts.items()):
            class_name = self.class_names[class_idx]
            self.logger.info(f"  {class_name}: {count} samples")
        
        # Compute balanced class weights
        unique_classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=labels
        )
        
        self.class_weights = {int(cls): float(weight) 
                             for cls, weight in zip(unique_classes, weights)}
        
        self.logger.info("Class weights:")
        for class_idx, weight in self.class_weights.items():
            class_name = self.class_names[class_idx]
            self.logger.info(f"  {class_name}: {weight:.4f}")
    
    def _load_and_preprocess_image(self, path: str, label: int) -> Tuple[tf.Tensor, int]:
        """Load and preprocess a single image.
        
        Args:
            path: Path to the image file.
            label: Integer class label.
        
        Returns:
            Tuple of (preprocessed image tensor, label).
        """
        # Read and decode image
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        
        # Resize to target size
        img = tf.image.resize(img, self.config.image_size)
        
        # Normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        
        return img, label
    
    def _augment_image(self, image: tf.Tensor, label: int) -> Tuple[tf.Tensor, int]:
        """Apply data augmentation to training images.
        
        Args:
            image: Input image tensor.
            label: Integer class label.
        
        Returns:
            Tuple of (augmented image tensor, label).
        """
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random vertical flip
        image = tf.image.random_flip_up_down(image)
        
        # Random rotation (90 degree increments)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        
        # Random brightness adjustment
        image = tf.image.random_brightness(image, max_delta=0.2)
        
        # Random contrast adjustment
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        
        # Ensure values are still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def _create_datasets(self) -> None:
        """Create TensorFlow datasets for train, validation, and test."""
        self.logger.info("Creating TensorFlow datasets...")
        
        # Training dataset with augmentation
        train_paths = [str(p) for p, _ in self.train_paths]
        train_labels = [l for _, l in self.train_paths]
        
        self._train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_paths, train_labels)
        )
        self._train_dataset = (
            self._train_dataset
            .shuffle(buffer_size=len(train_paths), seed=self.config.random_seed)
            .map(self._load_and_preprocess_image, 
                 num_parallel_calls=tf.data.AUTOTUNE)
            .map(self._augment_image, 
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        # Validation dataset (no augmentation)
        val_paths = [str(p) for p, _ in self.val_paths]
        val_labels = [l for _, l in self.val_paths]
        
        self._val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_paths, val_labels)
        )
        self._val_dataset = (
            self._val_dataset
            .map(self._load_and_preprocess_image,
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        # Test dataset (no augmentation)
        test_paths = [str(p) for p, _ in self.test_paths]
        test_labels = [l for _, l in self.test_paths]
        
        self._test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_paths, test_labels)
        )
        self._test_dataset = (
            self._test_dataset
            .map(self._load_and_preprocess_image,
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    
    def get_train_dataset(self) -> tf.data.Dataset:
        """Get the training dataset.
        
        Returns:
            TensorFlow Dataset for training.
        
        Raises:
            ValueError: If data hasn't been prepared yet.
        """
        if self._train_dataset is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        return self._train_dataset
    
    def get_val_dataset(self) -> tf.data.Dataset:
        """Get the validation dataset.
        
        Returns:
            TensorFlow Dataset for validation.
        
        Raises:
            ValueError: If data hasn't been prepared yet.
        """
        if self._val_dataset is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        return self._val_dataset
    
    def get_test_dataset(self) -> tf.data.Dataset:
        """Get the test dataset.
        
        Returns:
            TensorFlow Dataset for testing.
        
        Raises:
            ValueError: If data hasn't been prepared yet.
        """
        if self._test_dataset is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        return self._test_dataset
    
    def get_class_weights(self) -> Dict[int, float]:
        """Get the calculated class weights.
        
        Returns:
            Dictionary mapping class indices to weights.
        
        Raises:
            ValueError: If class weights haven't been calculated yet.
        """
        if self.class_weights is None:
            raise ValueError("Class weights not calculated. Call prepare_data() first.")
        return self.class_weights
    
    def get_test_labels(self) -> np.ndarray:
        """Get the true labels for the test set.
        
        Returns:
            Array of test set labels.
        """
        return np.array([l for _, l in self.test_paths])
    
    def get_steps_per_epoch(self) -> int:
        """Calculate steps per epoch for training.
        
        Returns:
            Number of batches per training epoch.
        """
        return len(self.train_paths) // self.config.batch_size
    
    def get_validation_steps(self) -> int:
        """Calculate validation steps.
        
        Returns:
            Number of batches for validation.
        """
        return len(self.val_paths) // self.config.batch_size
