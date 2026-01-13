"""
Configuration module for PCB Defect Detection System.

This module centralizes all hyperparameters, paths, and settings to ensure
reproducibility and easy tuning for industrial deployment.
"""

import os
from pathlib import Path

try:
    import tensorflow as tf
except ImportError:
    tf = None


class Config:
    """
    Central configuration class for the PCB defect detection pipeline.
    
    Attributes:
        KAGGLE_DATASET (str): Kaggle dataset identifier
        IMG_SIZE (tuple): Target image dimensions for model input
        BATCH_SIZE (int): Training batch size
        EPOCHS (int): Number of training epochs
        LEARNING_RATE (float): Initial learning rate for optimizer
        VALIDATION_SPLIT (float): Fraction of data for validation
        RANDOM_SEED (int): Seed for reproducibility
    """
    
    # Dataset Configuration
    KAGGLE_DATASET = "akhatova/pcb-defects"
    
    # Model Hyperparameters
    IMG_SIZE = (224, 224)  # MobileNetV2 standard input
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Performance Optimization
    USE_MIXED_PRECISION = True  # Enable for GPU speedup
    PREFETCH_BUFFER = tf.data.AUTOTUNE if 'tf' in dir() else 2
    CACHE_DATA = True  # Cache dataset in memory for faster training
    
    # Data Augmentation Parameters (simulate conveyor belt variations)
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    ZOOM_RANGE = 0.15
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = True
    
    # Training Configuration
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-7
    
    # Path Detection (Kaggle vs Local)
    @staticmethod
    def get_data_path():
        """
        Automatically detect execution environment and return appropriate data path.
        
        Returns:
            Path: Path object pointing to dataset directory
        """
        kaggle_path = Path("/kaggle/input/pcb-defects")
        if kaggle_path.exists():
            return kaggle_path
        else:
            # Local environment - assumes data downloaded to project root
            return Path("data/pcb-defects")
    
    @staticmethod
    def get_output_path():
        """
        Get output directory for models and results.
        
        Returns:
            Path: Path object for output directory
        """
        kaggle_output = Path("/kaggle/working")
        if kaggle_output.exists():
            return kaggle_output
        else:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            return output_dir
    
    @staticmethod
    def is_kaggle_environment():
        """
        Check if code is running in Kaggle environment.
        
        Returns:
            bool: True if running on Kaggle, False otherwise
        """
        return os.path.exists("/kaggle/input")
