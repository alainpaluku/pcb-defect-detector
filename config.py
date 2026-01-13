"""
Configuration module for PCB Defect Detector pipeline.

This module contains all configurable parameters for the training pipeline,
including paths, model hyperparameters, and data processing settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List


@dataclass
class DataConfig:
    """Configuration for data processing and augmentation.
    
    Attributes:
        dataset_name: Kaggle dataset identifier.
        data_dir: Local directory for storing downloaded data.
        image_size: Target size for image resizing (height, width).
        batch_size: Number of samples per training batch.
        validation_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        random_seed: Seed for reproducibility.
    """
    dataset_name: str = "akhatova/pcb-defects"
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for model architecture.
    
    Attributes:
        base_model: Name of the pretrained base model.
        dropout_rate: Dropout rate for regularization.
        dense_units: Number of units in dense layers.
        freeze_base: Whether to freeze base model weights.
    """
    base_model: str = "MobileNetV2"
    dropout_rate: float = 0.5
    dense_units: int = 256
    freeze_base: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training process.
    
    Attributes:
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate.
        early_stopping_patience: Epochs to wait before early stopping.
        reduce_lr_patience: Epochs to wait before reducing learning rate.
        reduce_lr_factor: Factor to reduce learning rate by.
        min_learning_rate: Minimum learning rate threshold.
        checkpoint_dir: Directory for saving model checkpoints.
    """
    epochs: int = 50
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.2
    min_learning_rate: float = 1e-7
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))


@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configurations.
    
    Attributes:
        data: Data processing configuration.
        model: Model architecture configuration.
        training: Training process configuration.
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
