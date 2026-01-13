"""
Configuration module for PCB Defect Detector pipeline.

This module contains all configurable parameters for the training pipeline,
including paths, model hyperparameters, and data processing settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional
from enum import Enum


class Environment(Enum):
    """Execution environment enumeration."""
    LOCAL = "local"
    KAGGLE = "kaggle"
    COLAB = "colab"


def detect_environment() -> Environment:
    """Detect the current execution environment.
    
    Returns:
        Environment enum indicating current platform.
    """
    if os.path.exists("/kaggle/input"):
        return Environment.KAGGLE
    elif os.path.exists("/content"):
        return Environment.COLAB
    return Environment.LOCAL


def get_default_data_dir() -> Path:
    """Get default data directory based on environment.
    
    Returns:
        Path to data directory.
    """
    env = detect_environment()
    if env == Environment.KAGGLE:
        return Path("/kaggle/input/pcb-defects")
    elif env == Environment.COLAB:
        return Path("/content/data")
    return Path("./data")


def get_default_output_dir() -> Path:
    """Get default output directory based on environment.
    
    Returns:
        Path to output directory.
    """
    env = detect_environment()
    if env == Environment.KAGGLE:
        return Path("/kaggle/working")
    elif env == Environment.COLAB:
        return Path("/content/output")
    return Path(".")


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
        augmentation_strength: Strength of data augmentation (0.0-1.0).
    """
    dataset_name: str = "akhatova/pcb-defects"
    data_dir: Path = field(default_factory=get_default_data_dir)
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    augmentation_strength: float = 0.3


@dataclass
class ModelConfig:
    """Configuration for model architecture.
    
    Attributes:
        base_model: Name of the pretrained base model.
        dropout_rate: Dropout rate for regularization.
        dense_units: Number of units in dense layers.
        freeze_base: Whether to freeze base model weights initially.
        use_batch_norm: Whether to use batch normalization.
        l2_regularization: L2 regularization factor.
    """
    base_model: str = "MobileNetV2"
    dropout_rate: float = 0.5
    dense_units: int = 256
    freeze_base: bool = True
    use_batch_norm: bool = True
    l2_regularization: float = 0.01


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
        fine_tune_epochs: Additional epochs for fine-tuning.
        fine_tune_layers: Number of base layers to unfreeze for fine-tuning.
        fine_tune_lr: Learning rate for fine-tuning phase.
    """
    epochs: int = 30
    learning_rate: float = 1e-3
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7
    checkpoint_dir: Path = field(default_factory=lambda: get_default_output_dir() / "checkpoints")
    fine_tune_epochs: int = 20
    fine_tune_layers: int = 30
    fine_tune_lr: float = 1e-5


@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configurations.
    
    Attributes:
        data: Data processing configuration.
        model: Model architecture configuration.
        training: Training process configuration.
        results_dir: Directory for saving results.
        environment: Detected execution environment.
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    results_dir: Path = field(default_factory=lambda: get_default_output_dir() / "results")
    environment: Environment = field(default_factory=detect_environment)
    
    def __post_init__(self) -> None:
        """Ensure directories exist after initialization."""
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
