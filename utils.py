"""
Utility functions for PCB Defect Detector.

This module contains helper functions for various tasks including
visualization, file operations, and model utilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    

def get_model_summary(model: Model) -> str:
    """Get model summary as string.
    
    Args:
        model: Keras Model.
    
    Returns:
        Model summary string.
    """
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return '\n'.join(string_list)


def save_training_config(config: Any, path: Path) -> None:
    """Save training configuration to JSON file.
    
    Args:
        config: Configuration object (dataclass).
        path: Path to save JSON file.
    """
    from dataclasses import asdict
    
    def convert_path(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_path(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_path(item) for item in obj]
        return obj
    
    config_dict = convert_path(asdict(config))
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_training_config(path: Path) -> Dict[str, Any]:
    """Load training configuration from JSON file.
    
    Args:
        path: Path to JSON configuration file.
    
    Returns:
        Configuration dictionary.
    """
    with open(path, 'r') as f:
        return json.load(f)


def count_parameters(model: Model) -> Dict[str, int]:
    """Count trainable and non-trainable parameters.
    
    Args:
        model: Keras Model.
    
    Returns:
        Dictionary with parameter counts.
    """
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    
    return {
        'trainable': trainable,
        'non_trainable': non_trainable,
        'total': trainable + non_trainable
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds.
    
    Returns:
        Formatted time string (e.g., "1h 23m 45s").
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return ' '.join(parts)


def get_class_distribution(labels: np.ndarray, class_names: List[str]) -> Dict[str, int]:
    """Get distribution of samples across classes.
    
    Args:
        labels: Array of integer labels.
        class_names: List of class names.
    
    Returns:
        Dictionary mapping class names to sample counts.
    """
    unique, counts = np.unique(labels, return_counts=True)
    return {class_names[i]: int(counts[j]) 
            for j, i in enumerate(unique)}


class ProgressLogger(tf.keras.callbacks.Callback):
    """Custom callback for logging training progress.
    
    Attributes:
        logger: Logger instance.
        log_frequency: How often to log (in batches).
    """
    
    def __init__(self, log_frequency: int = 10) -> None:
        """Initialize ProgressLogger.
        
        Args:
            log_frequency: Log every N batches.
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_frequency = log_frequency
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of each epoch."""
        self.logger.info(f"Epoch {epoch + 1} started")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of each epoch."""
        if logs:
            metrics = ', '.join(f"{k}={v:.4f}" for k, v in logs.items())
            self.logger.info(f"Epoch {epoch + 1} completed: {metrics}")
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of each batch."""
        if batch % self.log_frequency == 0 and logs:
            loss = logs.get('loss', 0)
            acc = logs.get('accuracy', 0)
            self.logger.debug(f"Batch {batch}: loss={loss:.4f}, acc={acc:.4f}")
