"""
Utility functions for PCB Defect Detector.
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_config(config: Any, path: Path) -> None:
    """Save configuration to JSON."""
    def convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj
    
    config_dict = convert(asdict(config))
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
    """Count model parameters."""
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    total = sum(tf.keras.backend.count_params(w) for w in model.weights)
    return {'trainable': trainable, 'total': total, 'non_trainable': total - trainable}


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def get_class_distribution(labels: np.ndarray, class_names: List[str]) -> Dict[str, int]:
    """Get distribution of samples per class."""
    unique, counts = np.unique(labels, return_counts=True)
    return {class_names[i]: int(counts[j]) for j, i in enumerate(unique)}
