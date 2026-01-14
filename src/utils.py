"""Utility functions for PCB Defect Detection."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def count_images(directory: Path, formats: tuple = ("*.jpg", "*.jpeg", "*.png", "*.bmp")) -> int:
    """Count images in a directory.
    
    Args:
        directory: Path to directory
        formats: Tuple of file patterns to match
        
    Returns:
        Number of images found
    """
    count = 0
    for fmt in formats:
        count += len(list(directory.glob(fmt)))
    return count


def get_all_images(directory: Path, formats: tuple = ("*.jpg", "*.jpeg", "*.png", "*.bmp")) -> List[Path]:
    """Get all image files from a directory.
    
    Args:
        directory: Path to directory
        formats: Tuple of file patterns to match
        
    Returns:
        List of image paths
    """
    images = []
    for fmt in formats:
        images.extend(list(directory.glob(fmt)))
    return images


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "14.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def print_section_header(title: str, width: int = 60) -> None:
    """Print a formatted section header.
    
    Args:
        title: Section title
        width: Width of the header
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection(title: str, width: int = 60) -> None:
    """Print a formatted subsection header.
    
    Args:
        title: Subsection title
        width: Width of the header
    """
    print("-" * width)
    print(title)
    print("-" * width)


def calculate_steps_per_epoch(num_samples: int, batch_size: int) -> int:
    """Calculate steps per epoch.
    
    Args:
        num_samples: Number of samples
        batch_size: Batch size
        
    Returns:
        Steps per epoch (minimum 1)
    """
    return max(1, num_samples // batch_size)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Apply softmax to logits.
    
    Args:
        logits: Input logits
        
    Returns:
        Softmax probabilities
    """
    max_logit = np.max(logits)
    exp_scores = np.exp(logits - max_logit)
    return exp_scores / np.sum(exp_scores)
