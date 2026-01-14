"""Utility functions for PCB Defect Detection."""

from pathlib import Path
from typing import List, Dict


def count_images(directory: Path, formats: tuple = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")) -> int:
    """Count images in a directory."""
    count = 0
    for fmt in formats:
        count += len(list(directory.glob(fmt)))
    return count


def get_all_images(directory: Path, formats: tuple = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")) -> List[Path]:
    """Get all image files from a directory."""
    images = []
    for fmt in formats:
        images.extend(list(directory.glob(fmt)))
    return images


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def print_section_header(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_detection_results(detections: List[Dict]) -> None:
    """Print detection results in a formatted way."""
    if not detections:
        print("No defects detected.")
        return
    
    print(f"\nDetected {len(detections)} defect(s):")
    for i, det in enumerate(detections, 1):
        bbox = det["bbox"]
        print(f"  {i}. {det['class_name']}: {det['confidence']:.2%}")
        print(f"     Box: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) - ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
