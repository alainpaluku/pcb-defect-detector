"""Configuration for PCB Defect Detection System."""

import os
from pathlib import Path


class Config:
    """Central configuration for PCB defect detection with YOLOv8."""
    
    # Dataset
    KAGGLE_DATASET = "akhatova/pcb-defects"
    
    # Classes (6 types of defects)
    CLASS_NAMES = [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper"
    ]
    
    # Class name mapping (handles different naming conventions)
    CLASS_MAP = {
        "missing_hole": 0, "Missing_hole": 0, "MISSING_HOLE": 0,
        "mouse_bite": 1, "Mouse_bite": 1, "MOUSE_BITE": 1,
        "open_circuit": 2, "Open_circuit": 2, "OPEN_CIRCUIT": 2,
        "short": 3, "Short": 3, "SHORT": 3,
        "spur": 4, "Spur": 4, "SPUR": 4,
        "spurious_copper": 5, "Spurious_copper": 5, "SPURIOUS_COPPER": 5,
    }
    
    NUM_CLASSES = 6
    
    # Model settings
    MODEL_NAME = "yolov8n.pt"  # nano version (fast), alternatives: yolov8s.pt, yolov8m.pt
    IMG_SIZE = 640
    BATCH_SIZE = 16
    EPOCHS = 50
    PATIENCE = 10
    
    # Training
    LEARNING_RATE = 0.001
    OPTIMIZER = "Adam"
    AUGMENT = True
    MOSAIC = 0.5
    MIXUP = 0.1
    
    # Validation split
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Confidence threshold for inference
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    
    @staticmethod
    def is_kaggle():
        """Check if running in Kaggle environment."""
        return os.path.exists("/kaggle/input")
    
    @staticmethod
    def get_data_path():
        """Return dataset path."""
        if Config.is_kaggle():
            paths = [
                "/kaggle/input/pcb-defects",
                "/kaggle/input/pcbdefects",
            ]
            for p in paths:
                if Path(p).exists():
                    return Path(p)
        return Path("data/pcb-defects")
    
    @staticmethod
    def get_output_path():
        """Return output directory path."""
        if Config.is_kaggle():
            return Path("/kaggle/working")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @staticmethod
    def get_yolo_dataset_path():
        """Return YOLO formatted dataset path."""
        return Config.get_output_path() / "yolo_dataset"
