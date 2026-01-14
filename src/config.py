"""Configuration for PCB Defect Detection System."""

import os
from pathlib import Path


class Config:
    """Central configuration for the PCB defect detection pipeline."""
    
    # Dataset
    KAGGLE_DATASET = "akhatova/pcb-defects"
    
    # Classes de défauts - format standard (lowercase avec underscore)
    DEFECT_CLASSES = [
        "missing_hole",
        "mouse_bite", 
        "open_circuit",
        "short",
        "spur",
        "spurious_copper"
    ]
    
    # Tous les noms possibles (lowercase, CamelCase, alternatives)
    DEFECT_CLASSES_ALT = [
        # Standard lowercase
        "missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper",
        # CamelCase (Kaggle akhatova/pcb-defects)
        "Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper",
        # Alternatives
        "pin_hole", "mousebite", "open",
        # Uppercase
        "MISSING_HOLE", "MOUSE_BITE", "OPEN_CIRCUIT", "SHORT", "SPUR", "SPURIOUS_COPPER",
    ]
    
    NUM_CLASSES = 6
    
    # Model
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001  # Plus haut pour transfer learning
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    RANDOM_SEED = 42
    
    # Augmentation - Modéré
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    ZOOM_RANGE = 0.1
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = True
    BRIGHTNESS_RANGE = (0.9, 1.1)
    SHEAR_RANGE = 0.05
    FILL_MODE = 'nearest'
    
    # Training callbacks
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-7
    
    # Fine-tuning
    FINE_TUNE_EPOCHS = 10
    FINE_TUNE_LAYERS = 50
    FINE_TUNE_LR = 1e-4
    
    @staticmethod
    def is_kaggle():
        """Check if running in Kaggle environment."""
        return os.path.exists("/kaggle/input")
    
    @staticmethod
    def _has_class_folders(path):
        """Check if path contains class folders (supports both naming conventions)."""
        path = Path(path)
        if not path.exists():
            return False
        
        # Check both standard and alternative class names
        all_possible_classes = set(Config.DEFECT_CLASSES + Config.DEFECT_CLASSES_ALT)
        return any((path / cls).exists() for cls in all_possible_classes)
    
    @staticmethod
    def _find_data_in_path(base_path):
        """Recursively find the folder containing class directories."""
        base_path = Path(base_path)
        
        # Check if base path has class folders
        if Config._has_class_folders(base_path):
            return base_path
        
        # Check subdirectories (max 2 levels deep)
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir():
                    if Config._has_class_folders(item):
                        return item
                    # One more level
                    for subitem in item.iterdir():
                        if subitem.is_dir() and Config._has_class_folders(subitem):
                            return subitem
        
        return None
    
    @staticmethod
    def get_data_path():
        """Return dataset path (auto-detects Kaggle vs local)."""
        
        # On Kaggle
        if Config.is_kaggle():
            kaggle_input = Path("/kaggle/input")
            
            # Chemins connus en priorité
            known_paths = [
                "/kaggle/input/pcb-defects/PCB_DATASET/images",
                "/kaggle/input/pcb-defects/PCB_DATASET",
                "/kaggle/input/pcb-defects/images",
                "/kaggle/input/pcb-defects",
                "/kaggle/input/pcbdefects/PCB_DATASET/images",
                "/kaggle/input/pcbdefects/PCB_DATASET",
                "/kaggle/input/pcbdefects",
            ]
            
            for p in known_paths:
                path = Path(p)
                if path.exists() and Config._has_class_folders(path):
                    print(f"📁 Dataset found: {path}")
                    return path
            
            # Search all datasets and subdirectories
            if kaggle_input.exists():
                for dataset_folder in kaggle_input.iterdir():
                    if dataset_folder.is_dir():
                        # Check dataset root
                        if Config._has_class_folders(dataset_folder):
                            print(f"📁 Dataset found: {dataset_folder}")
                            return dataset_folder
                        
                        # Check all subdirectories (PCB_DATASET, images, etc.)
                        try:
                            for subdir in dataset_folder.iterdir():
                                if subdir.is_dir():
                                    if Config._has_class_folders(subdir):
                                        print(f"📁 Dataset found: {subdir}")
                                        return subdir
                                    # One more level deep
                                    try:
                                        for subsubdir in subdir.iterdir():
                                            if subsubdir.is_dir() and Config._has_class_folders(subsubdir):
                                                print(f"📁 Dataset found: {subsubdir}")
                                                return subsubdir
                                    except PermissionError:
                                        pass
                        except PermissionError:
                            pass
        
        # Local
        local_paths = [
            "data/pcb-defects",
            "data",
            "../data/pcb-defects",
        ]
        for p in local_paths:
            found = Config._find_data_in_path(p)
            if found:
                return found
        
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
    def get_device_info():
        """Get device information for logging."""
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return {
            'tensorflow_version': tf.__version__,
            'gpu_available': len(gpus) > 0,
            'gpu_devices': [g.name for g in gpus],
            'environment': 'Kaggle' if Config.is_kaggle() else 'Local'
        }
