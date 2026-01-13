"""Configuration for PCB Defect Detection System."""

import os
from pathlib import Path


class Config:
    """Central configuration for the PCB defect detection pipeline."""
    
    # Dataset
    KAGGLE_DATASET = "akhatova/pcb-defects"
    DEFECT_CLASSES = [
        "missing_hole",
        "mouse_bite", 
        "open_circuit",
        "short",
        "spur",
        "spurious_copper"
    ]
    NUM_CLASSES = 6
    
    # Model
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001  # Lower LR for transfer learning
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1  # Hold-out test set
    RANDOM_SEED = 42
    
    # Augmentation - Optimisé pour PCB (images industrielles)
    ROTATION_RANGE = 15  # PCB ont des orientations limitées
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    ZOOM_RANGE = 0.1
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = True
    BRIGHTNESS_RANGE = (0.9, 1.1)  # Léger pour images industrielles
    SHEAR_RANGE = 0.05
    FILL_MODE = 'reflect'  # Meilleur pour PCB que 'nearest'
    
    # Training callbacks
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-7
    
    # Fine-tuning
    FINE_TUNE_EPOCHS = 20
    FINE_TUNE_LAYERS = 50  # Dernières couches à débloquer
    FINE_TUNE_LR = 1e-5
    
    @staticmethod
    def get_data_path():
        """Return dataset path (auto-detects Kaggle vs local).
        
        Dataset structure on Kaggle:
        /kaggle/input/pcb-defects/
        ├── missing_hole/
        ├── mouse_bite/
        ├── open_circuit/
        ├── short/
        ├── spur/
        └── spurious_copper/
        """
        from pathlib import Path
        
        # Chemins Kaggle possibles (ordre de priorité)
        kaggle_paths = [
            "/kaggle/input/pcb-defects",
            "/kaggle/input/pcb-defects/PCB_DATASET/images",
            "/kaggle/input/pcb-defects/images",
            "/kaggle/input/pcb-defects/PCB_DATASET",
        ]
        
        # Chercher dans les chemins Kaggle
        for p in kaggle_paths:
            path = Path(p)
            if path.exists():
                # Vérifier si contient les dossiers de classes
                if Config._has_class_folders(path):
                    print(f"📁 Dataset found at: {path}")
                    return path
                # Sinon chercher dans les sous-dossiers
                for subdir in path.iterdir():
                    if subdir.is_dir() and Config._has_class_folders(subdir):
                        print(f"📁 Dataset found at: {subdir}")
                        return subdir
        
        # Local
        local_path = Path("data/pcb-defects")
        if local_path.exists():
            return local_path
            
        return Path("data/pcb-defects")
    
    @staticmethod
    def _has_class_folders(path):
        """Check if path contains class folders."""
        return any((path / cls).exists() for cls in Config.DEFECT_CLASSES)
    
    @staticmethod
    def get_output_path():
        """Return output directory path."""
        if Config.is_kaggle():
            return Path("/kaggle/working")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @staticmethod
    def is_kaggle():
        """Check if running in Kaggle environment."""
        return os.path.exists("/kaggle/input")
    
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
