"""Configuration for PCB Defect Detection System."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Constantes globales
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


@dataclass
class ModelConfig:
    """Configuration du modèle YOLOv8."""
    name: str = "yolov8n.pt"
    img_size: int = 640
    batch_size: int = 16
    epochs: int = 50
    patience: int = 10
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    augment: bool = True
    mosaic: float = 0.5
    mixup: float = 0.1


@dataclass
class InferenceConfig:
    """Configuration pour l'inférence."""
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45


@dataclass
class DataConfig:
    """Configuration des données."""
    kaggle_dataset: str = "akhatova/pcb-defects"
    val_split: float = 0.2
    random_seed: int = 42


class Config:
    """Configuration centrale pour la détection de défauts PCB."""
    
    CLASS_NAMES: List[str] = [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper"
    ]
    
    # Mapping des noms de classes (gère différentes conventions de casse)
    CLASS_MAP: Dict[str, int] = {}
    for idx, name in enumerate(CLASS_NAMES):
        CLASS_MAP[name] = idx                                    # missing_hole
        CLASS_MAP[name.title().replace("_", "_")] = idx          # Missing_Hole (title case)
        CLASS_MAP[name.replace("_", " ").title().replace(" ", "_")] = idx  # Missing_hole
        CLASS_MAP[name.upper()] = idx                            # MISSING_HOLE
    
    NUM_CLASSES: int = len(CLASS_NAMES)
    
    # Configurations par défaut
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()
    data: DataConfig = DataConfig()
    
    @staticmethod
    def is_kaggle() -> bool:
        """Vérifie si on est dans l'environnement Kaggle."""
        return os.path.exists("/kaggle/input")
    
    @staticmethod
    def get_data_path() -> Path:
        """Retourne le chemin du dataset."""
        if Config.is_kaggle():
            for p in ("/kaggle/input/pcb-defects", "/kaggle/input/pcbdefects"):
                if Path(p).exists():
                    return Path(p)
        return Path("data/pcb-defects")
    
    @staticmethod
    def get_output_path() -> Path:
        """Retourne le répertoire de sortie."""
        if Config.is_kaggle():
            return Path("/kaggle/working")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @staticmethod
    def get_yolo_dataset_path() -> Path:
        """Retourne le chemin du dataset au format YOLO."""
        return Config.get_output_path() / "yolo_dataset"
    
    @classmethod
    def create(
        cls,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        img_size: Optional[int] = None,
        conf_threshold: Optional[float] = None,
    ) -> "Config":
        """Crée une configuration personnalisée."""
        config = cls()
        if epochs is not None:
            config.model.epochs = epochs
        if batch_size is not None:
            config.model.batch_size = batch_size
        if img_size is not None:
            config.model.img_size = img_size
        if conf_threshold is not None:
            config.inference.conf_threshold = conf_threshold
        return config
