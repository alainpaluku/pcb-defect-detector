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
    name: str = "yolov8s.pt"  # Small model - meilleur équilibre
    img_size: int = 640
    batch_size: int = 16
    epochs: int = 100  # Plus d'époques pour convergence
    patience: int = 20  # Early stopping
    learning_rate: float = 0.0005  # LR plus bas pour stabilité
    optimizer: str = "AdamW"
    augment: bool = True
    mosaic: float = 1.0  # Mosaic complet pour mieux apprendre
    mixup: float = 0.15
    warmup_epochs: float = 5.0  # Plus de warmup
    weight_decay: float = 0.0005
    dropout: float = 0.1  # Régularisation
    close_mosaic: int = 15
    workers: int = 4
    cache: str = "ram"


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
    
    NUM_CLASSES: int = len(CLASS_NAMES)
    
    # Mapping des noms de classes (gère différentes conventions de casse)
    CLASS_MAP: Dict[str, int] = {}
    
    @classmethod
    def _init_class_map(cls) -> None:
        """Initialise le mapping des classes avec différentes variantes de casse."""
        for idx, name in enumerate(cls.CLASS_NAMES):
            variants = {
                name,                                      # missing_hole
                name.upper(),                              # MISSING_HOLE
                name.title(),                              # Missing_Hole
                name.replace("_", " ").title().replace(" ", "_"),  # Missing_Hole
            }
            for variant in variants:
                cls.CLASS_MAP[variant] = idx
    
    @classmethod
    def get_class_id(cls, class_name: str) -> int:
        """Retourne l'ID de classe, insensible à la casse."""
        # Recherche directe
        if class_name in cls.CLASS_MAP:
            return cls.CLASS_MAP[class_name]
        # Recherche insensible à la casse
        lower_name = class_name.lower().replace(" ", "_")
        for name, idx in cls.CLASS_MAP.items():
            if name.lower() == lower_name:
                return idx
        raise ValueError(f"Classe inconnue: {class_name}")
    
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


# Initialiser le mapping des classes au chargement du module
Config._init_class_map()
