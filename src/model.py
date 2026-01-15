"""YOLOv8 Model for PCB Defect Detection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import Config, ModelConfig, InferenceConfig
from src.utils import get_logger

logger = get_logger(__name__)


class ModelLoadError(Exception):
    """Erreur lors du chargement du modèle."""
    pass


class YOLOWrapper:
    """Wrapper pour le modèle YOLO avec gestion des erreurs."""
    
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = str(model_path)
        self.model = self._load_model()
    
    def _load_model(self):
        """Charge le modèle YOLO."""
        try:
            from ultralytics import YOLO
            
            # Vérifier si le fichier existe (sauf pour les modèles pré-entraînés)
            model_file = Path(self.model_path)
            if not model_file.suffix == '.pt' or (model_file.exists() or self.model_path in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']):
                model = YOLO(self.model_path)
                logger.info(f"Modèle chargé: {self.model_path}")
                return model
            else:
                raise ModelLoadError(f"Fichier modèle non trouvé: {self.model_path}")
                
        except ImportError as e:
            raise ModelLoadError(
                "ultralytics non installé. Exécutez: pip install ultralytics"
            ) from e
        except Exception as e:
            raise ModelLoadError(f"Erreur de chargement du modèle: {e}") from e
    
    def __getattr__(self, name: str):
        """Délègue les appels au modèle sous-jacent."""
        return getattr(self.model, name)


class PCBDetector:
    """Détecteur YOLOv8 pour les défauts PCB."""
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None
    ):
        """Initialise le détecteur.
        
        Args:
            model_path: Chemin vers le modèle entraîné ou nom du modèle pré-entraîné
            config: Configuration personnalisée
        """
        self.config = config or Config()
        self.model_path = model_path or self.config.model.name
        self._model: Optional[YOLOWrapper] = None
    
    @property
    def model(self) -> YOLOWrapper:
        """Lazy loading du modèle."""
        if self._model is None:
            self._model = YOLOWrapper(self.model_path)
        return self._model
    
    def train(
        self,
        data_yaml: Union[str, Path],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        img_size: Optional[int] = None,
        project: Optional[str] = None,
        name: str = "pcb_yolo"
    ) -> Any:
        """Entraîne le modèle.
        
        Args:
            data_yaml: Chemin vers la config YAML du dataset
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille du batch
            img_size: Taille des images
            project: Répertoire de sortie
            name: Nom de l'expérience
        
        Returns:
            Résultats de l'entraînement
        """
        model_cfg = self.config.model
        epochs = epochs or model_cfg.epochs
        batch_size = batch_size or model_cfg.batch_size
        img_size = img_size or model_cfg.img_size
        project = project or str(Config.get_output_path())
        
        logger.info(f"Entraînement YOLOv8 pour {epochs} époques...")
        logger.info(f"Dataset: {data_yaml}")
        logger.info(f"Batch: {batch_size}, Image: {img_size}")
        logger.info(f"Modèle: {self.model_path}, Optimizer: {model_cfg.optimizer}")
        logger.info(f"LR: {model_cfg.learning_rate}, Patience: {model_cfg.patience}")
        
        return self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=model_cfg.patience,
            save=True,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=True,
            optimizer=model_cfg.optimizer,
            lr0=model_cfg.learning_rate,
            lrf=0.01,  # Learning rate final (fraction of lr0)
            augment=model_cfg.augment,
            mosaic=model_cfg.mosaic,
            mixup=model_cfg.mixup,
            # Nouveaux paramètres pour améliorer la précision
            warmup_epochs=model_cfg.warmup_epochs,
            weight_decay=model_cfg.weight_decay,
            dropout=model_cfg.dropout,
            close_mosaic=model_cfg.close_mosaic,
            # Paramètres supplémentaires
            hsv_h=0.015,  # Augmentation couleur
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,  # Rotation
            translate=0.1,  # Translation
            scale=0.5,  # Scale
            fliplr=0.5,  # Flip horizontal
            flipud=0.2,  # Flip vertical (utile pour PCB)
            verbose=True,
        )
    
    def validate(self, data_yaml: Optional[Union[str, Path]] = None) -> Any:
        """Valide le modèle."""
        return self.model.val(data=str(data_yaml) if data_yaml else None)
    
    def predict(
        self,
        source: Union[str, Path],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        save: bool = True,
        show: bool = False
    ) -> List[Any]:
        """Exécute l'inférence sur des images."""
        inf_cfg = self.config.inference
        return self.model.predict(
            source=str(source),
            conf=conf or inf_cfg.conf_threshold,
            iou=iou or inf_cfg.iou_threshold,
            save=save,
            show=show,
        )
    
    def export(self, format: str = "onnx") -> Path:
        """Exporte le modèle vers différents formats."""
        logger.info(f"Export du modèle vers {format}...")
        path = self.model.export(format=format)
        logger.info(f"Exporté: {path}")
        return Path(path)
    
    @staticmethod
    def extract_metrics(results: Any) -> Dict[str, float]:
        """Extrait les métriques des résultats de validation."""
        return {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
    
    @classmethod
    def load_trained(cls, model_path: Union[str, Path]) -> "PCBDetector":
        """Charge un modèle entraîné."""
        return cls(model_path=model_path)
