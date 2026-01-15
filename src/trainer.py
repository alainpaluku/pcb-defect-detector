"""Training pipeline for PCB Defect Detection with YOLOv8."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import Config
from src.data_ingestion import DataIngestion
from src.model import PCBDetector
from src.utils import format_metrics, get_logger, print_section_header

logger = get_logger(__name__)


class DatasetError(Exception):
    """Erreur liée au dataset."""
    pass


class TrainingManager:
    """Gère le pipeline complet d'entraînement."""
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        config: Optional[Config] = None
    ):
        """Initialise le gestionnaire d'entraînement.
        
        Args:
            data_path: Chemin vers le dataset (auto-détecté si None)
            config: Configuration personnalisée
        """
        self.data_path = data_path
        self.config = config or Config()
        self.output_path = Config.get_output_path()
        self.data: Optional[DataIngestion] = None
        self.model: Optional[PCBDetector] = None
        self.metrics: Dict[str, float] = {}
        
        self._print_header()
    
    def _print_header(self) -> None:
        """Affiche les informations système."""
        print_section_header("PCB DEFECT DETECTION - YOLOv8")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Environnement: {'Kaggle' if Config.is_kaggle() else 'Local'}")
        logger.info(f"Sortie: {self.output_path}")
    
    def setup_data(self) -> DataIngestion:
        """Configure le pipeline de données."""
        logger.info("[1/4] Configuration des données...")
        
        self.data = DataIngestion(data_path=self.data_path)
        
        if not self.data.find_data_structure():
            raise DatasetError(f"Dataset non trouvé à {self.data.data_path}")
        
        self.data.collect_images()
        
        if not self.data.all_images:
            raise DatasetError("Aucune image trouvée dans le dataset")
        
        stats = self.data.get_stats()
        logger.info(f"Images: {stats['total_images']}")
        logger.info(f"  Avec annotations XML: {stats['with_xml']}")
        logger.info(f"  Depuis dossiers de classes: {stats['from_folders']}")
        
        self.data.create_yolo_dataset()
        
        return self.data
    
    def setup_model(self) -> PCBDetector:
        """Initialise le modèle."""
        logger.info("[2/4] Configuration du modèle...")
        
        self.model = PCBDetector(config=self.config)
        return self.model
    
    def train(self, epochs: Optional[int] = None) -> Any:
        """Entraîne le modèle.
        
        Args:
            epochs: Nombre d'époques (utilise Config si None)
        """
        logger.info("[3/4] Entraînement...")
        
        if self.data is None:
            raise RuntimeError("Appelez setup_data() d'abord")
        if self.model is None:
            raise RuntimeError("Appelez setup_model() d'abord")
        
        yaml_path = self.data.get_yaml_path()
        
        return self.model.train(
            data_yaml=yaml_path,
            epochs=epochs,
            project=str(self.output_path),
            name="pcb_yolo"
        )
    
    def evaluate(self) -> Dict[str, float]:
        """Évalue le modèle."""
        logger.info("[4/4] Évaluation...")
        
        if self.data is None or self.model is None:
            raise RuntimeError("Appelez setup_data() et setup_model() d'abord")
        
        yaml_path = self.data.get_yaml_path()
        results = self.model.validate(data_yaml=yaml_path)
        
        self.metrics = PCBDetector.extract_metrics(results)
        
        logger.info("Résultats:")
        print(format_metrics(self.metrics))
        
        return self.metrics
    
    def save_model(self) -> Optional[Path]:
        """Sauvegarde le modèle entraîné."""
        logger.info("Sauvegarde du modèle...")
        
        best_model = self.output_path / "pcb_yolo" / "weights" / "best.pt"
        if not best_model.exists():
            logger.warning("Modèle best.pt non trouvé")
            return None
        
        dst = self.output_path / "pcb_model.pt"
        shutil.copy(best_model, dst)
        logger.info(f"Modèle sauvegardé: {dst}")
        
        # Export ONNX - important pour l'application desktop
        onnx_path = None
        try:
            if self.model:
                # Charger le meilleur modèle pour l'export
                from src.model import PCBDetector
                best_detector = PCBDetector(model_path=str(best_model))
                onnx_path = best_detector.export(format="onnx")
                
                # Copier le modèle ONNX vers le répertoire de sortie
                onnx_dst = self.output_path / "pcb_model.onnx"
                if onnx_path and onnx_path.exists():
                    shutil.copy(onnx_path, onnx_dst)
                    logger.info(f"Modèle ONNX sauvegardé: {onnx_dst}")
        except Exception as e:
            logger.warning(f"Export ONNX échoué: {e}")
            logger.info("L'application desktop fonctionnera en mode démo sans le modèle ONNX")
        
        return dst
    
    def run_pipeline(self, epochs: Optional[int] = None) -> Dict[str, float]:
        """Exécute le pipeline complet d'entraînement.
        
        Args:
            epochs: Nombre d'époques d'entraînement
        
        Returns:
            Dictionnaire des métriques
        """
        self.setup_data()
        self.setup_model()
        self.train(epochs=epochs)
        self.evaluate()
        self.save_model()
        
        print_section_header("ENTRAÎNEMENT TERMINÉ!")
        logger.info(f"mAP@50: {self.metrics.get('mAP50', 0):.4f}")
        logger.info(f"Sortie: {self.output_path}")
        
        return self.metrics
