"""Inference module for PCB Defect Detection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import Config
from src.model import PCBDetector, ModelLoadError
from src.utils import get_all_images, get_logger

logger = get_logger(__name__)


class PCBInspector:
    """Interface haut niveau pour l'inspection de défauts PCB."""
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None
    ):
        """Initialise l'inspecteur.
        
        Args:
            model_path: Chemin vers le modèle entraîné (.pt)
            config: Configuration personnalisée
        """
        self.config = config or Config()
        self.model_path = model_path or self._find_model()
        self._detector: Optional[PCBDetector] = None
    
    def _find_model(self) -> str:
        """Recherche un modèle entraîné dans le répertoire de sortie."""
        output = Config.get_output_path()
        
        candidates = [
            output / "pcb_model.pt",
            output / "pcb_yolo" / "weights" / "best.pt",
            Path("pcb_model.pt"),
        ]
        
        for path in candidates:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            "Aucun modèle entraîné trouvé. Entraînez d'abord un modèle."
        )
    
    @property
    def detector(self) -> PCBDetector:
        """Lazy loading du détecteur."""
        if self._detector is None:
            self._detector = PCBDetector(
                model_path=self.model_path,
                config=self.config
            )
        return self._detector
    
    def inspect(
        self,
        image_path: Union[str, Path],
        conf: Optional[float] = None,
        save: bool = False
    ) -> List[Dict[str, Any]]:
        """Inspecte une image pour détecter les défauts.
        
        Args:
            image_path: Chemin vers le fichier image
            conf: Seuil de confiance
            save: Sauvegarder l'image annotée
        
        Returns:
            Liste des défauts détectés avec leurs bounding boxes
        """
        conf = conf or self.config.inference.conf_threshold
        
        results = self.detector.predict(
            source=image_path,
            conf=conf,
            save=save,
        )
        
        return self._parse_results(results)
    
    def _parse_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Parse les résultats YOLO en format structuré."""
        detections = []
        
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": Config.CLASS_NAMES[cls_id],
                    "confidence": confidence,
                    "bbox": {
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                    }
                })
        
        return detections
    
    def inspect_batch(
        self,
        image_dir: Union[str, Path],
        conf: Optional[float] = None,
        save: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Inspecte plusieurs images.
        
        Args:
            image_dir: Répertoire contenant les images
            conf: Seuil de confiance
            save: Sauvegarder les images annotées
        
        Returns:
            Dictionnaire associant les noms d'images aux détections
        """
        image_dir = Path(image_dir)
        images = get_all_images(image_dir)
        
        results = {}
        for img_path in images:
            detections = self.inspect(img_path, conf=conf, save=save)
            results[img_path.name] = detections
        
        return results
    
    @staticmethod
    def get_summary(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Génère un résumé des détections.
        
        Args:
            detections: Liste des détections de inspect()
        
        Returns:
            Dictionnaire de résumé
        """
        if not detections:
            return {"status": "OK", "defect_count": 0, "defects": {}}
        
        defect_counts: Dict[str, int] = {}
        for det in detections:
            cls_name = det["class_name"]
            defect_counts[cls_name] = defect_counts.get(cls_name, 0) + 1
        
        return {
            "status": "DEFECT",
            "defect_count": len(detections),
            "defects": defect_counts,
            "max_confidence": max(d["confidence"] for d in detections),
        }
    
    def visualize(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Any:
        """Visualise les détections sur une image.
        
        Args:
            image_path: Chemin vers l'image
            output_path: Chemin pour sauvegarder l'image annotée
        
        Returns:
            Image annotée (numpy array)
        """
        results = self.detector.predict(
            source=image_path,
            save=output_path is not None,
        )
        
        return results[0].plot()
