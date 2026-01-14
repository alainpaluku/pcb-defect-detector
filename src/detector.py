"""Inference module for PCB Defect Detection."""

import numpy as np
from pathlib import Path
from src.config import Config


class PCBInspector:
    """High-level interface for PCB defect inspection."""
    
    def __init__(self, model_path=None):
        """Initialize inspector.
        
        Args:
            model_path: Path to trained model (.pt file)
        """
        self.model_path = model_path or self._find_model()
        self.model = None
        self._load_model()
    
    def _find_model(self):
        """Find trained model in output directory."""
        output = Config.get_output_path()
        
        candidates = [
            output / "pcb_model.pt",
            output / "pcb_yolo" / "weights" / "best.pt",
            "pcb_model.pt",
        ]
        
        for path in candidates:
            if Path(path).exists():
                return str(path)
        
        raise FileNotFoundError("No trained model found. Train a model first.")
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"Model loaded: {self.model_path}")
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
    
    def inspect(self, image_path, conf=None, save=False):
        """Inspect a single image for defects.
        
        Args:
            image_path: Path to image file
            conf: Confidence threshold
            save: Save annotated image
        
        Returns:
            List of detected defects with bounding boxes
        """
        conf = conf or Config.CONF_THRESHOLD
        
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=Config.IOU_THRESHOLD,
            save=save,
            verbose=False,
        )
        
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
    
    def inspect_batch(self, image_dir, conf=None, save=False):
        """Inspect multiple images.
        
        Args:
            image_dir: Directory containing images
            conf: Confidence threshold
            save: Save annotated images
        
        Returns:
            Dictionary mapping image names to detections
        """
        image_dir = Path(image_dir)
        results = {}
        
        for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG"]:
            for img_path in image_dir.glob(ext):
                detections = self.inspect(img_path, conf=conf, save=save)
                results[img_path.name] = detections
        
        return results
    
    def get_summary(self, detections):
        """Get summary of detections.
        
        Args:
            detections: List of detections from inspect()
        
        Returns:
            Summary dictionary
        """
        if not detections:
            return {"status": "OK", "defect_count": 0, "defects": []}
        
        defect_counts = {}
        for det in detections:
            cls_name = det["class_name"]
            defect_counts[cls_name] = defect_counts.get(cls_name, 0) + 1
        
        return {
            "status": "DEFECT",
            "defect_count": len(detections),
            "defects": defect_counts,
            "max_confidence": max(d["confidence"] for d in detections),
        }
    
    def visualize(self, image_path, output_path=None):
        """Visualize detections on image.
        
        Args:
            image_path: Path to image
            output_path: Path to save annotated image
        
        Returns:
            Annotated image array
        """
        results = self.model.predict(
            source=image_path,
            conf=Config.CONF_THRESHOLD,
            save=output_path is not None,
            project=str(Path(output_path).parent) if output_path else None,
            name=Path(output_path).stem if output_path else None,
        )
        
        # Get annotated image
        annotated = results[0].plot()
        return annotated
