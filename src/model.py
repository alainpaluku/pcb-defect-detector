"""YOLOv8 Model for PCB Defect Detection."""

from pathlib import Path
from src.config import Config


class PCBDetector:
    """YOLOv8-based detector for PCB defects."""
    
    def __init__(self, model_path=None):
        """Initialize detector.
        
        Args:
            model_path: Path to trained model or pretrained model name
        """
        self.model_path = model_path or Config.MODEL_NAME
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"Model loaded: {self.model_path}")
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
    
    def train(self, data_yaml, epochs=None, batch_size=None, img_size=None, 
              project=None, name="pcb_yolo"):
        """Train the model.
        
        Args:
            data_yaml: Path to dataset YAML config
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size
            project: Output project directory
            name: Experiment name
        
        Returns:
            Training results
        """
        epochs = epochs or Config.EPOCHS
        batch_size = batch_size or Config.BATCH_SIZE
        img_size = img_size or Config.IMG_SIZE
        project = project or str(Config.get_output_path())
        
        print(f"\nTraining YOLOv8 for {epochs} epochs...")
        print(f"Dataset: {data_yaml}")
        print(f"Batch size: {batch_size}, Image size: {img_size}")
        
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=Config.PATIENCE,
            save=True,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=True,
            optimizer=Config.OPTIMIZER,
            lr0=Config.LEARNING_RATE,
            augment=Config.AUGMENT,
            mosaic=Config.MOSAIC,
            mixup=Config.MIXUP,
            verbose=True,
        )
        
        return results
    
    def validate(self, data_yaml=None):
        """Validate the model.
        
        Args:
            data_yaml: Path to dataset YAML config
        
        Returns:
            Validation metrics
        """
        metrics = self.model.val(data=data_yaml)
        return metrics
    
    def predict(self, source, conf=None, iou=None, save=True, show=False):
        """Run inference on images.
        
        Args:
            source: Image path, directory, or URL
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Save results
            show: Display results
        
        Returns:
            Detection results
        """
        conf = conf or Config.CONF_THRESHOLD
        iou = iou or Config.IOU_THRESHOLD
        
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            show=show,
        )
        
        return results
    
    def export(self, format="onnx"):
        """Export model to different formats.
        
        Args:
            format: Export format (onnx, torchscript, tflite, etc.)
        
        Returns:
            Path to exported model
        """
        print(f"Exporting model to {format}...")
        path = self.model.export(format=format)
        print(f"Exported: {path}")
        return path
    
    def get_metrics(self, results):
        """Extract metrics from validation results.
        
        Args:
            results: Validation results from model.val()
        
        Returns:
            Dictionary of metrics
        """
        return {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
    
    @staticmethod
    def load_trained(model_path):
        """Load a trained model.
        
        Args:
            model_path: Path to trained .pt file
        
        Returns:
            PCBDetector instance
        """
        return PCBDetector(model_path=model_path)
