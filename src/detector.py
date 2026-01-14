"""Object Detection model for PCB Defect Detection using YOLOv8."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not installed. Run: pip install ultralytics")

from src.config import Config


class BoundingBox:
    """Represents a detected defect bounding box."""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, 
                 class_id: int, class_name: str, confidence: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
    
    @property
    def center(self) -> tuple:
        """Get center point of the box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'x1': round(self.x1, 2),
            'y1': round(self.y1, 2),
            'x2': round(self.x2, 2),
            'y2': round(self.y2, 2),
            'width': round(self.width, 2),
            'height': round(self.height, 2),
            'center': {'x': round(self.center[0], 2), 'y': round(self.center[1], 2)},
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': round(self.confidence, 4)
        }


class PCBDetector:
    """YOLOv8-based detector for PCB defect localization.
    
    This model detects AND localizes defects on PCB images,
    returning bounding boxes around each detected defect.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.model = None
        self.class_names = Config.DEFECT_CLASSES
        
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load a trained YOLO model.
        
        Args:
            model_path: Path to .pt model file
        """
        path = model_path or self.model_path
        if path and Path(path).exists():
            self.model = YOLO(path)
            print(f"✅ Detector model loaded: {path}")
        else:
            # Load pretrained YOLOv8n for fine-tuning
            self.model = YOLO('yolov8n.pt')
            print("✅ Loaded pretrained YOLOv8n (ready for fine-tuning)")
    
    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 640,
              batch: int = 16, project: str = 'output', name: str = 'pcb_detector') -> None:
        """Train the detector on PCB defect dataset.
        
        Args:
            data_yaml: Path to YOLO format data.yaml
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            project: Output project directory
            name: Experiment name
        """
        if self.model is None:
            self.load_model()
        
        print(f"\n🚀 Starting YOLO training...")
        print(f"   Dataset: {data_yaml}")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            patience=20,
            save=True,
            plots=True,
            verbose=True
        )
        
        print(f"✅ Training complete! Model saved to: {project}/{name}")
        return results
    
    def detect(self, image_path: str, conf_threshold: float = 0.25,
               iou_threshold: float = 0.45) -> List[BoundingBox]:
        """Detect defects in an image.
        
        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of BoundingBox objects for detected defects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                    
                    detections.append(BoundingBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf
                    ))
        
        return detections
    
    def detect_and_visualize(self, image_path: str, output_path: Optional[str] = None,
                             conf_threshold: float = 0.25) -> tuple:
        """Detect defects and save annotated image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)
            conf_threshold: Confidence threshold
            
        Returns:
            Tuple of (detections list, annotated image array)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )
        
        # Get annotated image
        annotated = results[0].plot()
        
        # Save if output path provided
        if output_path:
            import cv2
            cv2.imwrite(output_path, annotated)
            print(f"💾 Annotated image saved: {output_path}")
        
        # Extract detections
        detections = self.detect(image_path, conf_threshold)
        
        return detections, annotated
    
    def export_onnx(self, output_path: str = 'output/pcb_detector.onnx',
                    imgsz: int = 640, simplify: bool = True) -> str:
        """Export model to ONNX format for web deployment.
        
        Args:
            output_path: Output ONNX file path
            imgsz: Image size for export
            simplify: Whether to simplify the ONNX graph
            
        Returns:
            Path to exported ONNX model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"📦 Exporting to ONNX...")
        
        export_path = self.model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=simplify,
            opset=12
        )
        
        # Move to desired location
        from shutil import move
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        move(export_path, output_path)
        
        print(f"✅ ONNX model exported: {output_path}")
        return output_path


def convert_classification_to_detection_dataset(
    classification_dir: Path,
    output_dir: Path,
    class_names: List[str]
) -> None:
    """Convert classification dataset to YOLO detection format.
    
    Note: This creates pseudo-bounding boxes covering the full image.
    For real detection, you need actual bounding box annotations.
    
    Args:
        classification_dir: Directory with class subfolders
        output_dir: Output directory for YOLO format
        class_names: List of class names
    """
    import shutil
    from PIL import Image
    
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    
    for split in ['train', 'val']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_idx, class_name in enumerate(class_names):
        class_dir = classification_dir / class_name
        if not class_dir.exists():
            continue
        
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        # Split 80/20
        split_idx = int(len(images) * 0.8)
        
        for idx, img_path in enumerate(images):
            split = 'train' if idx < split_idx else 'val'
            
            # Copy image
            dest_img = images_dir / split / f"{class_name}_{idx}.jpg"
            shutil.copy(img_path, dest_img)
            
            # Create label (full image bounding box as placeholder)
            # Format: class_id x_center y_center width height (normalized)
            label_path = labels_dir / split / f"{class_name}_{idx}.txt"
            with open(label_path, 'w') as f:
                # Full image box: center at 0.5, 0.5, size 1.0, 1.0
                f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
    
    # Create data.yaml
    yaml_content = f"""
path: {output_dir.absolute()}
train: images/train
val: images/val

names:
"""
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Dataset converted to YOLO format: {output_dir}")
    print(f"   data.yaml: {yaml_path}")
