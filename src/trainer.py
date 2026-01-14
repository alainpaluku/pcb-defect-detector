"""Training pipeline for PCB Defect Detection with YOLOv8."""

import shutil
from pathlib import Path
from datetime import datetime
from src.config import Config
from src.data_ingestion import DataIngestion
from src.model import PCBDetector


class TrainingManager:
    """Manages the complete training pipeline."""
    
    def __init__(self, data_path=None):
        """Initialize training manager.
        
        Args:
            data_path: Path to dataset (auto-detected if None)
        """
        self.data_path = data_path
        self.output_path = Config.get_output_path()
        self.data = None
        self.model = None
        self.metrics = {}
        
        self._print_header()
    
    def _print_header(self):
        """Print system information."""
        print("=" * 60)
        print("PCB DEFECT DETECTION - YOLOv8")
        print("=" * 60)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Environment: {'Kaggle' if Config.is_kaggle() else 'Local'}")
        print(f"Output: {self.output_path}")
        print("=" * 60)
    
    def setup_data(self):
        """Set up data pipeline."""
        print("\n[1/4] Setting up data...")
        
        self.data = DataIngestion(data_path=self.data_path)
        
        if not self.data.find_data_structure():
            raise FileNotFoundError(f"Dataset not found at {self.data.data_path}")
        
        self.data.collect_images()
        
        if len(self.data.all_images) == 0:
            raise ValueError("No images found in dataset")
        
        stats = self.data.get_stats()
        print(f"Images: {stats['total_images']}")
        print(f"  With XML annotations: {stats['with_xml']}")
        print(f"  From class folders: {stats['from_folders']}")
        
        # Convert to YOLO format
        self.data.create_yolo_dataset()
        
        return self.data
    
    def setup_model(self):
        """Initialize model."""
        print("\n[2/4] Setting up model...")
        
        self.model = PCBDetector()
        return self.model
    
    def train(self, epochs=None):
        """Train the model.
        
        Args:
            epochs: Number of epochs (uses Config.EPOCHS if None)
        """
        print("\n[3/4] Training...")
        
        yaml_path = self.data.get_yaml_path()
        
        results = self.model.train(
            data_yaml=yaml_path,
            epochs=epochs,
            project=str(self.output_path),
            name="pcb_yolo"
        )
        
        return results
    
    def evaluate(self):
        """Evaluate the model."""
        print("\n[4/4] Evaluating...")
        
        yaml_path = self.data.get_yaml_path()
        results = self.model.validate(data_yaml=str(yaml_path))
        
        self.metrics = self.model.get_metrics(results)
        
        print("\nResults:")
        print(f"  mAP@50:     {self.metrics['mAP50']:.4f}")
        print(f"  mAP@50-95:  {self.metrics['mAP50-95']:.4f}")
        print(f"  Precision:  {self.metrics['precision']:.4f}")
        print(f"  Recall:     {self.metrics['recall']:.4f}")
        
        return self.metrics
    
    def save_model(self):
        """Save the trained model."""
        print("\nSaving model...")
        
        # Copy best weights
        best_model = self.output_path / "pcb_yolo" / "weights" / "best.pt"
        if best_model.exists():
            dst = self.output_path / "pcb_model.pt"
            shutil.copy(best_model, dst)
            print(f"Model saved: {dst}")
        
        # Export to ONNX
        try:
            self.model.export(format="onnx")
        except Exception as e:
            print(f"ONNX export failed: {e}")
    
    def run_pipeline(self, epochs=None):
        """Run complete training pipeline.
        
        Args:
            epochs: Number of training epochs
        
        Returns:
            Dictionary of metrics
        """
        self.setup_data()
        self.setup_model()
        self.train(epochs=epochs)
        self.evaluate()
        self.save_model()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"mAP@50: {self.metrics.get('mAP50', 0):.4f}")
        print(f"Output: {self.output_path}")
        
        return self.metrics
