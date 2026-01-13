#!/usr/bin/env python3
"""
PCB Defect Detector - Main Pipeline Entry Point.

Usage:
    python main.py [--epochs N] [--batch-size N] [--no-finetune]

Environment Variables:
    KAGGLE_API_TOKEN: JSON string with Kaggle credentials (local only)
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

import tensorflow as tf

from config import (
    PipelineConfig, DataConfig, ModelConfig, TrainingConfig,
    Environment, detect_environment
)
from data_manager import KaggleDataManager
from data_pipeline import DataPipeline
from model_builder import PCBModelBuilder
from trainer import Trainer
from evaluator import Evaluator


class PCBDefectDetectorPipeline:
    """Main pipeline orchestrator for PCB defect detection."""
    
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.data_manager: Optional[KaggleDataManager] = None
        self.data_pipeline: Optional[DataPipeline] = None
        self.model_builder: Optional[PCBModelBuilder] = None
        self.trainer: Optional[Trainer] = None
        self.evaluator: Optional[Evaluator] = None
    
    def setup_logging(self) -> None:
        """Configure logging."""
        log_file = self.config.results_dir / 'pipeline.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(str(log_file))
            ]
        )
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel(logging.WARNING)
    
    def setup_gpu(self) -> None:
        """Configure GPU memory growth."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            self.logger.info(f"Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    self.logger.warning(f"GPU config error: {e}")
        else:
            self.logger.warning("No GPU found, using CPU.")

    def run(self, api_token: Optional[str] = None, fine_tune: bool = True) -> dict:
        """Execute the complete training pipeline."""
        self.setup_logging()
        
        self.logger.info("=" * 50)
        self.logger.info("PCB Defect Detector Pipeline")
        self.logger.info(f"Environment: {self.config.environment.value}")
        self.logger.info("=" * 50)
        
        try:
            # Setup
            self.setup_gpu()
            
            # Data
            self.logger.info("\n[1/6] Loading dataset...")
            self.data_manager = KaggleDataManager(self.config.data)
            self.data_manager.authenticate(api_token)
            self.data_manager.download_dataset()
            class_images = self.data_manager.parse_directory_structure()
            class_names = self.data_manager.get_class_names()
            
            # Pipeline
            self.logger.info("\n[2/6] Preparing data pipeline...")
            self.data_pipeline = DataPipeline(self.config.data, class_names)
            self.data_pipeline.prepare_data(class_images)
            
            # Model
            self.logger.info("\n[3/6] Building model...")
            self.model_builder = PCBModelBuilder(
                self.config.model, self.config.data, num_classes=len(class_names)
            )
            model = self.model_builder.build()
            
            # Training
            self.logger.info("\n[4/6] Training (frozen base)...")
            self.trainer = Trainer(self.config.training, model)
            self.trainer.compile()
            
            self.trainer.train(
                train_dataset=self.data_pipeline.get_train_dataset(),
                val_dataset=self.data_pipeline.get_val_dataset(),
                class_weights=self.data_pipeline.get_class_weights()
            )
            
            # Fine-tuning
            if fine_tune and self.config.training.fine_tune_epochs > 0:
                self.logger.info("\n[5/6] Fine-tuning...")
                self.model_builder.unfreeze_layers(self.config.training.fine_tune_layers)
                
                self.trainer.fine_tune(
                    train_dataset=self.data_pipeline.get_train_dataset(),
                    val_dataset=self.data_pipeline.get_val_dataset(),
                    class_weights=self.data_pipeline.get_class_weights()
                )
            else:
                self.logger.info("\n[5/6] Skipping fine-tuning...")
            
            # Save model
            self.trainer.save_model()
            
            # Evaluation
            self.logger.info("\n[6/6] Evaluating...")
            self.evaluator = Evaluator(
                model=model,
                class_names=class_names,
                output_dir=self.config.results_dir
            )
            
            results = self.evaluator.generate_full_report(
                test_dataset=self.data_pipeline.get_test_dataset(),
                true_labels=self.data_pipeline.get_test_labels(),
                history=self.trainer.get_combined_history(),
                test_paths=self.data_pipeline.get_test_paths()
            )
            
            # Summary
            self._print_summary(results)
            return results
            
        except KeyboardInterrupt:
            self.logger.warning("\nInterrupted by user.")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"\nPipeline failed: {e}", exc_info=True)
            raise
    
    def _print_summary(self, results: dict) -> None:
        """Print final summary."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("COMPLETED")
        self.logger.info("=" * 50)
        self.logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
        self.logger.info(f"F1 Macro: {results['f1_macro']:.4f}")
        self.logger.info(f"F1 Weighted: {results['f1_weighted']:.4f}")
        self.logger.info(f"Results: {self.config.results_dir}")
        self.logger.info(f"Model: {self.config.training.checkpoint_dir}")
        self.logger.info("=" * 50)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PCB Defect Detector Training')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model', type=str, default='MobileNetV2',
                       choices=['MobileNetV2', 'ResNet50', 'EfficientNetB0'])
    parser.add_argument('--no-finetune', action='store_true', help='Skip fine-tuning')
    parser.add_argument('--finetune-epochs', type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    config = PipelineConfig(
        data=DataConfig(
            batch_size=args.batch_size,
            image_size=(224, 224),
            validation_split=0.15,
            test_split=0.15
        ),
        model=ModelConfig(
            base_model=args.model,
            dropout_rate=0.5,
            dense_units=256,
            freeze_base=True
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            fine_tune_epochs=0 if args.no_finetune else args.finetune_epochs,
            fine_tune_layers=30,
            fine_tune_lr=1e-5
        )
    )
    
    pipeline = PCBDefectDetectorPipeline(config)
    pipeline.run(fine_tune=not args.no_finetune)


if __name__ == "__main__":
    main()
