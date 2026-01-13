#!/usr/bin/env python3
"""
PCB Defect Detector - Main Pipeline Entry Point.

This module serves as the main entry point for the PCB defect detection
training pipeline. It orchestrates all components including data management,
preprocessing, model building, training, and evaluation.

Usage:
    python main.py

Environment Variables:
    KAGGLE_API_TOKEN: JSON string containing Kaggle credentials
                      Format: {"username":"your_username","key":"your_api_key"}

Example:
    $ export KAGGLE_API_TOKEN='{"username":"user","key":"abc123"}'
    $ python main.py
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional

import tensorflow as tf

from config import PipelineConfig, DataConfig, ModelConfig, TrainingConfig
from data_manager import KaggleDataManager
from data_pipeline import DataPipeline
from model_builder import PCBModelBuilder
from trainer import Trainer
from evaluator import Evaluator


class PCBDefectDetectorPipeline:
    """Main pipeline orchestrator for PCB defect detection.
    
    This class coordinates all pipeline components to execute the complete
    training workflow from data download to model evaluation.
    
    Attributes:
        config: Master pipeline configuration.
        logger: Logger instance for this class.
        data_manager: KaggleDataManager instance.
        data_pipeline: DataPipeline instance.
        model_builder: PCBModelBuilder instance.
        trainer: Trainer instance.
        evaluator: Evaluator instance.
    
    Example:
        >>> config = PipelineConfig()
        >>> pipeline = PCBDefectDetectorPipeline(config)
        >>> pipeline.run()
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """Initialize the pipeline with configuration.
        
        Args:
            config: Optional PipelineConfig. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Component instances (initialized during run)
        self.data_manager: Optional[KaggleDataManager] = None
        self.data_pipeline: Optional[DataPipeline] = None
        self.model_builder: Optional[PCBModelBuilder] = None
        self.trainer: Optional[Trainer] = None
        self.evaluator: Optional[Evaluator] = None
    
    def setup_logging(self, level: int = logging.INFO) -> None:
        """Configure logging for the pipeline.
        
        Args:
            level: Logging level (default: INFO).
        """
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('pipeline.log')
            ]
        )
        
        # Reduce TensorFlow logging verbosity
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel(logging.WARNING)
    
    def setup_gpu(self) -> None:
        """Configure GPU settings for training."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            self.logger.info(f"Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.info(f"  Enabled memory growth for {gpu}")
                except RuntimeError as e:
                    self.logger.warning(f"  Could not set memory growth: {e}")
        else:
            self.logger.warning("No GPU found. Training will use CPU.")

    def run(self, api_token: Optional[str] = None) -> None:
        """Execute the complete training pipeline.
        
        This method runs all pipeline stages in sequence:
        1. Setup logging and GPU
        2. Authenticate with Kaggle
        3. Download and parse dataset
        4. Prepare data splits and generators
        5. Build model
        6. Train model
        7. Evaluate and generate reports
        
        Args:
            api_token: Optional Kaggle API token. If not provided,
                      will check environment variable or prompt user.
        """
        self.setup_logging()
        self.logger.info("=" * 60)
        self.logger.info("PCB Defect Detector Pipeline - Starting")
        self.logger.info("=" * 60)
        
        try:
            # Stage 1: Setup
            self.logger.info("\n[Stage 1/7] Setting up environment...")
            self.setup_gpu()
            
            # Stage 2: Data Management
            self.logger.info("\n[Stage 2/7] Authenticating with Kaggle...")
            self.data_manager = KaggleDataManager(self.config.data)
            self.data_manager.authenticate(api_token)
            
            # Stage 3: Download Dataset
            self.logger.info("\n[Stage 3/7] Downloading dataset...")
            self.data_manager.download_dataset()
            class_images = self.data_manager.parse_directory_structure()
            class_names = self.data_manager.get_class_names()
            
            # Stage 4: Data Pipeline
            self.logger.info("\n[Stage 4/7] Preparing data pipeline...")
            self.data_pipeline = DataPipeline(self.config.data, class_names)
            self.data_pipeline.prepare_data(class_images)
            
            # Stage 5: Model Building
            self.logger.info("\n[Stage 5/7] Building model...")
            self.model_builder = PCBModelBuilder(
                self.config.model,
                self.config.data,
                num_classes=len(class_names)
            )
            model = self.model_builder.build()
            
            # Stage 6: Training
            self.logger.info("\n[Stage 6/7] Training model...")
            self.trainer = Trainer(self.config.training, model)
            self.trainer.compile()
            self.trainer.setup_callbacks()
            
            history = self.trainer.train(
                train_dataset=self.data_pipeline.get_train_dataset(),
                val_dataset=self.data_pipeline.get_val_dataset(),
                class_weights=self.data_pipeline.get_class_weights()
            )
            
            # Save final model
            self.trainer.save_model()
            
            # Stage 7: Evaluation
            self.logger.info("\n[Stage 7/7] Evaluating model...")
            self.evaluator = Evaluator(
                model=model,
                class_names=class_names,
                output_dir=Path('./results')
            )
            
            results = self.evaluator.generate_full_report(
                test_dataset=self.data_pipeline.get_test_dataset(),
                true_labels=self.data_pipeline.get_test_labels(),
                history=self.trainer.get_history()
            )
            
            # Final summary
            self._print_final_summary(results)
            
        except KeyboardInterrupt:
            self.logger.warning("\nPipeline interrupted by user.")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"\nPipeline failed: {e}", exc_info=True)
            raise
    
    def _print_final_summary(self, results: dict) -> None:
        """Print final pipeline summary.
        
        Args:
            results: Dictionary containing evaluation results.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 60)
        self.logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
        self.logger.info(f"Test Loss: {results['test_loss']:.4f}")
        self.logger.info(f"Model saved to: {self.config.training.checkpoint_dir}")
        self.logger.info(f"Results saved to: ./results")
        self.logger.info("=" * 60)


def main() -> None:
    """Main entry point for the PCB defect detection pipeline."""
    # Create configuration
    config = PipelineConfig(
        data=DataConfig(
            dataset_name="akhatova/pcb-defects",
            data_dir=Path("./data"),
            image_size=(224, 224),
            batch_size=32,
            validation_split=0.15,
            test_split=0.15,
            random_seed=42
        ),
        model=ModelConfig(
            base_model="MobileNetV2",
            dropout_rate=0.5,
            dense_units=256,
            freeze_base=True
        ),
        training=TrainingConfig(
            epochs=50,
            learning_rate=1e-4,
            early_stopping_patience=10,
            reduce_lr_patience=5,
            reduce_lr_factor=0.2,
            min_learning_rate=1e-7,
            checkpoint_dir=Path("./checkpoints")
        )
    )
    
    # Run pipeline
    pipeline = PCBDefectDetectorPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
