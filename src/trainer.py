"""
Training Manager Module for PCB Defect Detection.

Orchestrates the complete training pipeline including callbacks,
model checkpointing, and performance visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.config import Config
from src.data_ingestion import DataIngestion
from src.model import PCBClassifier


class TrainingManager:
    """
    Manages the complete training pipeline for PCB defect detection.
    
    Responsibilities:
    - Data loading and preprocessing
    - Model training with callbacks
    - Performance evaluation
    - Visualization and reporting
    - Model saving for deployment
    """
    
    def __init__(self):
        """Initialize training manager with configuration."""
        self.config = Config()
        self.output_path = Config.get_output_path()
        self.data_ingestion = None
        self.model_wrapper = None
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(Config.RANDOM_SEED)
        tf.random.set_seed(Config.RANDOM_SEED)
        
        print("="*60)
        print("PCB DEFECT DETECTION SYSTEM")
        print("Automated Optical Inspection (AOI) for Electronics Manufacturing")
        print("="*60)
        print(f"Environment: {'Kaggle' if Config.is_kaggle_environment() else 'Local'}")
        print(f"Output Path: {self.output_path}")
        print("="*60 + "\n")
    
    def setup_data(self):
        """
        Set up data ingestion pipeline.
        
        Returns:
            DataIngestion: Configured data ingestion object
        """
        print("PHASE 1: DATA INGESTION")
        print("-" * 60)
        
        self.data_ingestion = DataIngestion()
        
        # Analyze dataset
        stats = self.data_ingestion.analyze_dataset()
        
        # Compute class weights for imbalanced data
        self.data_ingestion.compute_class_weights()
        
        # Create data generators
        self.data_ingestion.create_data_generators()
        
        return self.data_ingestion
    
    def setup_model(self):
        """
        Set up and compile the model.
        
        Returns:
            PCBClassifier: Configured model wrapper
        """
        print("\nPHASE 2: MODEL ARCHITECTURE")
        print("-" * 60)
        
        self.model_wrapper = PCBClassifier(
            num_classes=self.data_ingestion.num_classes,
            img_size=Config.IMG_SIZE
        )
        
        # Build model
        self.model_wrapper.build_model(trainable_base_layers=0)
        
        # Compile model
        self.model_wrapper.compile_model(learning_rate=Config.LEARNING_RATE)
        
        # Print summary
        self.model_wrapper.get_model_summary()
        
        return self.model_wrapper
    
    def get_callbacks(self):
        """
        Create training callbacks for optimization and monitoring.
        
        Returns:
            list: List of Keras callbacks
        """
        callbacks = []
        
        # Model checkpoint - save best model
        checkpoint_path = self.output_path / "best_model.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping - prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=Config.REDUCE_LR_FACTOR,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=Config.MIN_LEARNING_RATE,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging (if not in Kaggle)
        if not Config.is_kaggle_environment():
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=str(self.output_path / "logs"),
                histogram_freq=1
            )
            callbacks.append(tensorboard)
        
        return callbacks
    
    def train_model(self):
        """
        Train the model with configured callbacks.
        
        Returns:
            tf.keras.callbacks.History: Training history
        """
        print("\nPHASE 3: MODEL TRAINING")
        print("-" * 60)
        
        train_steps, val_steps = self.data_ingestion.get_steps_per_epoch()
        
        self.history = self.model_wrapper.model.fit(
            self.data_ingestion.train_generator,
            steps_per_epoch=train_steps,
            validation_data=self.data_ingestion.val_generator,
            validation_steps=val_steps,
            epochs=Config.EPOCHS,
            callbacks=self.get_callbacks(),
            class_weight=self.data_ingestion.class_weights,
            verbose=1
        )
        
        print("\n✓ Training completed successfully")
        return self.history
    
    def evaluate_model(self):
        """
        Evaluate model performance on validation set.
        
        Returns:
            dict: Evaluation metrics
        """
        print("\nPHASE 4: MODEL EVALUATION")
        print("-" * 60)
        
        # Evaluate on validation set
        val_loss, val_acc, val_precision, val_recall, val_auc = \
            self.model_wrapper.model.evaluate(
                self.data_ingestion.val_generator,
                verbose=1
            )
        
        # Calculate F1 score
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)
        
        print("\n" + "="*60)
        print("FINAL PERFORMANCE METRICS")
        print("="*60)
        print(f"Validation Loss:      {val_loss:.4f}")
        print(f"Validation Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"Precision:            {val_precision:.4f}")
        print(f"Recall:               {val_recall:.4f}")
        print(f"F1 Score:             {f1_score:.4f}")
        print(f"AUC:                  {val_auc:.4f}")
        print("="*60 + "\n")
        
        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1_score': f1_score,
            'auc': val_auc
        }
    
    def generate_predictions(self):
        """
        Generate predictions for confusion matrix and classification report.
        
        Returns:
            tuple: (y_true, y_pred)
        """
        # Reset generator
        self.data_ingestion.val_generator.reset()
        
        # Get predictions
        predictions = self.model_wrapper.model.predict(
            self.data_ingestion.val_generator,
            verbose=1
        )
        y_pred = np.argmax(predictions, axis=1)
        
        # Get true labels
        y_true = self.data_ingestion.val_generator.classes
        
        return y_true, y_pred
    
    def plot_training_history(self):
        """Plot training and validation metrics over epochs."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'training_history.png', dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to: {self.output_path / 'training_history.png'}")
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix for model predictions.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.data_ingestion.class_names,
            yticklabels=self.data_ingestion.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - PCB Defect Classification', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {self.output_path / 'confusion_matrix.png'}")
        plt.show()
    
    def print_classification_report(self, y_true, y_pred):
        """
        Print detailed classification report.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
        """
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.data_ingestion.class_names,
            digits=4
        )
        print(report)
        
        # Save report to file
        report_path = self.output_path / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Classification report saved to: {report_path}")
    
    def save_model(self):
        """Save trained model in multiple formats for deployment."""
        print("\nPHASE 5: MODEL EXPORT")
        print("-" * 60)
        
        # Save in Keras format
        keras_path = self.output_path / 'pcb_defect_model.h5'
        self.model_wrapper.model.save(keras_path)
        print(f"✓ Keras model saved to: {keras_path}")
        
        # Save in SavedModel format (for TensorFlow Serving)
        savedmodel_path = self.output_path / 'saved_model'
        self.model_wrapper.model.save(savedmodel_path)
        print(f"✓ SavedModel format saved to: {savedmodel_path}")
        
        # Save model architecture as JSON
        model_json = self.model_wrapper.model.to_json()
        json_path = self.output_path / 'model_architecture.json'
        with open(json_path, 'w') as f:
            f.write(model_json)
        print(f"✓ Model architecture saved to: {json_path}")
        
        print("\n✓ All model artifacts saved successfully")
    
    def run_pipeline(self):
        """
        Execute the complete training pipeline.
        
        This is the main entry point for training the PCB defect detector.
        """
        try:
            # Phase 1: Data Setup
            self.setup_data()
            
            # Phase 2: Model Setup
            self.setup_model()
            
            # Phase 3: Training
            self.train_model()
            
            # Phase 4: Evaluation
            metrics = self.evaluate_model()
            
            # Generate predictions
            y_true, y_pred = self.generate_predictions()
            
            # Visualizations
            self.plot_training_history()
            self.plot_confusion_matrix(y_true, y_pred)
            self.print_classification_report(y_true, y_pred)
            
            # Phase 5: Save Model
            self.save_model()
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            print("The model is ready for deployment in industrial AOI systems.")
            print(f"All artifacts saved to: {self.output_path}")
            print("="*60 + "\n")
            
            return metrics
            
        except Exception as e:
            print(f"\n❌ Error in training pipeline: {e}")
            raise
