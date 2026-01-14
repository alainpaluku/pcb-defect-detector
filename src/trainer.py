"""Training manager for PCB Defect Detection."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from src.config import Config
from src.data_ingestion import DataIngestion
from src.model import PCBClassifier
from src.utils import print_section_header, print_subsection, format_bytes


class TrainingManager:
    """Manages the complete training pipeline for PCB defect classification."""
    
    def __init__(self):
        self.output_path = Config.get_output_path()
        self.data = None
        self.model = None
        self.history = None
        self.metrics = {}
        
        # Set random seeds for reproducibility
        np.random.seed(Config.RANDOM_SEED)
        tf.random.set_seed(Config.RANDOM_SEED)
        
        self._print_header()
    
    def _print_header(self):
        """Print system information header."""
        device_info = Config.get_device_info()
        
        print_section_header("🔬 PCB DEFECT DETECTION SYSTEM")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  Environment: {device_info['environment']}")
        print(f"📦 TensorFlow: {device_info['tensorflow_version']}")
        print(f"🎮 GPU: {'✅ ' + str(device_info['gpu_devices']) if device_info['gpu_available'] else '❌ Not available'}")
        print(f"📁 Output: {self.output_path}")
        print("=" * 60 + "\n")
    
    def setup_data(self):
        """Set up data pipeline."""
        print("📊 Setting up data pipeline...")
        
        self.data = DataIngestion()
        self.data.analyze_dataset()
        self.data.compute_class_weights()
        self.data.create_generators()
        
        return self.data
    
    def setup_model(self):
        """Build and compile model."""
        print("🏗️  Building model...")
        
        self.model = PCBClassifier(num_classes=self.data.num_classes)
        self.model.build_model(
            dropout_rate=0.5,
            l2_reg=0.01
        )
        self.model.compile_model(learning_rate=Config.LEARNING_RATE)
        self.model.get_model_summary()
        
        return self.model
    
    def get_callbacks(self, phase='training'):
        """Create training callbacks.
        
        Args:
            phase: 'training' or 'fine_tuning'
        """
        checkpoint_path = self.output_path / f"best_model_{phase}.keras"
        
        callbacks = [
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            # Learning rate reduction
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=Config.REDUCE_LR_FACTOR,
                patience=Config.REDUCE_LR_PATIENCE,
                min_lr=Config.MIN_LEARNING_RATE,
                verbose=1
            )
        ]
        
        # TensorBoard for Kaggle environment
        if Config.is_kaggle():
            log_dir = self.output_path / "logs" / phase
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=str(log_dir),
                    histogram_freq=1,
                    write_graph=True
                )
            )
        
        return callbacks
    
    def train(self, epochs=None):
        """Train the model (transfer learning phase).
        
        Args:
            epochs: Number of epochs (default: Config.EPOCHS)
        """
        epochs = epochs or Config.EPOCHS
        
        print_section_header("🚀 PHASE 1: TRANSFER LEARNING")
        
        train_steps, val_steps = self.data.get_steps()
        print(f"📈 Steps per epoch - Train: {train_steps}, Val: {val_steps}")
        print(f"🔄 Epochs: {epochs}\n")
        
        self.history = self.model.model.fit(
            self.data.train_generator,
            steps_per_epoch=train_steps,
            validation_data=self.data.val_generator,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=self.get_callbacks('training'),
            class_weight=self.data.class_weights,
            verbose=1
        )
        
        return self.history
    
    def fine_tune(self, epochs=None, layers=None):
        """Fine-tune the model with unfrozen base layers.
        
        Args:
            epochs: Number of fine-tuning epochs
            layers: Number of layers to unfreeze
        """
        epochs = epochs or Config.FINE_TUNE_EPOCHS
        layers = layers or Config.FINE_TUNE_LAYERS
        
        print_section_header(f"🔧 PHASE 2: FINE-TUNING (unfreezing last {layers} layers)")
        
        # Enable fine-tuning
        self.model.enable_fine_tuning(
            num_layers=layers,
            learning_rate=Config.FINE_TUNE_LR
        )
        
        train_steps, val_steps = self.data.get_steps()
        
        # Continue training
        history_fine = self.model.model.fit(
            self.data.train_generator,
            steps_per_epoch=train_steps,
            validation_data=self.data.val_generator,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=self.get_callbacks('fine_tuning'),
            class_weight=self.data.class_weights,
            verbose=1
        )
        
        # Merge histories
        for key in self.history.history:
            if key in history_fine.history:
                self.history.history[key].extend(history_fine.history[key])
        
        return history_fine
    
    def evaluate(self):
        """Evaluate model and return metrics."""
        print_section_header("📊 EVALUATION")
        
        # Evaluate on validation set
        results = self.model.model.evaluate(
            self.data.val_generator,
            verbose=1
        )
        
        self.metrics = dict(zip(self.model.model.metrics_names, results))
        
        # Calculate F1 score
        precision = self.metrics.get('precision', 0)
        recall = self.metrics.get('recall', 0)
        self.metrics['f1_score'] = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        # Print results
        print_subsection("📈 FINAL METRICS")
        for k, v in self.metrics.items():
            print(f"   {k:15s}: {v:.4f}")
        print("-" * 40)
        
        return self.metrics
    
    def plot_training_history(self):
        """Plot training history curves."""
        print("\n📊 Generating training history plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics_to_plot = ['accuracy', 'loss', 'precision', 'recall']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
        
        for ax, metric, color in zip(axes.flat, metrics_to_plot, colors):
            if metric in self.history.history:
                epochs = range(1, len(self.history.history[metric]) + 1)
                
                ax.plot(epochs, self.history.history[metric], 
                       label='Train', linewidth=2, color=color)
                ax.plot(epochs, self.history.history[f'val_{metric}'], 
                       label='Validation', linewidth=2, color=color, linestyle='--')
                
                ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.capitalize())
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
                # Add best value annotation
                best_val = max(self.history.history[f'val_{metric}']) if metric != 'loss' \
                          else min(self.history.history[f'val_{metric}'])
                ax.axhline(y=best_val, color='gray', linestyle=':', alpha=0.5)
        
        plt.suptitle('Training History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_path / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"   Saved to: {save_path}")
    
    def plot_confusion_matrix(self):
        """Generate and plot confusion matrix."""
        print("📊 Generating confusion matrix...")
        
        # Get predictions
        self.data.val_generator.reset()
        predictions = self.model.model.predict(
            self.data.val_generator,
            verbose=1
        )
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.data.val_generator.classes
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.data.class_names,
            yticklabels=self.data.class_names,
            ax=axes[0],
            annot_kws={'size': 11}
        )
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Predicted', fontsize=11)
        axes[0].set_ylabel('True', fontsize=11)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Normalized
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=self.data.class_names,
            yticklabels=self.data.class_names,
            ax=axes[1],
            annot_kws={'size': 11}
        )
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted', fontsize=11)
        axes[1].set_ylabel('True', fontsize=11)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        save_path = self.output_path / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"   Saved to: {save_path}")
        
        return cm, y_true, y_pred
    
    def plot_roc_curves(self, y_true, predictions):
        """Plot ROC curves for each class."""
        print("📊 Generating ROC curves...")
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.data.num_classes))
        
        # Compute ROC curve for each class
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, self.data.num_classes))
        
        for i, (class_name, color) in enumerate(zip(self.data.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curves per Class', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_path / 'roc_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"   Saved to: {save_path}")
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate and save classification report."""
        print("📊 Generating classification report...")
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.data.class_names,
            digits=4
        )
        
        print_section_header("📋 CLASSIFICATION REPORT")
        print(report)
        
        # Save to file
        save_path = self.output_path / 'classification_report.txt'
        with open(save_path, 'w') as f:
            f.write("PCB Defect Classification Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        
        print(f"   Saved to: {save_path}")
        
        return report
    
    def save_model(self):
        """Save model in multiple formats."""
        print("\n💾 Saving model...")
        
        # Keras format (recommended)
        keras_path = self.output_path / 'pcb_model.keras'
        self.model.model.save(keras_path)
        print(f"   ✅ Keras format: {keras_path}")
        
        # H5 format (legacy)
        h5_path = self.output_path / 'pcb_model.h5'
        self.model.model.save(h5_path)
        print(f"   ✅ H5 format: {h5_path}")
        
        # SavedModel format (for TF Serving)
        savedmodel_path = self.output_path / 'saved_model'
        try:
            self.model.model.export(savedmodel_path)
            print(f"   ✅ SavedModel: {savedmodel_path}")
        except Exception as e:
            print(f"   ⚠️ SavedModel export skipped: {e}")
        
        # ONNX and TFLite formats
        self._export_onnx()
        self._export_tflite()
        
        # Model size
        model_size_bytes = keras_path.stat().st_size
        print(f"\n   📦 Model size: {format_bytes(model_size_bytes)}")
    
    def _export_onnx(self):
        """Export model to ONNX format."""
        try:
            import tf2onnx
            import onnx
            
            onnx_path = self.output_path / 'pcb_model.onnx'
            
            # Convert to ONNX
            input_signature = [tf.TensorSpec(
                shape=(None, *Config.IMG_SIZE, 3), 
                dtype=tf.float32, 
                name='input_image'
            )]
            
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model.model,
                input_signature=input_signature,
                opset=13,
                output_path=str(onnx_path)
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            onnx_size = format_bytes(onnx_path.stat().st_size)
            print(f"   ✅ ONNX format: {onnx_path} ({onnx_size})")
            
        except ImportError:
            print("   ⚠️ ONNX export skipped: tf2onnx not installed")
            print("      Install with: pip install tf2onnx onnx")
        except Exception as e:
            print(f"   ⚠️ ONNX export failed: {e}")
    
    def _export_tflite(self):
        """Export model to TFLite format."""
        try:
            tflite_path = self.output_path / 'pcb_model.tflite'
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            tflite_size = format_bytes(tflite_path.stat().st_size)
            print(f"   ✅ TFLite format: {tflite_path} ({tflite_size})")
            
        except Exception as e:
            print(f"   ⚠️ TFLite export failed: {e}")
    
    def run_pipeline(self, fine_tune=True, visualize_augmentation=False):
        """Execute complete training pipeline.
        
        Args:
            fine_tune: Whether to perform fine-tuning phase
            visualize_augmentation: Whether to visualize augmentation effects
        """
        # Setup
        self.setup_data()
        
        if visualize_augmentation:
            self.data.visualize_augmentation()
        
        self.setup_model()
        
        # Training
        self.train()
        
        if fine_tune:
            self.fine_tune()
        
        # Evaluation
        self.metrics = self.evaluate()
        
        # Visualizations
        self.plot_training_history()
        cm, y_true, y_pred = self.plot_confusion_matrix()
        
        # Get predictions for ROC curves
        self.data.val_generator.reset()
        predictions = self.model.model.predict(self.data.val_generator, verbose=0)
        self.plot_roc_curves(y_true, predictions)
        
        # Reports
        self.generate_classification_report(y_true, y_pred)
        
        # Save
        self.save_model()
        
        # Final summary
        print_section_header("🎉 TRAINING COMPLETE!")
        print(f"   📊 Final Accuracy: {self.metrics['accuracy']:.2%}")
        print(f"   📊 Final F1 Score: {self.metrics['f1_score']:.2%}")
        print(f"   📊 Final AUC: {self.metrics.get('auc', 0):.4f}")
        print(f"   📁 Output directory: {self.output_path}")
        print("=" * 60 + "\n")
        
        return self.metrics
