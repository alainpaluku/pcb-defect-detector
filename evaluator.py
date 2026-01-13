"""
Evaluator module for PCB Defect Detector.

This module handles model evaluation, generating classification reports,
confusion matrices, and training visualization plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


class Evaluator:
    """Evaluates trained models and generates performance visualizations."""
    
    def __init__(
        self,
        model: Model,
        class_names: List[str],
        output_dir: Optional[Path] = None
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = Path(output_dir) if output_dir else Path('./results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions: Optional[np.ndarray] = None
        self.pred_labels: Optional[np.ndarray] = None
        self.true_labels: Optional[np.ndarray] = None
        self.pred_probs: Optional[np.ndarray] = None
    
    def evaluate(
        self,
        test_dataset: tf.data.Dataset,
        true_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate model on test dataset."""
        self.logger.info("Evaluating model...")
        
        self.predictions = self.model.predict(test_dataset, verbose=1)
        self.pred_labels = np.argmax(self.predictions, axis=1)
        self.pred_probs = np.max(self.predictions, axis=1)
        self.true_labels = true_labels
        
        test_loss, test_acc = self.model.evaluate(test_dataset, verbose=0)
        
        # Calculate metrics
        f1_macro = f1_score(true_labels, self.pred_labels, average='macro')
        f1_weighted = f1_score(true_labels, self.pred_labels, average='weighted')
        precision = precision_score(true_labels, self.pred_labels, average='weighted')
        recall = recall_score(true_labels, self.pred_labels, average='weighted')
        
        self.logger.info(f"Test Loss: {test_loss:.4f}")
        self.logger.info(f"Test Accuracy: {test_acc:.4f}")
        self.logger.info(f"F1 Macro: {f1_macro:.4f}")
        self.logger.info(f"F1 Weighted: {f1_weighted:.4f}")
        
        report = classification_report(
            true_labels, self.pred_labels,
            target_names=self.class_names, digits=4
        )
        self.logger.info(f"\n{report}")
        
        # Save report
        with open(self.output_dir / "classification_report.txt", "w") as f:
            f.write(report)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'report': report
        }
    
    def plot_confusion_matrix(self, normalize: bool = True, save: bool = True) -> plt.Figure:
        """Plot confusion matrix."""
        if self.pred_labels is None:
            raise ValueError("Call evaluate() first.")
        
        cm = confusion_matrix(self.true_labels, self.pred_labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt, title = '.1%', 'Normalized Confusion Matrix'
        else:
            fmt, title = 'd', 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax, square=True)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save:
            suffix = '_normalized' if normalize else ''
            fig.savefig(self.output_dir / f'confusion_matrix{suffix}.png', dpi=150)
        
        return fig
    
    def plot_training_curves(self, history: Dict[str, List[float]], save: bool = True) -> plt.Figure:
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history['loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, history['loss'], 'b-', label='Train')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, history['accuracy'], 'b-', label='Train')
        axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'training_curves.png', dpi=150)
        
        return fig
    
    def plot_misclassified(
        self,
        test_paths: List[str],
        num_samples: int = 12,
        save: bool = True
    ) -> Optional[plt.Figure]:
        """Plot misclassified samples."""
        if self.pred_labels is None:
            raise ValueError("Call evaluate() first.")
        
        # Find misclassified indices
        misclassified = np.where(self.pred_labels != self.true_labels)[0]
        
        if len(misclassified) == 0:
            self.logger.info("No misclassified samples!")
            return None
        
        # Sort by confidence (most confident mistakes first)
        confidences = self.pred_probs[misclassified]
        sorted_idx = np.argsort(confidences)[::-1]
        misclassified = misclassified[sorted_idx]
        
        num_samples = min(num_samples, len(misclassified))
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for i, idx in enumerate(misclassified[:num_samples]):
            img = plt.imread(test_paths[idx])
            axes[i].imshow(img)
            axes[i].axis('off')
            
            true_name = self.class_names[self.true_labels[idx]]
            pred_name = self.class_names[self.pred_labels[idx]]
            conf = self.pred_probs[idx]
            
            axes[i].set_title(f'T:{true_name}\nP:{pred_name} ({conf:.0%})', fontsize=8)
        
        # Hide empty subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Misclassified Samples ({len(misclassified)} total)', fontsize=12)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'misclassified.png', dpi=150)
        
        return fig
    
    def generate_full_report(
        self,
        test_dataset: tf.data.Dataset,
        true_labels: np.ndarray,
        history: Optional[Dict[str, List[float]]] = None,
        test_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate complete evaluation report."""
        self.logger.info("Generating full report...")
        
        results = self.evaluate(test_dataset, true_labels)
        
        self.plot_confusion_matrix(normalize=True)
        self.plot_confusion_matrix(normalize=False)
        
        if history:
            self.plot_training_curves(history)
        
        if test_paths:
            self.plot_misclassified(test_paths)
        
        plt.close('all')
        
        self.logger.info(f"Results saved to {self.output_dir}")
        return results
