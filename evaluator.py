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
    precision_recall_fscore_support
)


class Evaluator:
    """Evaluates trained models and generates performance visualizations.
    
    This class provides comprehensive model evaluation including
    classification metrics, confusion matrix visualization, and
    training history plots.
    
    Attributes:
        model: Trained Keras model to evaluate.
        class_names: List of class names for labeling.
        logger: Logger instance for this class.
        output_dir: Directory for saving plots.
    
    Example:
        >>> evaluator = Evaluator(model, class_names, output_dir='./results')
        >>> evaluator.evaluate(test_dataset, test_labels)
        >>> evaluator.plot_confusion_matrix()
        >>> evaluator.plot_training_curves(history)
    """
    
    def __init__(
        self,
        model: Model,
        class_names: List[str],
        output_dir: Optional[Path] = None
    ) -> None:
        """Initialize Evaluator with model and class information.
        
        Args:
            model: Trained Keras Model to evaluate.
            class_names: List of class names for labeling outputs.
            output_dir: Optional directory for saving plots.
        """
        self.model = model
        self.class_names = class_names
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = Path(output_dir) if output_dir else Path('./results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions: Optional[np.ndarray] = None
        self.true_labels: Optional[np.ndarray] = None
        self.pred_labels: Optional[np.ndarray] = None
    
    def evaluate(
        self,
        test_dataset: tf.data.Dataset,
        true_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate model on test dataset.
        
        Args:
            test_dataset: TensorFlow Dataset for testing.
            true_labels: Array of true labels for the test set.
        
        Returns:
            Dictionary containing evaluation metrics.
        """
        self.logger.info("Evaluating model on test set...")
        
        # Get model predictions
        self.predictions = self.model.predict(test_dataset, verbose=1)
        self.pred_labels = np.argmax(self.predictions, axis=1)
        self.true_labels = true_labels
        
        # Calculate overall metrics
        test_loss, test_accuracy = self.model.evaluate(test_dataset, verbose=0)
        
        self.logger.info(f"Test Loss: {test_loss:.4f}")
        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Generate classification report
        report = self._generate_classification_report()
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report
        }
    
    def _generate_classification_report(self) -> str:
        """Generate and log classification report.
        
        Returns:
            Classification report as string.
        """
        if self.pred_labels is None or self.true_labels is None:
            raise ValueError("No predictions available. Call evaluate() first.")
        
        report = classification_report(
            self.true_labels,
            self.pred_labels,
            target_names=self.class_names,
            digits=4
        )
        
        self.logger.info("\nClassification Report:")
        self.logger.info(f"\n{report}")
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels,
            self.pred_labels,
            average=None
        )
        
        self.logger.info("\nPer-class metrics:")
        for i, class_name in enumerate(self.class_names):
            self.logger.info(
                f"  {class_name}: P={precision[i]:.4f}, "
                f"R={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}"
            )
        
        return report

    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        save: bool = True
    ) -> plt.Figure:
        """Plot confusion matrix using Seaborn heatmap.
        
        Args:
            normalize: If True, normalize values to percentages.
            figsize: Figure size as (width, height).
            save: If True, save plot to output directory.
        
        Returns:
            Matplotlib Figure object.
        
        Raises:
            ValueError: If no predictions available.
        """
        if self.pred_labels is None or self.true_labels is None:
            raise ValueError("No predictions available. Call evaluate() first.")
        
        self.logger.info("Generating confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.true_labels, self.pred_labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            square=True,
            cbar_kws={'shrink': 0.8}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'confusion_matrix.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        figsize: Tuple[int, int] = (14, 5),
        save: bool = True
    ) -> plt.Figure:
        """Plot training and validation loss/accuracy curves.
        
        Args:
            history: Training history dictionary from Keras.
            figsize: Figure size as (width, height).
            save: If True, save plot to output directory.
        
        Returns:
            Matplotlib Figure object.
        """
        self.logger.info("Generating training curves...")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        epochs = range(1, len(history['loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Find best epoch
        best_epoch = np.argmin(history['val_loss']) + 1
        best_val_loss = min(history['val_loss'])
        axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, 
                       label=f'Best Epoch ({best_epoch})')
        axes[0].scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)
        
        # Accuracy plot
        axes[1].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        
        # Mark best epoch on accuracy plot
        best_val_acc = history['val_accuracy'][best_epoch - 1]
        axes[1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
        axes[1].scatter([best_epoch], [best_val_acc], color='g', s=100, zorder=5)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'training_curves.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Training curves saved to {save_path}")
        
        return fig
    
    def plot_class_distribution(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save: bool = True
    ) -> plt.Figure:
        """Plot distribution of predictions vs true labels.
        
        Args:
            figsize: Figure size as (width, height).
            save: If True, save plot to output directory.
        
        Returns:
            Matplotlib Figure object.
        """
        if self.pred_labels is None or self.true_labels is None:
            raise ValueError("No predictions available. Call evaluate() first.")
        
        self.logger.info("Generating class distribution plot...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        # Count occurrences
        true_counts = [np.sum(self.true_labels == i) for i in range(len(self.class_names))]
        pred_counts = [np.sum(self.pred_labels == i) for i in range(len(self.class_names))]
        
        bars1 = ax.bar(x - width/2, true_counts, width, label='True Labels', color='steelblue')
        bars2 = ax.bar(x + width/2, pred_counts, width, label='Predictions', color='coral')
        
        ax.set_xlabel('Class', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('True Labels vs Predictions Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'class_distribution.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Class distribution plot saved to {save_path}")
        
        return fig
    
    def generate_full_report(
        self,
        history: Dict[str, List[float]],
        show_plots: bool = True
    ) -> None:
        """Generate complete evaluation report with all visualizations.
        
        Args:
            history: Training history dictionary.
            show_plots: If True, display plots interactively.
        """
        self.logger.info("Generating full evaluation report...")
        
        # Generate all plots
        self.plot_confusion_matrix(normalize=True)
        self.plot_confusion_matrix(normalize=False)
        self.plot_training_curves(history)
        self.plot_class_distribution()
        
        if show_plots:
            plt.show()
        
        self.logger.info(f"All evaluation artifacts saved to {self.output_dir}")

    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        save: bool = True
    ) -> plt.Figure:
        """Plot confusion matrix using Seaborn heatmap.
        
        Args:
            normalize: If True, normalize values to percentages.
            figsize: Figure size as (width, height).
            save: If True, save plot to output directory.
        
        Returns:
            Matplotlib Figure object.
        
        Raises:
            ValueError: If no predictions available.
        """
        if self.pred_labels is None or self.true_labels is None:
            raise ValueError("No predictions available. Call evaluate() first.")
        
        self.logger.info("Generating confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.true_labels, self.pred_labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            square=True,
            cbar_kws={'shrink': 0.8}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'confusion_matrix.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        figsize: Tuple[int, int] = (14, 5),
        save: bool = True
    ) -> plt.Figure:
        """Plot training and validation loss/accuracy curves.
        
        Args:
            history: Training history dictionary from Keras.
            figsize: Figure size as (width, height).
            save: If True, save plot to output directory.
        
        Returns:
            Matplotlib Figure object.
        """
        self.logger.info("Generating training curves...")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        epochs = range(1, len(history['loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = np.argmin(history['val_loss']) + 1
        best_val_loss = min(history['val_loss'])
        axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, 
                       label=f'Best Epoch ({best_epoch})')
        axes[0].scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)
        
        # Accuracy plot
        axes[1].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        
        # Mark best epoch on accuracy plot
        best_val_acc = history['val_accuracy'][best_epoch - 1]
        axes[1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
        axes[1].scatter([best_epoch], [best_val_acc], color='g', s=100, zorder=5)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'training_curves.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Training curves saved to {save_path}")
        
        return fig
    
    def plot_class_distribution(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save: bool = True
    ) -> plt.Figure:
        """Plot distribution of predictions vs true labels.
        
        Args:
            figsize: Figure size as (width, height).
            save: If True, save plot to output directory.
        
        Returns:
            Matplotlib Figure object.
        """
        if self.pred_labels is None or self.true_labels is None:
            raise ValueError("No predictions available. Call evaluate() first.")
        
        self.logger.info("Generating class distribution plot...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        # Count occurrences
        true_counts = [np.sum(self.true_labels == i) for i in range(len(self.class_names))]
        pred_counts = [np.sum(self.pred_labels == i) for i in range(len(self.class_names))]
        
        bars1 = ax.bar(x - width/2, true_counts, width, label='True Labels', color='steelblue')
        bars2 = ax.bar(x + width/2, pred_counts, width, label='Predictions', color='coral')
        
        ax.set_xlabel('Class', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('True Labels vs Predictions Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'class_distribution.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Class distribution plot saved to {save_path}")
        
        return fig
    
    def generate_full_report(
        self,
        test_dataset: tf.data.Dataset,
        true_labels: np.ndarray,
        history: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Any]:
        """Generate complete evaluation report with all visualizations.
        
        Args:
            test_dataset: TensorFlow Dataset for testing.
            true_labels: Array of true labels.
            history: Optional training history for plotting curves.
        
        Returns:
            Dictionary containing all evaluation results.
        """
        self.logger.info("Generating full evaluation report...")
        
        # Evaluate model
        results = self.evaluate(test_dataset, true_labels)
        
        # Generate plots
        self.plot_confusion_matrix()
        self.plot_class_distribution()
        
        if history:
            self.plot_training_curves(history)
        
        self.logger.info(f"All evaluation artifacts saved to {self.output_dir}")
        
        return results
