"""Visualization module for PCB Defect Detection."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from src.config import Config
from src.utils import print_section_header, print_subsection, get_all_images

# Set matplotlib style
available_styles = plt.style.available
for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot', 'default']:
    if style in available_styles or style == 'default':
        try:
            plt.style.use(style)
            break
        except OSError:
            continue


class Visualizer:
    """Handles all visualization tasks for the project."""

    @staticmethod
    def plot_training_history(history, output_path):
        """Plot training history curves.

        Args:
            history: Training history object or dict
            output_path: Path to save the plot
        """
        print("\n📊 Generating training history plots...")

        # Handle both Keras History object and plain dict
        history_dict = history.history if hasattr(history, 'history') else history

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics_to_plot = ['accuracy', 'loss', 'precision', 'recall']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

        for ax, metric, color in zip(axes.flat, metrics_to_plot, colors):
            if metric in history_dict:
                epochs = range(1, len(history_dict[metric]) + 1)

                ax.plot(epochs, history_dict[metric],
                       label='Train', linewidth=2, color=color)

                val_key = f'val_{metric}'
                if val_key in history_dict:
                    ax.plot(epochs, history_dict[val_key],
                           label='Validation', linewidth=2, color=color, linestyle='--')

                    # Add best value annotation
                    best_val = max(history_dict[val_key]) if metric != 'loss' \
                              else min(history_dict[val_key])
                    ax.axhline(y=best_val, color='gray', linestyle=':', alpha=0.5)

                ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.capitalize())
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)

        plt.suptitle('Training History', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = output_path / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   Saved to: {save_path}")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
        """Generate and plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            output_path: Path to save the plot

        Returns:
            Confusion matrix array
        """
        print("📊 Generating confusion matrix...")

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Avoid division by zero
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = cm.astype('float') / (cm_sum + 1e-7)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
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
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1],
            annot_kws={'size': 11}
        )
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted', fontsize=11)
        axes[1].set_ylabel('True', fontsize=11)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        save_path = output_path / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   Saved to: {save_path}")

        return cm

    @staticmethod
    def plot_roc_curves(y_true, predictions, class_names, output_path):
        """Plot ROC curves for each class.

        Args:
            y_true: True labels (indices)
            predictions: Prediction probabilities
            class_names: List of class names
            output_path: Path to save the plot
        """
        print("📊 Generating ROC curves...")

        num_classes = len(class_names)

        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(num_classes))

        # Compute ROC curve for each class
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
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

        save_path = output_path / 'roc_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   Saved to: {save_path}")

    @staticmethod
    def visualize_augmentation(data_path, class_names, num_samples=5):
        """Visualize augmentation effects on sample images.

        Args:
            data_path: Path to dataset
            class_names: List of class names
            num_samples: Number of augmented samples to show per class
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Get one image per class
        num_classes = len(class_names)
        fig, axes = plt.subplots(num_classes, num_samples + 1,
                                  figsize=(3 * (num_samples + 1), 3 * num_classes))

        # Make axes iterable if only one class
        if num_classes == 1:
            axes = np.expand_dims(axes, axis=0)

        for class_idx, class_name in enumerate(class_names):
            class_dir = data_path / class_name
            # Use utility to find images
            supported_formats = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
            images = get_all_images(class_dir, supported_formats)

            if not images:
                continue

            # Load original image
            img = tf.keras.preprocessing.image.load_img(
                images[0], target_size=Config.IMG_SIZE
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)

            # Show original
            axes[class_idx, 0].imshow(img_array.astype('uint8'))
            axes[class_idx, 0].set_title(f'{class_name}\n(Original)')
            axes[class_idx, 0].axis('off')

            # Show augmented versions
            datagen = ImageDataGenerator(
                rotation_range=Config.ROTATION_RANGE,
                width_shift_range=Config.WIDTH_SHIFT_RANGE,
                height_shift_range=Config.HEIGHT_SHIFT_RANGE,
                zoom_range=Config.ZOOM_RANGE,
                horizontal_flip=Config.HORIZONTAL_FLIP,
                vertical_flip=Config.VERTICAL_FLIP,
                brightness_range=Config.BRIGHTNESS_RANGE,
                fill_mode=Config.FILL_MODE
            )

            img_batch = np.expand_dims(img_array, 0)
            aug_iter = datagen.flow(img_batch, batch_size=1)

            for i in range(num_samples):
                aug_img = next(aug_iter)[0].astype('uint8')
                axes[class_idx, i + 1].imshow(aug_img)
                axes[class_idx, i + 1].set_title(f'Aug {i + 1}')
                axes[class_idx, i + 1].axis('off')

        plt.suptitle('Data Augmentation Preview', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = Config.get_output_path() / 'augmentation_preview.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📸 Augmentation preview saved to: {output_path}")

    @staticmethod
    def print_dataset_analysis(dataset_stats, data_path):
        """Print formatted dataset analysis.

        Args:
            dataset_stats: Dictionary with dataset statistics
            data_path: Path to dataset
        """
        print_section_header("📊 DATASET ANALYSIS")
        print(f"📁 Path: {data_path}")
        print(f"🖼️  Total images: {dataset_stats['total']}")
        print(f"🏷️  Classes: {dataset_stats['classes']}")
        print(f"📈 Avg per class: {dataset_stats['avg_samples_per_class']:.1f}")
        print(f"⚖️  Imbalance ratio: {dataset_stats['imbalance_ratio']:.2f}")
        print_subsection("Class Distribution:")

        for name, count in sorted(dataset_stats['distribution'].items()):
            pct = count / dataset_stats['total'] * 100
            bar_len = int(pct / 100 * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {name:20s} │ {bar} │ {count:4d} ({pct:5.1f}%)")

        print("=" * 60 + "\n")
