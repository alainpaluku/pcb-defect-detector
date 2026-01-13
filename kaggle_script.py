"""
PCB Defect Detector - Kaggle Standalone Script
Copy this entire file into a Kaggle notebook cell.
Make sure to: 1) Add dataset akhatova/pcb-defects  2) Enable GPU
"""

import os, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============== CONFIGURATION ==============
DATA_DIR = Path('/kaggle/input/pcb-defects')
OUTPUT_DIR = Path('/kaggle/working')
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
RESULTS_DIR = OUTPUT_DIR / 'results'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
FINE_TUNE_EPOCHS = 15
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
DROPOUT = 0.5
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============== DATA LOADING ==============
def find_dataset() -> Path:
    """Find dataset root directory."""
    candidates = [DATA_DIR / 'PCB_DATASET', DATA_DIR / 'pcb_defects', DATA_DIR]
    for path in candidates:
        if path.exists():
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if subdirs:
                return path
    for item in DATA_DIR.iterdir():
        if item.is_dir():
            return item
    raise FileNotFoundError(f"Dataset not found in {DATA_DIR}")

def parse_dataset(dataset_path: Path) -> Tuple[Dict[str, List[Path]], List[str]]:
    """Parse dataset directory structure."""
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    class_images = {}
    
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_ext]
        if images:
            class_images[class_dir.name] = images
            logger.info(f"  {class_dir.name}: {len(images)} images")
    
    class_names = sorted(class_images.keys())
    total = sum(len(v) for v in class_images.values())
    logger.info(f"Total: {len(class_names)} classes, {total} images")
    return class_images, class_names

# ============== DATA PIPELINE ==============
def prepare_splits(class_images: Dict, class_names: List[str]) -> Tuple:
    """Create train/val/test splits."""
    all_paths, all_labels = [], []
    for name, images in class_images.items():
        label = class_names.index(name)
        all_paths.extend(images)
        all_labels.extend([label] * len(images))
    
    all_paths, all_labels = np.array(all_paths), np.array(all_labels)
    
    # Split
    train_val_p, test_p, train_val_l, test_l = train_test_split(
        all_paths, all_labels, test_size=TEST_SPLIT, stratify=all_labels, random_state=SEED)
    
    val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
    train_p, val_p, train_l, val_l = train_test_split(
        train_val_p, train_val_l, test_size=val_ratio, stratify=train_val_l, random_state=SEED)
    
    logger.info(f"Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")
    return (train_p, train_l), (val_p, val_l), (test_p, test_l)

def compute_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute class weights for imbalanced data."""
    unique = np.unique(labels)
    weights = compute_class_weight('balanced', classes=unique, y=labels)
    return {int(c): float(w) for c, w in zip(unique, weights)}

def load_image(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load and preprocess image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply data augmentation."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return tf.clip_by_value(image, 0.0, 1.0), label

def create_dataset(paths, labels, training: bool = False) -> tf.data.Dataset:
    """Create tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in paths], labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ============== MODEL ==============
def build_model(num_classes: int) -> Tuple[Model, Model]:
    """Build transfer learning model."""
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False
    
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    logger.info(f"Model built. Trainable params: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    return model, base

def unfreeze_layers(base: Model, num_layers: int = 30) -> None:
    """Unfreeze last N layers for fine-tuning."""
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False
    logger.info(f"Unfroze last {num_layers} layers")

# ============== TRAINING ==============
def get_callbacks(prefix: str = "") -> List:
    """Create training callbacks."""
    return [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(CHECKPOINT_DIR / f'{prefix}best.keras'), monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
    ]

def train_model(model: Model, train_ds, val_ds, class_weights: Dict, epochs: int, lr: float, prefix: str = "") -> dict:
    """Train the model."""
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, 
                       callbacks=get_callbacks(prefix), class_weight=class_weights, verbose=1)
    return history.history

# ============== EVALUATION ==============
def evaluate_model(model: Model, test_ds, test_labels: np.ndarray, class_names: List[str]) -> Dict:
    """Evaluate model and generate metrics."""
    preds = model.predict(test_ds, verbose=1)
    pred_labels = np.argmax(preds, axis=1)
    
    loss, acc = model.evaluate(test_ds, verbose=0)
    f1_m = f1_score(test_labels, pred_labels, average='macro')
    f1_w = f1_score(test_labels, pred_labels, average='weighted')
    
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"F1 Macro: {f1_m:.4f}, F1 Weighted: {f1_w:.4f}")
    
    report = classification_report(test_labels, pred_labels, target_names=class_names, digits=4)
    logger.info(f"\n{report}")
    
    with open(RESULTS_DIR / 'report.txt', 'w') as f:
        f.write(report)
    
    return {'accuracy': acc, 'f1_macro': f1_m, 'f1_weighted': f1_w, 'pred_labels': pred_labels}

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    """Plot confusion matrix."""
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=150)
    plt.show()

def plot_history(history: Dict):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['loss']) + 1)
    
    axes[0].plot(epochs, history['loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['accuracy'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Val')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'training_curves.png', dpi=150)
    plt.show()

# ============== MAIN EXECUTION ==============
def main():
    """Run the complete pipeline."""
    logger.info("=" * 50)
    logger.info("PCB Defect Detector - Starting")
    logger.info("=" * 50)
    
    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Using {len(gpus)} GPU(s)")
    
    # Load data
    logger.info("\n[1/5] Loading dataset...")
    dataset_path = find_dataset()
    class_images, class_names = parse_dataset(dataset_path)
    
    # Prepare splits
    logger.info("\n[2/5] Preparing data...")
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = prepare_splits(class_images, class_names)
    class_weights = compute_weights(train_l)
    logger.info(f"Class weights: {class_weights}")
    
    train_ds = create_dataset(train_p, train_l, training=True)
    val_ds = create_dataset(val_p, val_l, training=False)
    test_ds = create_dataset(test_p, test_l, training=False)
    
    # Build model
    logger.info("\n[3/5] Building model...")
    model, base_model = build_model(len(class_names))
    
    # Train (frozen base)
    logger.info("\n[4/5] Training (frozen base)...")
    history1 = train_model(model, train_ds, val_ds, class_weights, EPOCHS, LEARNING_RATE)
    
    # Fine-tune
    logger.info("\n[4/5] Fine-tuning...")
    unfreeze_layers(base_model, 30)
    history2 = train_model(model, train_ds, val_ds, class_weights, FINE_TUNE_EPOCHS, FINE_TUNE_LR, prefix="ft_")
    
    # Combine histories
    combined_history = {k: history1[k] + history2.get(k, []) for k in history1}
    
    # Save model
    model.save(str(OUTPUT_DIR / 'pcb_model.keras'))
    logger.info(f"Model saved to {OUTPUT_DIR / 'pcb_model.keras'}")
    
    # Evaluate
    logger.info("\n[5/5] Evaluating...")
    results = evaluate_model(model, test_ds, test_l, class_names)
    
    # Plots
    plot_history(combined_history)
    plot_confusion_matrix(test_l, results['pred_labels'], class_names)
    
    logger.info("\n" + "=" * 50)
    logger.info("COMPLETED")
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info(f"F1 Macro: {results['f1_macro']:.4f}")
    logger.info("=" * 50)
    
    return results

# Run
if __name__ == "__main__":
    results = main()
