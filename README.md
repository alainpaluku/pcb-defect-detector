# PCB Defect Detector

Production-grade deep learning pipeline for detecting defects in Printed Circuit Boards using transfer learning.

## Dataset

[PCB Defects by Akhatova](https://www.kaggle.com/datasets/akhatova/pcb-defects) - 1,386 images with 6 defect classes:
- Missing Hole
- Mouse Bite  
- Open Circuit
- Short
- Spur
- Spurious Copper

## Features

✅ Transfer learning (MobileNetV2, ResNet50, EfficientNetB0)  
✅ Automatic class weight balancing for imbalanced data  
✅ Two-phase training: frozen base → fine-tuning  
✅ Advanced data augmentation pipeline  
✅ Comprehensive evaluation (F1, precision, recall, confusion matrix)  
✅ Auto-detection of environment (Kaggle/Colab/Local)  

## Project Structure

```
pcb-defect-detector/
├── config.py           # Configuration with auto-detection
├── data_manager.py     # Dataset loading & parsing
├── data_pipeline.py    # Data splits, augmentation, tf.data
├── model_builder.py    # Transfer learning models
├── trainer.py          # Training with fine-tuning
├── evaluator.py        # Metrics & visualizations
├── main.py             # CLI entry point
├── kaggle_script.py    # Standalone script for Kaggle
├── requirements.txt
└── README.md
```

## Quick Start

### On Kaggle (Recommended)

**⚠️ IMPORTANT: Vous DEVEZ ajouter le dataset avant d'exécuter le code !**

1. Créer un nouveau Notebook Kaggle
2. **Ajouter le dataset** : Cliquez sur "+ Add Data" → Cherchez `akhatova/pcb-defects` → Cliquez "Add"
3. Activer le GPU : Settings → Accelerator → GPU T4 x2
4. Exécuter le code :

```python
!wget https://raw.githubusercontent.com/alainpaluku/pcb-defect-detector/main/kaggle_script.py
%run kaggle_script.py
```

📖 **Guide détaillé** : Voir [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) pour les instructions complètes.

### Local Training

```bash
# Clone repository
git clone https://github.com/alainpaluku/pcb-defect-detector.git
cd pcb-defect-detector

# Install dependencies
pip install -r requirements.txt

# Set Kaggle credentials
export KAGGLE_API_TOKEN='{"username":"your_username","key":"your_api_key"}'

# Train with default settings
python main.py

# Or customize
python main.py --epochs 30 --model ResNet50 --finetune-epochs 20
```

## CLI Options

```bash
python main.py [OPTIONS]

Options:
  --epochs N              Training epochs (default: 30)
  --batch-size N          Batch size (default: 32)
  --lr RATE               Learning rate (default: 1e-3)
  --model NAME            MobileNetV2|ResNet50|EfficientNetB0
  --no-finetune           Skip fine-tuning phase
  --finetune-epochs N     Fine-tuning epochs (default: 20)
```

## Output

After training, you'll find:

- `checkpoints/` - Saved model weights (best.keras, ft_best.keras, final_model.keras)
- `results/` - Evaluation artifacts:
  - `confusion_matrix.png` - Normalized confusion matrix
  - `training_curves.png` - Loss and accuracy plots
  - `misclassified.png` - Examples of misclassified images
  - `classification_report.txt` - Detailed metrics per class

## Architecture

**Base Model:** MobileNetV2 (pretrained on ImageNet)  
**Custom Head:**
- Global Average Pooling
- Batch Normalization
- Dense(256) + ReLU + L2 regularization
- Dropout(0.5)
- Dense(6) + Softmax

**Training Strategy:**
1. Phase 1: Train with frozen base (30 epochs)
2. Phase 2: Fine-tune last 30 layers (20 epochs)

## Results

Expected performance on test set:
- Accuracy: ~92-95%
- F1 Macro: ~0.90-0.93
- F1 Weighted: ~0.92-0.95

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- See `requirements.txt` for full list

## License

MIT License - See LICENSE file for details

## Author

Alain Paluku - [GitHub](https://github.com/alainpaluku)
