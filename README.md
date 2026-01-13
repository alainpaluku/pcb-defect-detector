# 🔬 PCB Defect Detection

Deep Learning system for Automated Optical Inspection (AOI) of PCBs using MobileNetV2.

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/datasets/akhatova/pcb-defects)
[![Python](https://img.shields.io/badge/Python-3.8+-green?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## ✨ Features

- **6 defect types**: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
- **>95% accuracy** with MobileNetV2 transfer learning + fine-tuning
- **Edge-deployable**: ~7MB TFLite model, <50ms inference
- **Kaggle-ready**: Auto-detects environment, works out of the box
- **Complete pipeline**: Data analysis, augmentation, training, evaluation, export

## 🚀 Quick Start

### Kaggle (Recommended)

1. Create a new Kaggle notebook
2. Add dataset: `akhatova/pcb-defects`
3. Enable GPU accelerator
4. Run:

```python
!git clone https://github.com/alainpaluku/pcb-defect-detector.git
%cd pcb-defect-detector

from src.trainer import TrainingManager
trainer = TrainingManager()
metrics = trainer.run_pipeline(fine_tune=True)
```

### Local

```bash
git clone https://github.com/alainpaluku/pcb-defect-detector.git
cd pcb-defect-detector
pip install -r requirements.txt

# Download dataset (requires Kaggle API)
python main.py --download

# Train with fine-tuning
python main.py --epochs 50 --batch-size 32
```

## 📁 Project Structure

```
pcb-defect-detector/
├── src/
│   ├── config.py          # Configuration (auto-detects Kaggle)
│   ├── data_ingestion.py  # Data loading, augmentation, analysis
│   ├── model.py           # MobileNetV2 classifier
│   ├── trainer.py         # Training pipeline with fine-tuning
│   └── kaggle_setup.py    # Kaggle API helper
├── notebooks/
│   └── pcb_defect_detection.ipynb  # Complete Kaggle notebook
├── tests/
│   └── test_model.py
├── main.py
└── requirements.txt
```

## 🏗️ Model Architecture

```
Input (224x224x3)
    ↓
MobileNetV2 (ImageNet pretrained)
    ↓
GlobalAveragePooling → BatchNorm
    ↓
Dense(256, L2) → Dropout(0.5)
    ↓
Dense(128, L2) → Dropout(0.3)
    ↓
Softmax (6 classes)
```

## 📊 Training Pipeline

1. **Data Analysis**: Class distribution, imbalance detection
2. **Augmentation**: Rotation, shifts, zoom, flips, brightness (optimized for PCB)
3. **Phase 1 - Transfer Learning**: Frozen MobileNetV2 base
4. **Phase 2 - Fine-tuning**: Unfreeze last 50 layers with low LR (1e-5)
5. **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
6. **Evaluation**: Confusion matrix, ROC curves, classification report

## 📈 Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | >95% | ~96% |
| Precision | >93% | ~95% |
| Recall | >90% | ~94% |
| F1 Score | >92% | ~94% |
| AUC | >0.98 | ~0.99 |
| Inference | <50ms | ~30ms |

## ⚙️ CLI Options

```bash
python main.py --help

Options:
  --download       Download dataset from Kaggle
  --epochs         Number of training epochs (default: 50)
  --batch-size     Batch size (default: 32)
  --lr             Learning rate (default: 0.0001)
  --no-fine-tune   Skip fine-tuning phase
```

## 📦 Dataset

[PCB Defects by Akhatova](https://www.kaggle.com/datasets/akhatova/pcb-defects)

- 1386 images (693 normal + 693 rotated)
- 6 defect classes
- ~115-116 images per class
- Each image contains 3-5 defects of the same category

## 💾 Output Files

After training:

| File | Description |
|------|-------------|
| `pcb_model.keras` | Keras model (recommended) |
| `pcb_model.h5` | H5 format (legacy) |
| `saved_model/` | TensorFlow SavedModel |
| `pcb_model_fp16.tflite` | TFLite for edge deployment |
| `training_history.png` | Training curves |
| `confusion_matrix.png` | Confusion matrix |
| `roc_curves.png` | ROC curves per class |
| `classification_report.txt` | Per-class metrics |

## 🔧 Configuration

Key parameters in `src/config.py`:

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LAYERS = 50
FINE_TUNE_LR = 1e-5
```

## 📝 License

MIT - [alainpaluku](https://github.com/alainpaluku)

## 🙏 Acknowledgments

- Dataset: [Akhatova](https://www.kaggle.com/akhatova)
- Original paper: Huang & Wei, "A PCB Dataset for Defects Detection and Classification"
