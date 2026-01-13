# PCB Defect Detector - Kaggle Training Pipeline

Production-grade deep learning pipeline for detecting defects in Printed Circuit Boards using transfer learning.

**Optimized for Kaggle Notebooks**

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
✅ Automatic class weight balancing  
✅ Two-phase training: frozen base → fine-tuning  
✅ Advanced data augmentation  
✅ Comprehensive evaluation metrics  

## Project Structure

```
pcb-defect-detector/
├── config.py              # Configuration
├── data_manager.py        # Dataset loading
├── data_pipeline.py       # Data preprocessing
├── model_builder.py       # Model architecture
├── trainer.py             # Training logic
├── evaluator.py           # Evaluation & metrics
├── train_notebook.ipynb   # Main notebook
└── requirements.txt       # Dependencies
```

## 🚀 How to Use on Kaggle

### Step 1: Create Notebook
- Go to [Kaggle](https://www.kaggle.com)
- Click **"Create"** → **"New Notebook"**

### Step 2: Add Dataset
- Click **"+ Add Data"** (top right)
- Search: `akhatova/pcb-defects`
- Click **"Add"**

### Step 3: Enable GPU
- Click **⋮** (3 dots, top right)
- Select **"Accelerator"** → **"GPU T4 x2"**
- Click **"Save"**

### Step 4: Run Code

**Option A: Upload Notebook (Easiest)**
1. Download [train_notebook.ipynb](https://github.com/alainpaluku/pcb-defect-detector/blob/main/train_notebook.ipynb)
2. In Kaggle: **File** → **Upload Notebook**
3. Click **"Run All"**

**Option B: Clone Repository**

Copy this into a cell and run:

```python
# Clone repository
!git clone https://github.com/alainpaluku/pcb-defect-detector.git
import sys
sys.path.insert(0, 'pcb-defect-detector')

# Import modules
from pathlib import Path
from config import PipelineConfig, DataConfig, ModelConfig, TrainingConfig
from data_manager import KaggleDataManager
from data_pipeline import DataPipeline
from model_builder import PCBModelBuilder
from trainer import Trainer
from evaluator import Evaluator

# Configure
config = PipelineConfig(
    data=DataConfig(data_dir=Path('/kaggle/input/pcb-defects')),
    training=TrainingConfig(
        epochs=25,
        fine_tune_epochs=15,
        checkpoint_dir=Path('/kaggle/working/checkpoints')
    ),
    results_dir=Path('/kaggle/working/results')
)

# Load data
dm = KaggleDataManager(config.data)
dm.download_dataset()
class_images = dm.parse_directory_structure()
class_names = dm.get_class_names()

# Prepare pipeline
dp = DataPipeline(config.data, class_names)
dp.prepare_data(class_images)

# Build model
mb = PCBModelBuilder(config.model, config.data, len(class_names))
model = mb.build()

# Train
trainer = Trainer(config.training, model)
trainer.compile()
trainer.train(dp.get_train_dataset(), dp.get_val_dataset(), dp.get_class_weights())

# Fine-tune
mb.unfreeze_layers(30)
trainer.fine_tune(dp.get_train_dataset(), dp.get_val_dataset(), dp.get_class_weights())

# Evaluate
evaluator = Evaluator(model, class_names, config.results_dir)
results = evaluator.generate_full_report(
    dp.get_test_dataset(), 
    dp.get_test_labels(),
    trainer.get_combined_history(),
    dp.get_test_paths()
)

print(f"\n✅ Training Complete!")
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
print(f"F1 Macro: {results['f1_macro']:.4f}")

# Display results
from IPython.display import Image, display
display(Image(str(config.results_dir / 'training_curves.png')))
display(Image(str(config.results_dir / 'confusion_matrix_normalized.png')))
```

## Output Files

After training, find results in `/kaggle/working/`:

- `checkpoints/best.keras` - Best model (phase 1)
- `checkpoints/ft_best.keras` - Best model (fine-tuned)
- `results/confusion_matrix_normalized.png`
- `results/training_curves.png`
- `results/misclassified.png`
- `results/classification_report.txt`

## Architecture

- **Base:** MobileNetV2 (ImageNet pretrained)
- **Head:** GAP → BatchNorm → Dense(256) → Dropout(0.5) → Dense(6)
- **Training:** 2-phase (frozen → fine-tuned)

## Expected Performance

- Accuracy: ~92-95%
- F1 Macro: ~0.90-0.93
- Training time: ~30-40 min (GPU T4 x2)

## Troubleshooting

**Error: Dataset not found**
→ Make sure you added `akhatova/pcb-defects` dataset

**Error: Out of memory**
→ Enable GPU or reduce batch size in config

**Error: git clone forbidden**
→ Use "Upload Notebook" method instead

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- See `requirements.txt`

## License

MIT

## Author

Alain Paluku - [GitHub](https://github.com/alainpaluku)
