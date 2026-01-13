# PCB Defect Detector

Deep learning pipeline for detecting defects in Printed Circuit Boards using transfer learning.

## Dataset

[PCB Defects](https://www.kaggle.com/datasets/akhatova/pcb-defects) - 6 defect classes:
- Missing Hole
- Mouse Bite  
- Open Circuit
- Short
- Spur
- Spurious Copper

## Features

- Transfer learning (MobileNetV2, ResNet50, EfficientNetB0)
- Automatic class weight balancing
- Two-phase training: frozen base → fine-tuning
- Data augmentation pipeline
- Comprehensive evaluation metrics

## Project Structure

```
├── config.py           # Configuration (auto-detects Kaggle/Colab/Local)
├── data_manager.py     # Dataset download & parsing
├── data_pipeline.py    # Data splits, augmentation, tf.data
├── model_builder.py    # Transfer learning models
├── trainer.py          # Training with fine-tuning support
├── evaluator.py        # Metrics, confusion matrix, plots
├── main.py             # CLI entry point
├── kaggle_notebook.ipynb
└── requirements.txt
```

## Quick Start

### On Kaggle

1. Create new Notebook
2. Add dataset: `akhatova/pcb-defects`
3. Enable GPU
4. Run:

```python
!git clone https://github.com/alainpaluku/pcb-defect-detector.git
%cd pcb-defect-detector
!python main.py --epochs 25 --finetune-epochs 15
```

Or upload `kaggle_notebook.ipynb` directly.

### Local

```bash
pip install -r requirements.txt

# Set Kaggle credentials
export KAGGLE_API_TOKEN='{"username":"...","key":"..."}'

# Train
python main.py --epochs 30 --model MobileNetV2
```

## CLI Options

```
--epochs N          Training epochs (default: 30)
--batch-size N      Batch size (default: 32)
--lr RATE           Learning rate (default: 1e-3)
--model NAME        MobileNetV2|ResNet50|EfficientNetB0
--no-finetune       Skip fine-tuning phase
--finetune-epochs N Fine-tuning epochs (default: 20)
```

## Output

- `checkpoints/` - Model weights
- `results/` - Confusion matrix, training curves, classification report

## License

MIT
