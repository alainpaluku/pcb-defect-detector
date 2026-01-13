# PCB Defect Detector

A production-grade deep learning pipeline for detecting defects in Printed Circuit Boards (PCBs) using transfer learning.

## Dataset

Uses the [PCB Defects dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects) with 6 defect classes:
- Missing Hole
- Mouse Bite  
- Open Circuit
- Short
- Spur
- Spurious Copper

## Project Structure

```
pcb-defect-detector/
├── config.py           # Configuration dataclasses
├── data_manager.py     # Kaggle authentication & download
├── data_pipeline.py    # Data splits, augmentation, class weights
├── model_builder.py    # Transfer learning model (MobileNetV2/ResNet50)
├── trainer.py          # Training loop with callbacks
├── evaluator.py        # Metrics, confusion matrix, plots
├── main.py             # Pipeline orchestrator
├── utils.py            # Helper functions
├── kaggle_notebook.ipynb  # Ready-to-run Kaggle notebook
└── requirements.txt
```

## Quick Start on Kaggle

1. Create a new Kaggle Notebook
2. Add dataset: `akhatova/pcb-defects`
3. Enable GPU (Settings → Accelerator → GPU)
4. Run:

```python
!git clone https://github.com/alainpaluku/pcb-defect-detector.git
%cd pcb-defect-detector
!python main.py
```

Or upload `kaggle_notebook.ipynb` directly.

## Local Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set Kaggle credentials
export KAGGLE_API_TOKEN='{"username":"your_user","key":"your_key"}'

# Run pipeline
python main.py
```

## Features

- **Transfer Learning**: MobileNetV2 or ResNet50 with frozen base
- **Class Imbalance Handling**: Automatic class weight calculation
- **Data Augmentation**: Flips, rotation, brightness, contrast
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Evaluation**: Classification report, confusion matrix, training curves

## License

MIT
