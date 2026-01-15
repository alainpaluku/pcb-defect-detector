# PCB Defect Detection

A YOLOv8-based Computer Vision solution for automated detection and classification of defects on Printed Circuit Boards (PCBs).

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/datasets/akhatova/pcb-defects)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## Features

- **Detection**: Locate defects with bounding boxes
- **Classification**: Identify 6 types of PCB defects
- **Real-time**: Fast inference with YOLOv8
- **Export**: ONNX format for deployment

## Defect Classes

| ID | Defect | Description |
|----|--------|-------------|
| 0 | `missing_hole` | Missing drill hole |
| 1 | `mouse_bite` | Irregular edge |
| 2 | `open_circuit` | Broken trace |
| 3 | `short` | Short circuit |
| 4 | `spur` | Copper protrusion |
| 5 | `spurious_copper` | Unwanted copper |

## Quick Start

### Kaggle (Recommended)

**Prerequisites:**
1. Add dataset `akhatova/pcb-defects` via **"+ Add Input"**
2. Enable **GPU** in notebook settings

```python
import os
os.chdir('/kaggle/working')
!rm -rf pcb-defect-detector
!git clone https://github.com/alainpaluku/pcb-defect-detector.git
os.chdir('/kaggle/working/pcb-defect-detector')
!pip install ultralytics -q
!python run_kaggle.py
```

### Local

```bash
git clone https://github.com/alainpaluku/pcb-defect-detector.git
cd pcb-defect-detector
pip install -r requirements.txt

# Train
python main.py train --epochs 50

# Detect
python main.py detect path/to/image.jpg --save

# Export
python main.py export --model output/pcb_model.pt --format onnx
```

## Usage

### Training

```python
from src.trainer import TrainingManager

trainer = TrainingManager()
metrics = trainer.run_pipeline(epochs=50)
print(f"mAP@50: {metrics['mAP50']:.4f}")
```

### Inference

```python
from src.detector import PCBInspector

inspector = PCBInspector("pcb_model.pt")
detections = inspector.inspect("pcb_image.jpg")

for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2%}")
    print(f"  Box: {det['bbox']}")
```

### Batch Processing

```python
results = inspector.inspect_batch("images_folder/")
for img_name, detections in results.items():
    summary = inspector.get_summary(detections)
    print(f"{img_name}: {summary['status']} - {summary['defect_count']} defects")
```

## Project Structure

```
pcb-defect-detector/
├── src/
│   ├── config.py         # Configuration
│   ├── data_ingestion.py # Data loading & conversion
│   ├── model.py          # YOLOv8 model wrapper
│   ├── detector.py       # Inference interface
│   ├── trainer.py        # Training pipeline
│   ├── utils.py          # Utilities
│   └── main.py           # CLI commands
├── tests/
│   └── test_model.py     # Unit tests
├── main.py               # Entry point
├── run_kaggle.py         # Kaggle runner
└── requirements.txt
```

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | 0.2219 |
| mAP@50-95 | 0.1667 |
| Precision | 0.1653 |
| Recall | 0.6999 |

*Results from 50 epochs training on Kaggle with YOLOv8n. Better results can be achieved with more epochs (100-150) and larger models (YOLOv8s/m).*

## Dataset

[PCB Defects - Akhatova](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- 1386 images with XML annotations
- 6 defect classes
- VOC format bounding boxes

## Author

**Alain Paluku** - [@alainpaluku](https://github.com/alainpaluku)

## License

MIT License
