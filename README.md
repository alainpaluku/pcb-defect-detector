# PCB Defect Detection

A YOLOv8-based Computer Vision solution for automated detection and classification of defects on Printed Circuit Boards (PCBs), with a desktop application built with Tauri.

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/datasets/akhatova/pcb-defects)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)](https://ultralytics.com)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-FFC131?style=flat-square&logo=tauri)](https://tauri.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## Features

- **Detection**: Locate defects with bounding boxes
- **Classification**: Identify 6 types of PCB defects
- **Real-time**: Fast inference with YOLOv8
- **Desktop App**: Cross-platform application with Tauri
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

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | **95.5%** |
| mAP@50-95 | 51.8% |
| Precision | 95.4% |
| Recall | 91.0% |

*Results from 100 epochs training on Kaggle with YOLOv8s.*

## Quick Start

### Training on Kaggle (Recommended)

**Prerequisites:**
1. Add dataset `akhatova/pcb-defects` via **"+ Add Input"**
2. Enable **GPU T4** in notebook settings

```python
!pip install ultralytics -q
!rm -rf /kaggle/working/pcb-defect-detector
!git clone https://github.com/alainpaluku/pcb-defect-detector.git
%cd /kaggle/working/pcb-defect-detector
!python run_kaggle.py
```

### Local Training

```bash
git clone https://github.com/alainpaluku/pcb-defect-detector.git
cd pcb-defect-detector
pip install -r requirements.txt

# Train
python main.py train --epochs 100

# Detect
python main.py detect path/to/image.jpg --save

# Export to ONNX
python main.py export --model output/pcb_model.pt --format onnx
```

## Desktop Application

A cross-platform desktop app built with Tauri 2 and React for real-time PCB inspection.

### Features
- Drag & drop image upload
- Real-time defect detection with bounding boxes
- Classification with confidence scores
- Dark/Light mode
- History of inspections

### Build from Source

**Prerequisites:**
- Node.js 18+
- Rust 1.70+
- System dependencies (see below)

**Fedora:**
```bash
sudo dnf install webkit2gtk4.1-devel javascriptcoregtk4.1-devel gtk3-devel glib2-devel openssl-devel libsoup-devel librsvg2-devel
```

**Ubuntu/Debian:**
```bash
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev
```

**Build:**
```bash
cd desktop-app
npm install
npm run tauri build
```

The built application will be in:
- **Linux**: `src-tauri/target/release/bundle/deb/` or `rpm/`
- **Windows**: `src-tauri/target/release/bundle/msi/`
- **macOS**: `src-tauri/target/release/bundle/dmg/`

### Using the Trained Model

1. Train on Kaggle and download `best.pt` or `pcb_model.onnx`
2. Place the ONNX model in `desktop-app/public/model/pcb_detector.onnx`
3. Build and run the app

## Project Structure

```
pcb-defect-detector/
├── src/                    # Python training code
│   ├── config.py           # Configuration
│   ├── data_ingestion.py   # Data loading & conversion
│   ├── model.py            # YOLOv8 model wrapper
│   ├── detector.py         # Inference interface
│   ├── trainer.py          # Training pipeline
│   └── utils.py            # Utilities
├── desktop-app/            # Tauri desktop application
│   ├── src/                # React frontend
│   ├── src-tauri/          # Rust backend
│   └── public/model/       # ONNX model location
├── run_kaggle.py           # Kaggle runner
└── requirements.txt
```

## Model Architecture

| Property | Value |
|----------|-------|
| Base Model | YOLOv8s |
| Input Shape | (1, 3, 640, 640) |
| Output | Bounding boxes + 6 classes |
| ONNX Size | ~22 MB |

## Dataset

[PCB Defects - Akhatova](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- 1386 images with XML annotations
- 6 defect classes
- VOC format bounding boxes

## Author

**Alain Paluku** - [@alainpaluku](https://github.com/alainpaluku)

## License

MIT License
