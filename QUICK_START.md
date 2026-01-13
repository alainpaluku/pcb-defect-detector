# Quick Start Guide

Get up and running with PCB Defect Detection in 5 minutes!

---

## 🚀 Kaggle (Fastest - Recommended)

### 3 Steps to Train

1. **Open Kaggle Notebook**
   - Go to [Kaggle.com](https://www.kaggle.com)
   - Create new notebook
   - Add dataset: Search "akhatova/pcb-defects" → Add

2. **Copy-Paste This Code**
   ```python
   # Install if needed
   !pip install -q tensorflow scikit-learn seaborn
   
   # Import
   import tensorflow as tf
   from pathlib import Path
   
   # Verify dataset
   data_path = Path("/kaggle/input/pcb-defects")
   print(f"Dataset found: {data_path.exists()}")
   print(f"Classes: {[d.name for d in data_path.iterdir() if d.is_dir()]}")
   ```

3. **Upload & Run Notebook**
   - Download: `notebooks/pcb_defect_detection.ipynb`
   - Upload to Kaggle
   - Enable GPU (Settings → Accelerator → GPU)
   - Click "Run All"

**Done!** Model trains in 30-60 minutes.

---

## 💻 Local Machine

### Prerequisites
- Python 3.8+
- 8GB+ RAM
- (Optional) NVIDIA GPU with CUDA

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pcb-defect-detector.git
cd pcb-defect-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Get Dataset

**Option A: Kaggle API (Automated)**
```bash
# Set up Kaggle credentials
# Get API key from: https://www.kaggle.com/settings/account
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Download dataset
python -c "from src.kaggle_setup import KaggleSetup; KaggleSetup().download_dataset('akhatova/pcb-defects')"
```

**Option B: Manual Download**
1. Go to [PCB Defects Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects)
2. Click "Download"
3. Extract to `data/pcb-defects/`

### Train Model

```bash
# Basic training (default settings)
python main.py

# Custom parameters
python main.py --epochs 100 --batch-size 32 --learning-rate 0.001
```

**Done!** Check `output/` folder for trained model.

---

## 🐍 Python API

### Minimal Example

```python
from src.trainer import TrainingManager

# One-line training
trainer = TrainingManager()
metrics = trainer.run_pipeline()

print(f"Accuracy: {metrics['accuracy']:.2%}")
```

### Custom Configuration

```python
from src.config import Config
from src.trainer import TrainingManager

# Modify settings
Config.EPOCHS = 100
Config.BATCH_SIZE = 16
Config.LEARNING_RATE = 0.0005

# Train
trainer = TrainingManager()
metrics = trainer.run_pipeline()
```

### Inference Only

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model('output/best_model.h5')

# Load and preprocess image
img = Image.open('test_pcb.jpg').resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 
               'short', 'spur', 'spurious_copper']

defect_type = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"Defect: {defect_type} ({confidence:.1%} confidence)")
```

---

## 📊 Expected Output

After training, you'll see:

```
==============================================================
PCB DEFECT DETECTION SYSTEM
Automated Optical Inspection (AOI) for Electronics Manufacturing
==============================================================

PHASE 1: DATA INGESTION
------------------------------------------------------------
Dataset Analysis:
  Total Images: 1386
  Number of Classes: 6
  Class Distribution:
    missing_hole        : 115 images (8.30%)
    mouse_bite          : 92 images (6.64%)
    open_circuit        : 246 images (17.75%)
    short               : 227 images (16.38%)
    spur                : 466 images (33.62%)
    spurious_copper     : 240 images (17.32%)

PHASE 2: MODEL ARCHITECTURE
------------------------------------------------------------
Model: MobileNetV2
Total Parameters: 3,538,984
Trainable Parameters: 1,862,152

PHASE 3: MODEL TRAINING
------------------------------------------------------------
Epoch 1/50
34/34 [==============================] - 45s - loss: 1.2345 - accuracy: 0.6543
...
Epoch 50/50
34/34 [==============================] - 38s - loss: 0.1234 - accuracy: 0.9678

PHASE 4: MODEL EVALUATION
------------------------------------------------------------
Final Performance Metrics:
  Validation Accuracy:  96.78%
  Precision:            95.23%
  Recall:               94.56%
  F1 Score:             94.89%

✓ Training completed successfully!
```

---

## 🎯 Verify Installation

Run this to check everything is set up correctly:

```python
# test_setup.py
import sys

def test_setup():
    print("Testing PCB Defect Detector Setup...\n")
    
    # Check Python version
    print(f"✓ Python {sys.version.split()[0]}")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"  {'✓' if gpus else '○'} GPU: {len(gpus)} device(s)")
    except ImportError:
        print("✗ TensorFlow not installed")
        return False
    
    # Check other dependencies
    deps = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'PIL']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} not installed")
            return False
    
    # Check project structure
    from pathlib import Path
    required_files = [
        'src/config.py',
        'src/data_ingestion.py',
        'src/model.py',
        'src/trainer.py',
        'requirements.txt'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} missing")
            return False
    
    print("\n✓ All checks passed! Ready to train.")
    return True

if __name__ == "__main__":
    test_setup()
```

Run: `python test_setup.py`

---

## 🔧 Troubleshooting

### "Dataset not found"
- **Kaggle**: Add "akhatova/pcb-defects" dataset to notebook
- **Local**: Download dataset to `data/pcb-defects/`

### "Out of memory"
```python
# Reduce batch size
Config.BATCH_SIZE = 16  # or 8
```

### "Slow training"
- Enable GPU in Kaggle (Settings → Accelerator → GPU)
- For local: Install `tensorflow-gpu` if you have NVIDIA GPU

### "Import errors"
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### "Low accuracy (<90%)"
- Train for more epochs: `--epochs 100`
- Enable fine-tuning (see notebook)
- Check dataset quality

---

## 📚 Next Steps

1. **Review Results**: Check confusion matrix and classification report
2. **Test Inference**: Try predictions on sample images
3. **Deploy**: See `DEPLOYMENT.md` for production setup
4. **Customize**: Modify `src/config.py` for your needs

---

## 🆘 Need Help?

- **Documentation**: `README.md`, `PROBLEM_DEFINITION.md`
- **Kaggle Guide**: `KAGGLE_SETUP.md`
- **Deployment**: `DEPLOYMENT.md`
- **Issues**: Open GitHub issue

---

**Happy Training! 🎉**
