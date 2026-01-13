# Kaggle Setup Guide

## Quick Start for Kaggle Environment

This guide helps you run the PCB Defect Detection system in a Kaggle notebook.

---

## Method 1: Direct Notebook Upload (Recommended)

### Step 1: Add Dataset
1. Go to your Kaggle notebook
2. Click "Add Data" in the right sidebar
3. Search for "akhatova/pcb-defects"
4. Click "Add" to attach the dataset

### Step 2: Upload Notebook
1. Download `notebooks/pcb_defect_detection.ipynb` from this repository
2. In Kaggle, click "File" → "Upload Notebook"
3. Select the downloaded notebook
4. Enable GPU: Settings → Accelerator → GPU

### Step 3: Run
1. Click "Run All" or execute cells sequentially
2. The notebook will automatically detect it's running on Kaggle
3. Training will start with the attached dataset

---

## Method 2: Copy-Paste Code

If you prefer to work in a fresh Kaggle notebook:

### Step 1: Create New Notebook
1. Go to Kaggle.com
2. Click "Code" → "New Notebook"
3. Add dataset: "akhatova/pcb-defects"
4. Enable GPU

### Step 2: Install Dependencies

```python
# Cell 1: Install packages (if needed)
!pip install -q tensorflow scikit-learn seaborn
```

### Step 3: Copy Source Code

Create separate cells for each module:

**Cell 2: Config Module**
```python
# Copy entire contents of src/config.py
```

**Cell 3: Data Ingestion Module**
```python
# Copy entire contents of src/data_ingestion.py
```

**Cell 4: Model Module**
```python
# Copy entire contents of src/model.py
```

**Cell 5: Trainer Module**
```python
# Copy entire contents of src/trainer.py
```

### Step 4: Run Training

**Cell 6: Execute Pipeline**
```python
from trainer import TrainingManager

trainer = TrainingManager()
metrics = trainer.run_pipeline()
```

---

## Method 3: Upload as Kaggle Dataset

For reusable code across multiple notebooks:

### Step 1: Create Dataset
1. Zip the `src/` folder
2. Go to Kaggle → "Datasets" → "New Dataset"
3. Upload the zip file
4. Name it "pcb-defect-detector-src"
5. Make it public or private

### Step 2: Use in Notebook
```python
# Add your dataset to the notebook
import sys
sys.path.append('/kaggle/input/pcb-defect-detector-src')

from src.trainer import TrainingManager

trainer = TrainingManager()
metrics = trainer.run_pipeline()
```

---

## Configuration for Kaggle

The code automatically detects Kaggle environment and adjusts paths:

```python
# Automatic path detection
data_path = "/kaggle/input/pcb-defects"  # Auto-detected
output_path = "/kaggle/working"          # Auto-detected
```

No manual configuration needed!

---

## Expected Output Structure

After running, you'll find in `/kaggle/working`:

```
/kaggle/working/
├── best_model.h5                    # Best model checkpoint
├── pcb_defect_model.h5             # Final trained model
├── saved_model/                     # TensorFlow SavedModel format
├── model_architecture.json          # Model architecture
├── training_history.png             # Training curves
├── confusion_matrix.png             # Confusion matrix
├── classification_report.txt        # Detailed metrics
└── logs/                           # TensorBoard logs
```

---

## Troubleshooting

### Issue: "Dataset not found"
**Solution:** Ensure you've added "akhatova/pcb-defects" dataset to your notebook

### Issue: "Out of memory"
**Solution:** 
- Enable GPU (Settings → Accelerator → GPU)
- Reduce batch size in config: `Config.BATCH_SIZE = 16`

### Issue: "Import errors"
**Solution:** 
- Use Method 2 (copy-paste) to avoid import issues
- Or ensure src/ folder is properly uploaded as dataset

### Issue: "Slow training"
**Solution:**
- Verify GPU is enabled (check with `tf.config.list_physical_devices('GPU')`)
- Reduce epochs for testing: `Config.EPOCHS = 10`

### Issue: "Session timeout"
**Solution:**
- Kaggle notebooks timeout after 9 hours
- Save checkpoints frequently (already implemented)
- Resume from `best_model.h5` if needed

---

## Performance Expectations

On Kaggle GPU (Tesla P100):
- **Training Time**: 30-60 minutes (50 epochs)
- **Inference Time**: ~20ms per image
- **Memory Usage**: ~4-6 GB GPU RAM
- **Expected Accuracy**: 95-98%

---

## Downloading Results

To download trained models:

1. **From Notebook Output:**
   - Click on output files in the right sidebar
   - Click download icon

2. **Using Code:**
```python
from IPython.display import FileLink

# Create download links
FileLink('best_model.h5')
FileLink('training_history.png')
FileLink('confusion_matrix.png')
```

3. **Save as Dataset:**
```python
# Save output as new dataset for reuse
!mkdir -p /kaggle/working/model_output
!cp *.h5 *.png *.txt /kaggle/working/model_output/
# Then create new dataset from /kaggle/working/model_output
```

---

## Next Steps After Training

1. **Analyze Results**: Review confusion matrix and classification report
2. **Test Inference**: Try predictions on sample images
3. **Fine-tune**: If accuracy < 95%, enable fine-tuning
4. **Export**: Download model for local deployment
5. **Deploy**: Follow DEPLOYMENT.md for production setup

---

## Tips for Best Results

1. **Use GPU**: Always enable GPU for 10-20x speedup
2. **Monitor Training**: Watch for overfitting in training curves
3. **Check Class Balance**: Review dataset analysis output
4. **Adjust Hyperparameters**: Modify Config class if needed
5. **Save Frequently**: Kaggle auto-saves, but manual saves help

---

## Example Kaggle Notebook

A complete working example is available at:
- Local: `notebooks/pcb_defect_detection.ipynb`
- Kaggle: [Link to your published notebook]

---

## Support

For issues or questions:
1. Check PROBLEM_DEFINITION.md for context
2. Review README.md for architecture details
3. See DEPLOYMENT.md for production guidance
4. Open an issue on GitHub

---

Happy Training! 🚀
