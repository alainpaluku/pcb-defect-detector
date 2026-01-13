# Deployment Guide

## Production Deployment Options for PCB Defect Detection

This guide covers various deployment strategies for the trained model in industrial environments.

---

## 1. Edge Device Deployment (Recommended for Factory Floor)

### Option A: Raspberry Pi 4 with Camera Module

**Hardware Requirements:**
- Raspberry Pi 4 (4GB+ RAM)
- Raspberry Pi Camera Module v2 or USB industrial camera
- MicroSD card (32GB+)
- Power supply (5V, 3A)
- Optional: Cooling fan for continuous operation

**Software Setup:**

```bash
# Install TensorFlow Lite
pip install tflite-runtime

# Convert model to TensorFlow Lite
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('output/pcb_defect_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('pcb_model.tflite', 'wb') as f:
    f.write(tflite_model)
"

# Run inference
python edge_inference.py
```

**Performance:**
- Inference time: 40-60ms per image
- Throughput: 15-25 boards/second
- Power consumption: 5-8W
- Cost: ~$100 per unit

### Option B: NVIDIA Jetson Nano (Higher Performance)

**Hardware Requirements:**
- NVIDIA Jetson Nano Developer Kit
- Industrial camera (USB3 or CSI)
- Power supply (5V, 4A)
- MicroSD card (64GB+)

**Software Setup:**

```bash
# Install JetPack SDK (includes TensorFlow)
# Convert to TensorRT for maximum performance
pip install tensorflow-gpu
pip install pycuda

# Optimize with TensorRT
python convert_to_tensorrt.py
```

**Performance:**
- Inference time: 10-20ms per image
- Throughput: 50-100 boards/second
- Power consumption: 10-15W
- Cost: ~$150 per unit

---

## 2. Cloud API Deployment

### Option A: TensorFlow Serving (Docker)

**Setup:**

```bash
# Save model in SavedModel format (already done)
# Create Docker container
docker pull tensorflow/serving

# Run TensorFlow Serving
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/saved_model,target=/models/pcb_detector \
  -e MODEL_NAME=pcb_detector \
  tensorflow/serving
```

**Client Code:**

```python
import requests
import numpy as np
from PIL import Image

def predict_defect(image_path):
    # Load and preprocess image
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    
    # Prepare request
    data = {
        "instances": [img_array.tolist()]
    }
    
    # Send request
    response = requests.post(
        "http://localhost:8501/v1/models/pcb_detector:predict",
        json=data
    )
    
    predictions = response.json()["predictions"][0]
    return predictions
```

### Option B: FastAPI REST API

**Create API Server:**

```python
# api_server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="PCB Defect Detection API")

# Load model at startup
model = tf.keras.models.load_model('output/pcb_defect_model.h5')
class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 
               'short', 'spur', 'spurious_copper']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)[0]
    
    # Format response
    results = {
        "defect_type": class_names[np.argmax(predictions)],
        "confidence": float(np.max(predictions)),
        "all_probabilities": {
            class_names[i]: float(predictions[i]) 
            for i in range(len(class_names))
        }
    }
    
    return JSONResponse(content=results)

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**Deploy with Docker:**

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_server.py .
COPY output/pcb_defect_model.h5 .

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 3. Factory Integration

### Integration with Manufacturing Execution System (MES)

**Architecture:**

```
[Camera] → [Edge Device] → [Local Server] → [MES/ERP]
                ↓
         [Alert System]
         [Dashboard]
```

**Implementation:**

```python
# factory_integration.py
import requests
import json
from datetime import datetime

class FactoryIntegration:
    def __init__(self, mes_endpoint, alert_threshold=0.95):
        self.mes_endpoint = mes_endpoint
        self.alert_threshold = alert_threshold
    
    def process_board(self, board_id, image_path):
        # Get prediction
        prediction = self.predict_defect(image_path)
        
        # Log to MES
        self.log_to_mes(board_id, prediction)
        
        # Alert if defect detected
        if prediction['is_defect'] and prediction['confidence'] > self.alert_threshold:
            self.send_alert(board_id, prediction)
        
        return prediction
    
    def log_to_mes(self, board_id, prediction):
        data = {
            "board_id": board_id,
            "timestamp": datetime.now().isoformat(),
            "defect_type": prediction['defect_type'],
            "confidence": prediction['confidence'],
            "status": "REJECT" if prediction['is_defect'] else "PASS"
        }
        
        requests.post(f"{self.mes_endpoint}/quality_log", json=data)
    
    def send_alert(self, board_id, prediction):
        # Send to operator dashboard
        alert = {
            "board_id": board_id,
            "defect_type": prediction['defect_type'],
            "confidence": prediction['confidence'],
            "action_required": "MANUAL_INSPECTION"
        }
        
        requests.post(f"{self.mes_endpoint}/alerts", json=alert)
```

---

## 4. Monitoring and Maintenance

### Performance Monitoring

```python
# monitoring.py
import logging
from datetime import datetime
import json

class ModelMonitor:
    def __init__(self, log_file='model_performance.log'):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
    
    def log_prediction(self, board_id, prediction, inference_time):
        log_entry = {
            "board_id": board_id,
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "inference_time_ms": inference_time,
        }
        logging.info(json.dumps(log_entry))
    
    def check_drift(self):
        # Implement data drift detection
        # Compare recent predictions with training distribution
        pass
    
    def generate_report(self):
        # Generate daily/weekly performance reports
        pass
```

### Retraining Pipeline

```python
# retraining.py
from pathlib import Path
import shutil
from datetime import datetime

class RetrainingPipeline:
    def __init__(self, edge_cases_dir='edge_cases'):
        self.edge_cases_dir = Path(edge_cases_dir)
        self.edge_cases_dir.mkdir(exist_ok=True)
    
    def collect_edge_case(self, image_path, prediction, operator_label):
        # Save misclassified or low-confidence images
        if prediction['confidence'] < 0.85 or prediction['defect_type'] != operator_label:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dest = self.edge_cases_dir / f"{timestamp}_{operator_label}.jpg"
            shutil.copy(image_path, dest)
    
    def trigger_retraining(self, min_samples=100):
        # Check if enough edge cases collected
        edge_cases = list(self.edge_cases_dir.glob('*.jpg'))
        
        if len(edge_cases) >= min_samples:
            print(f"Triggering retraining with {len(edge_cases)} new samples")
            # Call training pipeline
            # Update model
            # Deploy new version
```

---

## 5. Security Considerations

### Model Protection

```python
# Encrypt model for deployment
import tensorflow as tf
from cryptography.fernet import Fernet

def encrypt_model(model_path, key):
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    cipher = Fernet(key)
    encrypted_data = cipher.encrypt(model_data)
    
    with open(f"{model_path}.encrypted", 'wb') as f:
        f.write(encrypted_data)

def decrypt_and_load_model(encrypted_path, key):
    cipher = Fernet(key)
    
    with open(encrypted_path, 'rb') as f:
        encrypted_data = f.read()
    
    decrypted_data = cipher.decrypt(encrypted_data)
    
    # Load model from bytes
    # ... implementation
```

### API Security

```python
# Add authentication to API
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

@app.post("/predict")
async def predict(file: UploadFile = File(...), 
                 token: str = Depends(verify_token)):
    # ... prediction code
```

---

## 6. Cost Analysis

### Edge Deployment (Per Unit)

| Component | Cost |
|-----------|------|
| Raspberry Pi 4 (4GB) | $55 |
| Camera Module | $25 |
| Case + Cooling | $15 |
| Power Supply | $10 |
| MicroSD Card | $10 |
| **Total** | **$115** |

**Operating Costs:**
- Power: ~$0.50/month (24/7 operation)
- Maintenance: ~$50/year

### Cloud Deployment (Monthly)

| Service | Cost |
|---------|------|
| AWS EC2 (t3.medium) | $30 |
| Storage (100GB) | $10 |
| Data Transfer | $20 |
| **Total** | **$60/month** |

### ROI Calculation

**Assumptions:**
- Factory produces 10,000 boards/month
- Current defect escape rate: 15% (1,500 boards)
- Cost per escaped defect: $500
- Current monthly loss: $750,000

**With AI System:**
- Defect escape rate: 2% (200 boards)
- Monthly loss: $100,000
- **Monthly savings: $650,000**
- **System cost: $115 (one-time) + $0.50/month**
- **ROI: Immediate (payback in hours)**

---

## Conclusion

The PCB defect detection system can be deployed in multiple configurations depending on requirements:

- **Edge Deployment**: Best for factory floor, low latency, no cloud dependency
- **Cloud API**: Best for centralized processing, multiple locations
- **Hybrid**: Edge for real-time + Cloud for analytics and retraining

Choose based on your specific needs for latency, cost, scalability, and infrastructure.
