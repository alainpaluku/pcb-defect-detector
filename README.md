<p align="center">
  <img src="https://img.shields.io/badge/🔬-PCB_Defect_Detection-blue?style=for-the-badge" alt="PCB Defect Detection"/>
</p>

<p align="center">
  <b>Deep Learning pour l'Inspection Optique Automatisée des Circuits Imprimés</b>
</p>

<p align="center">
  <a href="https://www.kaggle.com/datasets/akhatova/pcb-defects"><img src="https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle"/></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://tensorflow.org"><img src="https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/></a>
</p>

---

## 🎯 Objectif

Système de détection automatique de défauts sur circuits imprimés (PCB) utilisant le transfer learning avec **MobileNetV2**. Conçu pour l'inspection qualité en environnement industriel.

## 🏷️ Classes de Défauts

| Défaut | Description |
|--------|-------------|
| 🕳️ `missing_hole` | Trou de perçage manquant |
| 🐭 `mouse_bite` | Bord rongé/irrégulier |
| ⚡ `open_circuit` | Circuit ouvert/interrompu |
| 🔗 `short` | Court-circuit |
| 📍 `spur` | Excroissance de cuivre |
| 🟤 `spurious_copper` | Cuivre parasite |

## 🚀 Démarrage Rapide

### Option 1 : Kaggle (Recommandé)

```python
# Une seule cellule pour tout lancer
!rm -rf /kaggle/working/pcb-defect-detector
!git clone https://github.com/alainpaluku/pcb-defect-detector.git
%cd /kaggle/working/pcb-defect-detector
!python run_kaggle.py
```

> ⚠️ **Prérequis OBLIGATOIRES** : 
> 1. Ajouter le dataset `akhatova/pcb-defects` via **"+ Add Input"**
> 2. Activer le **GPU** dans les paramètres du notebook
> 
> 📖 **[Guide complet Kaggle](KAGGLE_SETUP.md)** - Instructions détaillées et dépannage

### Option 2 : Local

```bash
git clone https://github.com/alainpaluku/pcb-defect-detector.git
cd pcb-defect-detector
pip install -r requirements.txt
python main.py --epochs 30 --fine-tune
```

## 🏗️ Architecture CNN

### Réseau de Neurones Convolutif

Ce projet utilise **MobileNetV2**, un CNN (Convolutional Neural Network) optimisé pour la vision par ordinateur.

**Pourquoi MobileNetV2 ?**
- 🧠 Pré-entraîné sur **ImageNet** (1.4M images, 1000 classes)
- ⚡ Léger : ~3.4M paramètres → rapide sur GPU/mobile
- 🎯 **Depthwise Separable Convolutions** : 8-9x moins de calculs qu'une convolution classique
- 🔗 **Inverted Residuals** : Skip connections pour un meilleur gradient

**Fonctionnement des convolutions :**
```
Image (224×224×3)
    ↓
┌─────────────────────────────────────┐
│  CONVOLUTIONS (53 couches)          │
│  • Détection de bords               │
│  • Extraction de textures           │
│  • Reconnaissance de formes         │
│  • Features de haut niveau          │
└─────────────────────────────────────┘
    ↓
Features Map (7×7×1280)
    ↓
Classification (6 défauts)
```

**Architecture complète :**

```
┌─────────────────────────────────────────┐
│           Input (224×224×3)             │
├─────────────────────────────────────────┤
│     MobileNetV2 (ImageNet weights)      │
│  [Conv2D → BatchNorm → ReLU6] × 53      │
│     Depthwise Separable Convolutions    │
│         [Fine-tuned: 30 layers]         │
├─────────────────────────────────────────┤
│       GlobalAveragePooling2D            │
│         BatchNormalization              │
├─────────────────────────────────────────┤
│    Dense(128) → Dropout(0.5) → ReLU     │
│    Dense(64)  → Dropout(0.4) → ReLU     │
├─────────────────────────────────────────┤
│         Softmax (6 classes)             │
└─────────────────────────────────────────┘
```

## 📊 Pipeline d'Entraînement

| Phase | Epochs | Learning Rate | Description |
|-------|--------|---------------|-------------|
| Transfer Learning | 30 | 1e-4 | Base MobileNetV2 gelée |
| Fine-tuning | 15 | 1e-5 | 30 dernières couches dégelées |

## � Résultats

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | ~85% |
| **Precision** | ~87% |
| **Recall** | ~83% |
| **F1 Score** | ~85% |
| **Temps d'inférence** | ~30ms |
| **Taille du modèle** | ~14MB |


## 🔮 Utilisation du Modèle

### Charger et prédire

```python
import tensorflow as tf
import numpy as np

# Charger le modèle entraîné
model = tf.keras.models.load_model('pcb_model.keras')

# Classes de défauts
CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

# Charger une image PCB
img = tf.keras.preprocessing.image.load_img('pcb_image.jpg', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array / 255.0, axis=0)  # Normaliser

# Prédiction
prediction = model.predict(img_array)
predicted_class = CLASSES[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Défaut détecté: {predicted_class}")
print(f"Confiance: {confidence:.1f}%")
```

### Prédiction sur plusieurs images

```python
from pathlib import Path

def predict_batch(image_folder, model):
    """Prédire sur un dossier d'images."""
    CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    results = []
    
    for img_path in Path(image_folder).glob('*.jpg'):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img) / 255.0, 0)
        
        pred = model.predict(img_array, verbose=0)
        results.append({
            'image': img_path.name,
            'defect': CLASSES[np.argmax(pred)],
            'confidence': f"{np.max(pred)*100:.1f}%"
        })
    
    return results

# Utilisation
model = tf.keras.models.load_model('pcb_model.keras')
results = predict_batch('mes_images_pcb/', model)
for r in results:
    print(f"{r['image']}: {r['defect']} ({r['confidence']})")
```

### Classe d'inspection pour production

```python
class PCBInspector:
    """Classe pour l'inspection de PCB en production."""
    
    def __init__(self, model_path='pcb_model.keras'):
        self.model = tf.keras.models.load_model(model_path)
        self.classes = ['missing_hole', 'mouse_bite', 'open_circuit', 
                        'short', 'spur', 'spurious_copper']
    
    def inspect(self, image_path):
        """Inspecte une image et retourne le résultat."""
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img) / 255.0, 0)
        
        prediction = self.model.predict(img_array, verbose=0)[0]
        
        return {
            'status': 'DEFECT' if np.max(prediction) > 0.5 else 'UNCERTAIN',
            'defect_type': self.classes[np.argmax(prediction)],
            'confidence': float(np.max(prediction)),
            'all_scores': {c: float(p) for c, p in zip(self.classes, prediction)}
        }

# Utilisation
inspector = PCBInspector('pcb_model.keras')
result = inspector.inspect('circuit_board.jpg')
print(f"Status: {result['status']}")
print(f"Défaut: {result['defect_type']} ({result['confidence']:.1%})")
```

### Conversion TFLite pour mobile/edge

```python
# Convertir en TFLite pour déploiement embarqué
model = tf.keras.models.load_model('pcb_model.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('pcb_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Modèle TFLite: {len(tflite_model) / 1024 / 1024:.1f} MB")
```

## 📁 Structure du Projet

```
pcb-defect-detector/
├── 📂 src/
│   ├── config.py           # Configuration centralisée
│   ├── data_ingestion.py   # Chargement & augmentation
│   ├── model.py            # Architecture MobileNetV2
│   └── trainer.py          # Pipeline d'entraînement
├── 📂 notebooks/
│   └── pcb_defect_detection.ipynb
├── 📂 tests/
│   └── test_model.py
├── 🐍 main.py              # Point d'entrée CLI
├── 🚀 run_kaggle.py        # Script Kaggle one-click
└── 📋 requirements.txt
```

## 💾 Fichiers Générés

| Fichier | Usage |
|---------|-------|
| `pcb_model.keras` | Modèle Keras (recommandé) |
| `pcb_model.h5` | Format legacy |
| `pcb_model.onnx` | Format ONNX (cross-platform) |
| `pcb_model.tflite` | Format TFLite (mobile/edge) |
| `training_history.png` | Courbes d'entraînement |
| `confusion_matrix.png` | Matrice de confusion |
| `roc_curves.png` | Courbes ROC par classe |

## 🔄 Formats d'Export

### ONNX (Open Neural Network Exchange)

Le modèle est automatiquement exporté en ONNX pour une compatibilité cross-platform :

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Charger le modèle ONNX
session = ort.InferenceSession('pcb_model.onnx')

# Préparer l'image
img = Image.open('pcb_image.jpg').resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Inférence
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
prediction = session.run([output_name], {input_name: img_array})[0]

CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
print(f"Défaut: {CLASSES[np.argmax(prediction)]}")
```

**Avantages ONNX :**
- 🌐 Compatible avec PyTorch, TensorFlow, scikit-learn
- ⚡ Optimisé pour l'inférence (ONNX Runtime)
- 🖥️ Fonctionne sur Windows, Linux, macOS, mobile
- 🔧 Intégrable dans des apps C++, C#, Java, JavaScript

## 🔧 Configuration

Paramètres clés dans `src/config.py` :

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
ROTATION_RANGE = 30
DROPOUT = 0.5
FINE_TUNE_EPOCHS = 15
FINE_TUNE_LAYERS = 30
```

## 📚 Dataset

**[PCB Defects - Akhatova](https://www.kaggle.com/datasets/akhatova/pcb-defects)**

- 🖼️ 1386 images (693 originales + 693 rotations)
- 🏷️ 6 classes de défauts
- 📐 ~115 images par classe

## 👤 Auteur

**Alain Paluku** - [@alainpaluku](https://github.com/alainpaluku)

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE)

---

<p align="center">
  ⭐ Star ce repo si tu le trouves utile !
</p>
