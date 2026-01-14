# PCB Defect Detection

Deep Learning pour l'Inspection Optique Automatisée des Circuits Imprimés avec **MobileNetV2**.

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/datasets/akhatova/pcb-defects)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Classes de Défauts

| Défaut | Description |
|--------|-------------|
| `missing_hole` | Trou de perçage manquant |
| `mouse_bite` | Bord rongé/irrégulier |
| `open_circuit` | Circuit ouvert/interrompu |
| `short` | Court-circuit |
| `spur` | Excroissance de cuivre |
| `spurious_copper` | Cuivre parasite |

## Démarrage Rapide

### Kaggle (Recommandé)

**Prérequis :**
1. Ajouter le dataset `akhatova/pcb-defects` via **"+ Add Input"**
2. Activer le **GPU** dans les paramètres du notebook

**Une seule cellule :**
```python
!rm -rf /kaggle/working/pcb-defect-detector
!git clone https://github.com/alainpaluku/pcb-defect-detector.git
%cd /kaggle/working/pcb-defect-detector
!python run_kaggle.py
```

### Local

```bash
git clone https://github.com/alainpaluku/pcb-defect-detector.git
cd pcb-defect-detector
pip install -r requirements.txt
python main.py --epochs 30
```

**Options CLI :**
```bash
python main.py --help
python main.py --epochs 50 --batch-size 16 --lr 0.0001
python main.py --no-fine-tune        # Sans fine-tuning
python main.py --download            # Télécharger dataset Kaggle
```

## Architecture

```
Input (224×224×3)
       ↓
MobileNetV2 (ImageNet weights)
       ↓
GlobalAveragePooling2D + BatchNorm
       ↓
Dense(128) → Dropout(0.5) → ReLU
Dense(64)  → Dropout(0.4) → ReLU
       ↓
Softmax (6 classes)
```

**Pipeline :**
| Phase | Epochs | Learning Rate |
|-------|--------|---------------|
| Transfer Learning | 30 | 1e-4 |
| Fine-tuning | 15 | 1e-5 |

## Résultats

| Métrique | Valeur |
|----------|--------|
| Accuracy | ~85% |
| Precision | ~87% |
| Recall | ~83% |
| F1 Score | ~85% |

## Utilisation du Modèle

```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('pcb_model.keras')
CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

img = tf.keras.preprocessing.image.load_img('pcb_image.jpg', target_size=(224, 224))
img_array = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img) / 255.0, axis=0)

prediction = model.predict(img_array)
print(f"Défaut: {CLASSES[np.argmax(prediction)]} ({np.max(prediction)*100:.1f}%)")
```

## Structure du Projet

```
pcb-defect-detector/
├── src/
│   ├── config.py           # Configuration
│   ├── data_ingestion.py   # Chargement données
│   ├── model.py            # Architecture MobileNetV2
│   └── trainer.py          # Pipeline entraînement
├── main.py                 # Point d'entrée local
├── run_kaggle.py           # Script Kaggle
└── requirements.txt
```

## Fichiers Générés

| Fichier | Usage |
|---------|-------|
| `pcb_model.keras` | Modèle Keras |
| `pcb_model.h5` | Format legacy |
| `pcb_model.onnx` | Format ONNX |
| `pcb_model.tflite` | Format TFLite (mobile) |
| `training_history.png` | Courbes entraînement |
| `confusion_matrix.png` | Matrice de confusion |

## Dataset

[PCB Defects - Akhatova](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- 1386 images, 6 classes

## Auteur

**Alain Paluku** - [@alainpaluku](https://github.com/alainpaluku)

## Licence

MIT License
