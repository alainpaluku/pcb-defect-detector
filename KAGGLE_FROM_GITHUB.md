# Comment Utiliser ce Projet GitHub sur Kaggle

## 🎯 Guide Complet: GitHub → Kaggle

Ce guide explique comment utiliser le code de ce repository GitHub directement dans un notebook Kaggle.

---

## 📋 Table des Matières

1. [Méthode 1: Cloner depuis GitHub (Recommandé)](#méthode-1-cloner-depuis-github-recommandé)
2. [Méthode 2: Télécharger et Uploader le Notebook](#méthode-2-télécharger-et-uploader-le-notebook)
3. [Méthode 3: Copier-Coller le Code](#méthode-3-copier-coller-le-code)
4. [Méthode 4: Créer un Dataset Kaggle](#méthode-4-créer-un-dataset-kaggle)

---

## Méthode 1: Cloner depuis GitHub (Recommandé) ⭐

### Étape 1: Créer un Nouveau Notebook Kaggle

1. Allez sur [Kaggle.com](https://www.kaggle.com)
2. Cliquez sur **"Code"** → **"New Notebook"**
3. Donnez un titre: "PCB Defect Detection"

### Étape 2: Ajouter le Dataset

1. Dans le panneau de droite, cliquez sur **"+ Add Data"**
2. Recherchez: **"akhatova/pcb-defects"**
3. Cliquez sur **"Add"**

### Étape 3: Activer GPU

1. Cliquez sur les **3 points** en haut à droite
2. **Settings** → **Accelerator** → **GPU T4 x2**
3. Cliquez **"Save"**

### Étape 4: Cloner le Repository GitHub

Dans la première cellule du notebook:

```python
# Cellule 1: Cloner le repository GitHub
!git clone https://github.com/VOTRE_USERNAME/pcb-defect-detector.git
%cd pcb-defect-detector
!ls -la
```

### Étape 5: Installer les Dépendances

```python
# Cellule 2: Installer les packages
!pip install -q -r requirements.txt
```

### Étape 6: Vérifier l'Installation

```python
# Cellule 3: Vérifier que tout fonctionne
import sys
sys.path.insert(0, '/kaggle/working/pcb-defect-detector')

from src.config import Config
from src.trainer import TrainingManager

print("✓ Tous les modules importés avec succès!")
print(f"✓ Dataset path: {Config.get_data_path()}")
print(f"✓ Output path: {Config.get_output_path()}")
```

### Étape 7: Lancer l'Entraînement

```python
# Cellule 4: Entraîner le modèle
trainer = TrainingManager()
metrics = trainer.run_pipeline()

print(f"\n✓ Entraînement terminé!")
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

### ✅ Avantages de cette Méthode
- ✅ Code toujours à jour depuis GitHub
- ✅ Structure complète du projet
- ✅ Facile à mettre à jour (`git pull`)
- ✅ Tous les fichiers disponibles

---

## Méthode 2: Télécharger et Uploader le Notebook

### Étape 1: Télécharger le Notebook depuis GitHub

1. Allez sur votre repository GitHub
2. Naviguez vers: `notebooks/pcb_defect_detection.ipynb`
3. Cliquez sur **"Raw"**
4. Faites **Ctrl+S** pour sauvegarder le fichier

### Étape 2: Uploader sur Kaggle

1. Allez sur [Kaggle.com](https://www.kaggle.com)
2. Cliquez sur **"Code"** → **"New Notebook"**
3. Cliquez sur **"File"** → **"Upload Notebook"**
4. Sélectionnez le fichier `.ipynb` téléchargé

### Étape 3: Ajouter le Dataset

1. **"+ Add Data"** → Recherchez **"akhatova/pcb-defects"**
2. Cliquez **"Add"**

### Étape 4: Activer GPU

1. **Settings** → **Accelerator** → **GPU**

### Étape 5: Modifier les Imports

Le notebook doit importer le code. Ajoutez cette cellule au début:

```python
# Cellule 1: Cloner le code source depuis GitHub
!git clone https://github.com/VOTRE_USERNAME/pcb-defect-detector.git
import sys
sys.path.insert(0, '/kaggle/working/pcb-defect-detector')

# Vérifier
from src.trainer import TrainingManager
print("✓ Code importé avec succès!")
```

### Étape 6: Run All

Cliquez sur **"Run All"** et attendez 30-60 minutes.

---

## Méthode 3: Copier-Coller le Code

Si vous ne voulez pas cloner le repository, vous pouvez copier le code directement.

### Étape 1: Créer un Nouveau Notebook

1. Kaggle → **"New Notebook"**
2. Ajouter dataset: **"akhatova/pcb-defects"**
3. Activer GPU

### Étape 2: Copier les Modules

Créez une cellule pour chaque module Python:

**Cellule 1: Config**
```python
# src/config.py
# Copiez tout le contenu de src/config.py depuis GitHub
```

**Cellule 2: Data Ingestion**
```python
# src/data_ingestion.py
# Copiez tout le contenu de src/data_ingestion.py depuis GitHub
```

**Cellule 3: Model**
```python
# src/model.py
# Copiez tout le contenu de src/model.py depuis GitHub
```

**Cellule 4: Trainer**
```python
# src/trainer.py
# Copiez tout le contenu de src/trainer.py depuis GitHub
```

### Étape 3: Lancer l'Entraînement

```python
# Cellule 5: Entraîner
trainer = TrainingManager()
metrics = trainer.run_pipeline()
```

### ⚠️ Inconvénients
- ❌ Beaucoup de copier-coller
- ❌ Difficile à maintenir
- ❌ Risque d'erreurs

---

## Méthode 4: Créer un Dataset Kaggle

Cette méthode permet de réutiliser le code dans plusieurs notebooks.

### Étape 1: Préparer le Code

Sur votre machine locale:

```bash
# Créer un zip du dossier src
cd pcb-defect-detector
zip -r pcb-detector-src.zip src/
```

### Étape 2: Créer un Dataset Kaggle

1. Allez sur [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Cliquez **"New Dataset"**
3. Uploadez `pcb-detector-src.zip`
4. Titre: "PCB Defect Detector Source Code"
5. Cliquez **"Create"**

### Étape 3: Utiliser dans un Notebook

```python
# Cellule 1: Importer le code depuis votre dataset
import sys
import zipfile

# Extraire le code
with zipfile.ZipFile('/kaggle/input/pcb-detector-src/pcb-detector-src.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# Ajouter au path
sys.path.insert(0, '/kaggle/working')

# Importer
from src.trainer import TrainingManager

# Entraîner
trainer = TrainingManager()
metrics = trainer.run_pipeline()
```

### ✅ Avantages
- ✅ Réutilisable dans plusieurs notebooks
- ✅ Pas besoin de cloner à chaque fois
- ✅ Versionné sur Kaggle

---

## 🎯 Méthode Recommandée: Résumé

### Pour Débutants
**Méthode 2** (Upload Notebook) - Simple et direct

### Pour Utilisateurs Avancés
**Méthode 1** (Clone GitHub) - Toujours à jour, professionnel

### Pour Réutilisation
**Méthode 4** (Dataset Kaggle) - Partageable, versionné

---

## 📝 Template Complet pour Kaggle

Voici un notebook complet prêt à l'emploi:

```python
# ============================================================
# PCB DEFECT DETECTION - KAGGLE NOTEBOOK
# Repository: https://github.com/VOTRE_USERNAME/pcb-defect-detector
# ============================================================

# CELLULE 1: Setup
print("📦 Installation et configuration...")

# Cloner le repository
!git clone https://github.com/VOTRE_USERNAME/pcb-defect-detector.git
%cd pcb-defect-detector

# Installer les dépendances
!pip install -q -r requirements.txt

print("✓ Installation terminée!")

# CELLULE 2: Vérification
print("🔍 Vérification de l'environnement...")

import sys
sys.path.insert(0, '/kaggle/working/pcb-defect-detector')

import tensorflow as tf
from src.config import Config
from src.trainer import TrainingManager

print(f"✓ TensorFlow: {tf.__version__}")
print(f"✓ GPU: {tf.config.list_physical_devices('GPU')}")
print(f"✓ Dataset: {Config.get_data_path()}")
print(f"✓ Output: {Config.get_output_path()}")

# CELLULE 3: Entraînement
print("🚀 Démarrage de l'entraînement...")

trainer = TrainingManager()
metrics = trainer.run_pipeline()

# CELLULE 4: Résultats
print("\n" + "="*60)
print("📊 RÉSULTATS FINAUX")
print("="*60)
print(f"Accuracy:  {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall:    {metrics['recall']:.2%}")
print(f"F1 Score:  {metrics['f1_score']:.2%}")
print("="*60)

# CELLULE 5: Télécharger les Résultats
from IPython.display import FileLink

print("\n📥 Télécharger les fichiers:")
print(FileLink('/kaggle/working/best_model.h5'))
print(FileLink('/kaggle/working/training_history.png'))
print(FileLink('/kaggle/working/confusion_matrix.png'))
```

---

## 🔧 Résolution de Problèmes

### Erreur: "Repository not found"
```python
# Vérifiez l'URL du repository
!git clone https://github.com/USERNAME/REPO.git
# Remplacez USERNAME et REPO par les vôtres
```

### Erreur: "Module not found"
```python
# Ajoutez le chemin au sys.path
import sys
sys.path.insert(0, '/kaggle/working/pcb-defect-detector')
```

### Erreur: "Dataset not found"
```python
# Vérifiez que le dataset est ajouté
from pathlib import Path
data_path = Path("/kaggle/input/pcb-defects")
print(f"Dataset exists: {data_path.exists()}")
print(f"Contents: {list(data_path.iterdir())}")
```

### Erreur: "Out of memory"
```python
# Réduisez le batch size
from src.config import Config
Config.BATCH_SIZE = 16  # ou 8
```

---

## 📚 Ressources Supplémentaires

### Documentation
- [README.md](README.md) - Vue d'ensemble
- [KAGGLE_SETUP.md](KAGGLE_SETUP.md) - Guide Kaggle détaillé
- [QUICK_START.md](QUICK_START.md) - Démarrage rapide

### Liens Utiles
- **Dataset**: [PCB Defects](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- **Kaggle Docs**: [Using Git in Notebooks](https://www.kaggle.com/docs/notebooks#using-git)
- **GitHub**: [Votre Repository]

---

## ✅ Checklist Avant de Commencer

- [ ] Compte Kaggle créé
- [ ] Dataset "akhatova/pcb-defects" ajouté
- [ ] GPU activé dans les settings
- [ ] Repository GitHub accessible (public ou avec token)
- [ ] Notebook créé ou uploadé

---

## 🎉 Exemple de Notebook Public

Pour voir un exemple fonctionnel, consultez:
- **Notebook Kaggle**: [Lien vers votre notebook public]
- **Repository GitHub**: [Lien vers votre repo]

---

## 💡 Conseils Pro

1. **Utilisez Git Clone** - Plus propre et professionnel
2. **Activez GPU** - 10-20x plus rapide
3. **Sauvegardez Régulièrement** - Kaggle auto-save, mais soyez prudent
4. **Commentez Votre Code** - Facilitez la compréhension
5. **Partagez Votre Notebook** - Contribuez à la communauté

---

## 🆘 Besoin d'Aide?

- **Issues GitHub**: Ouvrez une issue sur le repository
- **Kaggle Discussion**: Postez dans les discussions
- **Documentation**: Consultez les fichiers .md du projet

---

**Bon entraînement! 🚀**

*Ce guide est maintenu à jour. Dernière mise à jour: Janvier 2026*
