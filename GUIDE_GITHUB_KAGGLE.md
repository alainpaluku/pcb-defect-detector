# 🚀 Guide Complet: GitHub → Kaggle

## Guide Visuel Étape par Étape

---

## 📤 PARTIE 1: METTRE LE PROJET SUR GITHUB

### Étape 1: Préparer le Repository Local

```bash
# Vous êtes déjà dans le dossier pcb-defect-detector
cd pcb-defect-detector

# Vérifier le statut Git
git status

# Ajouter tous les fichiers
git add -A

# Faire le commit
git commit -m "Complete PCB Defect Detection System v1.1.0 - Optimized"
```

### Étape 2: Créer le Repository sur GitHub

1. **Aller sur GitHub.com**
   - Connectez-vous à votre compte GitHub
   - URL: https://github.com

2. **Créer un Nouveau Repository**
   - Cliquez sur le bouton **"+"** en haut à droite
   - Sélectionnez **"New repository"**

3. **Configurer le Repository**
   ```
   Repository name: pcb-defect-detector
   Description: Industrial PCB Defect Detection using Deep Learning (MobileNetV2)
   
   ☑️ Public (recommandé pour Kaggle)
   ☐ Add a README file (on en a déjà un)
   ☐ Add .gitignore (on en a déjà un)
   ☐ Choose a license (on a déjà MIT)
   ```

4. **Cliquer sur "Create repository"**

### Étape 3: Connecter et Pousser le Code

GitHub vous donnera des commandes. Utilisez celles-ci:

```bash
# Ajouter le remote (remplacez VOTRE_USERNAME par votre nom d'utilisateur GitHub)
git remote add origin https://github.com/VOTRE_USERNAME/pcb-defect-detector.git

# Vérifier le remote
git remote -v

# Pousser le code
git branch -M main
git push -u origin main
```

**Si vous avez une erreur d'authentification:**
```bash
# Utiliser un Personal Access Token
# 1. Aller sur GitHub → Settings → Developer settings → Personal access tokens
# 2. Generate new token (classic)
# 3. Cocher: repo, workflow
# 4. Copier le token
# 5. Utiliser le token comme mot de passe lors du push
```

### Étape 4: Vérifier sur GitHub

1. Rafraîchir la page GitHub
2. Vous devriez voir tous vos fichiers
3. Le README.md s'affiche automatiquement

**✅ Votre projet est maintenant sur GitHub!**

---

## 📥 PARTIE 2: UTILISER LE CODE GITHUB SUR KAGGLE

### Méthode 1: Cloner Directement (RECOMMANDÉ) ⭐

#### Étape 1: Créer un Notebook Kaggle

1. **Aller sur Kaggle.com**
   - URL: https://www.kaggle.com
   - Connectez-vous

2. **Créer un Nouveau Notebook**
   - Cliquez sur **"Code"** dans le menu
   - Cliquez sur **"New Notebook"**
   - Donnez un titre: **"PCB Defect Detection"**

#### Étape 2: Configurer le Notebook

1. **Activer Internet**
   - Cliquez sur les **3 points** en haut à droite
   - **Settings** → **Internet** → **ON** ✅

2. **Activer GPU**
   - **Settings** → **Accelerator** → **GPU T4 x2** ✅

3. **Ajouter le Dataset**
   - Dans le panneau de droite: **"+ Add Data"**
   - Rechercher: **"akhatova/pcb-defects"**
   - Cliquer sur **"Add"** ✅

#### Étape 3: Cloner le Repository GitHub

**Cellule 1: Cloner le Code**
```python
# Cloner votre repository GitHub
# IMPORTANT: Remplacez VOTRE_USERNAME par votre nom d'utilisateur GitHub
!git clone https://github.com/VOTRE_USERNAME/pcb-defect-detector.git

# Aller dans le dossier
%cd pcb-defect-detector

# Vérifier les fichiers
!ls -la
```

**Sortie attendue:**
```
Cloning into 'pcb-defect-detector'...
✓ README.md
✓ src/
✓ notebooks/
✓ requirements.txt
...
```

#### Étape 4: Installer les Dépendances

**Cellule 2: Installation**
```python
# Installer les packages requis
!pip install -q -r requirements.txt

print("✓ Installation terminée!")
```

#### Étape 5: Vérifier l'Environnement

**Cellule 3: Vérification**
```python
import sys
import os

# Ajouter le code au path Python
sys.path.insert(0, '/kaggle/working/pcb-defect-detector')

# Importer les modules
import tensorflow as tf
from src.config import Config
from src.trainer import TrainingManager

# Afficher les informations
print("="*60)
print("VÉRIFICATION DE L'ENVIRONNEMENT")
print("="*60)
print(f"✓ TensorFlow: {tf.__version__}")
print(f"✓ GPU: {len(tf.config.list_physical_devices('GPU'))} device(s)")
print(f"✓ Dataset: {Config.get_data_path()}")
print(f"✓ Output: {Config.get_output_path()}")
print("="*60)

# Vérifier le dataset
data_path = Config.get_data_path()
if data_path.exists():
    classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    print(f"\n✓ Dataset trouvé!")
    print(f"  Classes: {len(classes)}")
    print(f"  {classes}")
else:
    print(f"\n✗ Dataset non trouvé!")
    print("  → Ajoutez 'akhatova/pcb-defects' dans 'Add Data'")
```

#### Étape 6: Lancer l'Entraînement

**Cellule 4: Training**
```python
# Initialiser le training manager
trainer = TrainingManager()

# Lancer le pipeline complet
# Cela va prendre 25-30 minutes
metrics = trainer.run_pipeline()
```

**Ce qui va se passer:**
```
==============================================================
PCB DEFECT DETECTION SYSTEM
==============================================================

PHASE 1: DATA INGESTION
------------------------------------------------------------
Dataset Analysis:
  Total Images: 1386
  Number of Classes: 6
  ...

PHASE 2: MODEL ARCHITECTURE
------------------------------------------------------------
Model: MobileNetV2
Total Parameters: 3,538,984
...

PHASE 3: MODEL TRAINING
------------------------------------------------------------
Epoch 1/50
34/34 [==============================] - 45s
...

PHASE 4: MODEL EVALUATION
------------------------------------------------------------
Validation Accuracy: 96.2%
...

✓ Training completed successfully!
```

#### Étape 7: Voir les Résultats

**Cellule 5: Résultats**
```python
# Afficher les métriques finales
print("\n" + "="*60)
print("📊 RÉSULTATS FINAUX")
print("="*60)
print(f"Accuracy:  {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall:    {metrics['recall']:.2%}")
print(f"F1 Score:  {metrics['f1_score']:.2%}")
print("="*60)

# Vérifier les objectifs
print("\n🎯 Objectifs Industriels:")
print(f"  Accuracy > 95%:   {'✅' if metrics['accuracy'] > 0.95 else '❌'}")
print(f"  Precision > 93%:  {'✅' if metrics['precision'] > 0.93 else '❌'}")
print(f"  Recall > 90%:     {'✅' if metrics['recall'] > 0.90 else '❌'}")
```

#### Étape 8: Visualiser les Graphiques

**Cellule 6: Visualisations**
```python
from IPython.display import Image, display

output_path = Config.get_output_path()

# Historique d'entraînement
print("📈 Historique d'Entraînement:")
display(Image(filename=str(output_path / 'training_history.png')))

# Matrice de confusion
print("\n🎯 Matrice de Confusion:")
display(Image(filename=str(output_path / 'confusion_matrix.png')))
```

#### Étape 9: Télécharger le Modèle

**Cellule 7: Téléchargement**
```python
from IPython.display import FileLink

print("📥 Fichiers disponibles:")
print("\n1. Modèle entraîné:")
display(FileLink('/kaggle/working/best_model.h5'))

print("\n2. Graphiques:")
display(FileLink('/kaggle/working/training_history.png'))
display(FileLink('/kaggle/working/confusion_matrix.png'))

print("\n3. Rapport:")
display(FileLink('/kaggle/working/classification_report.txt'))
```

---

## 🎯 RÉSUMÉ VISUEL

### Workflow Complet

```
┌─────────────────────────────────────────────────────────┐
│  1. GITHUB                                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │ git add -A                                       │  │
│  │ git commit -m "message"                          │  │
│  │ git push origin main                             │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  2. KAGGLE NOTEBOOK                                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │ !git clone https://github.com/USER/repo.git     │  │
│  │ %cd pcb-defect-detector                          │  │
│  │ !pip install -q -r requirements.txt              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  3. TRAINING                                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ from src.trainer import TrainingManager          │  │
│  │ trainer = TrainingManager()                      │  │
│  │ metrics = trainer.run_pipeline()                 │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  4. RÉSULTATS                                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ ✓ Accuracy: 96.2%                                │  │
│  │ ✓ Model: best_model.h5                           │  │
│  │ ✓ Graphs: training_history.png                   │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 TEMPLATE COMPLET POUR KAGGLE

Voici un notebook complet prêt à copier-coller:

```python
# ============================================================
# PCB DEFECT DETECTION - KAGGLE NOTEBOOK
# Repository: https://github.com/VOTRE_USERNAME/pcb-defect-detector
# ============================================================

# ============================================================
# CELLULE 1: CLONER LE REPOSITORY
# ============================================================
print("📦 Clonage du repository GitHub...")

# IMPORTANT: Remplacez VOTRE_USERNAME par votre nom d'utilisateur GitHub
!git clone https://github.com/VOTRE_USERNAME/pcb-defect-detector.git
%cd pcb-defect-detector

print("✓ Repository cloné!")

# ============================================================
# CELLULE 2: INSTALLER LES DÉPENDANCES
# ============================================================
print("📦 Installation des dépendances...")

!pip install -q -r requirements.txt

print("✓ Installation terminée!")

# ============================================================
# CELLULE 3: VÉRIFIER L'ENVIRONNEMENT
# ============================================================
print("🔍 Vérification de l'environnement...")

import sys
sys.path.insert(0, '/kaggle/working/pcb-defect-detector')

import tensorflow as tf
from src.config import Config
from src.trainer import TrainingManager

print("="*60)
print("ENVIRONNEMENT")
print("="*60)
print(f"✓ TensorFlow: {tf.__version__}")
print(f"✓ GPU: {len(tf.config.list_physical_devices('GPU'))} device(s)")
print(f"✓ Dataset: {Config.get_data_path()}")
print("="*60)

# ============================================================
# CELLULE 4: ENTRAÎNER LE MODÈLE
# ============================================================
print("🚀 Démarrage de l'entraînement...")
print("⏱️  Temps estimé: 25-30 minutes")

trainer = TrainingManager()
metrics = trainer.run_pipeline()

# ============================================================
# CELLULE 5: AFFICHER LES RÉSULTATS
# ============================================================
print("\n" + "="*60)
print("📊 RÉSULTATS FINAUX")
print("="*60)
print(f"Accuracy:  {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall:    {metrics['recall']:.2%}")
print(f"F1 Score:  {metrics['f1_score']:.2%}")
print("="*60)

print("\n🎯 Objectifs:")
print(f"  Accuracy > 95%:   {'✅ ATTEINT' if metrics['accuracy'] > 0.95 else '❌'}")
print(f"  Precision > 93%:  {'✅ ATTEINT' if metrics['precision'] > 0.93 else '❌'}")
print(f"  Recall > 90%:     {'✅ ATTEINT' if metrics['recall'] > 0.90 else '❌'}")

# ============================================================
# CELLULE 6: VISUALISER LES GRAPHIQUES
# ============================================================
from IPython.display import Image, display

output_path = Config.get_output_path()

print("📈 Historique d'Entraînement:")
display(Image(filename=str(output_path / 'training_history.png')))

print("\n🎯 Matrice de Confusion:")
display(Image(filename=str(output_path / 'confusion_matrix.png')))

# ============================================================
# CELLULE 7: TÉLÉCHARGER LES FICHIERS
# ============================================================
from IPython.display import FileLink

print("📥 Fichiers disponibles au téléchargement:\n")

files = [
    ('best_model.h5', 'Meilleur modèle'),
    ('training_history.png', 'Graphiques d\'entraînement'),
    ('confusion_matrix.png', 'Matrice de confusion'),
    ('classification_report.txt', 'Rapport détaillé')
]

for filename, description in files:
    filepath = output_path / filename
    if filepath.exists():
        print(f"✓ {description}:")
        display(FileLink(str(filepath)))
    else:
        print(f"✗ {description}: Non trouvé")

print("\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
```

---

## 🔧 DÉPANNAGE

### Problème 1: "Repository not found"
```python
# Vérifiez l'URL
# Assurez-vous que le repository est PUBLIC
# Format correct: https://github.com/USERNAME/pcb-defect-detector.git
```

### Problème 2: "Module not found"
```python
# Ajoutez le chemin au sys.path
import sys
sys.path.insert(0, '/kaggle/working/pcb-defect-detector')
```

### Problème 3: "Dataset not found"
```python
# Vérifiez que le dataset est ajouté
# 1. Panneau de droite → "+ Add Data"
# 2. Rechercher: "akhatova/pcb-defects"
# 3. Cliquer "Add"
```

### Problème 4: "Out of memory"
```python
# Réduisez le batch size
from src.config import Config
Config.BATCH_SIZE = 16  # ou 8
```

### Problème 5: "Internet not enabled"
```python
# Activez Internet dans les settings
# Settings → Internet → ON
```

---

## ✅ CHECKLIST AVANT DE COMMENCER

### Sur GitHub
- [ ] Repository créé
- [ ] Code poussé (git push)
- [ ] Repository PUBLIC
- [ ] README visible

### Sur Kaggle
- [ ] Compte créé
- [ ] Notebook créé
- [ ] Internet activé
- [ ] GPU activé
- [ ] Dataset ajouté

---

## 🎓 CONSEILS PRO

1. **Utilisez GPU**: 10-20x plus rapide que CPU
2. **Sauvegardez Régulièrement**: Kaggle auto-save, mais soyez prudent
3. **Commentez Votre Code**: Facilitez la compréhension
4. **Partagez Votre Notebook**: Contribuez à la communauté
5. **Vérifiez les Logs**: Surveillez l'entraînement

---

## 📞 BESOIN D'AIDE?

### Documentation
- `README.md` - Vue d'ensemble
- `QUICK_START.md` - Démarrage rapide
- `KAGGLE_SETUP.md` - Guide Kaggle détaillé

### Support
- GitHub Issues: Pour bugs
- Kaggle Discussion: Pour questions
- Documentation: Pour guides

---

## 🎉 FÉLICITATIONS!

Vous savez maintenant:
- ✅ Mettre votre code sur GitHub
- ✅ Cloner depuis GitHub sur Kaggle
- ✅ Entraîner le modèle
- ✅ Télécharger les résultats

**Bon entraînement! 🚀**
