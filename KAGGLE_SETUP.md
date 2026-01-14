# 🚀 Guide d'Utilisation sur Kaggle

## ⚠️ Prérequis OBLIGATOIRES

Avant de lancer l'entraînement, tu DOIS :

### 1. Ajouter le Dataset

**Le dataset n'est PAS inclus dans ce repo** (trop volumineux). Tu dois l'ajouter manuellement :

1. Ouvre ton notebook Kaggle
2. Dans le panneau de droite, clique sur **"+ Add Input"**
3. Cherche : `akhatova/pcb-defects`
4. Clique sur **"Add"** pour l'ajouter à ton notebook
5. Le dataset apparaîtra dans `/kaggle/input/pcb-defects/`

### 2. Activer le GPU

1. Dans le menu de droite, section **"Accelerator"**
2. Sélectionne **"GPU T4 x2"** ou **"GPU P100"**
3. Clique sur **"Save"**

## 📝 Utilisation

### Option A : Une seule cellule (Recommandé)

```python
# Cellule 1 : Clone et lance tout
!rm -rf /kaggle/working/pcb-defect-detector
!git clone https://github.com/alainpaluku/pcb-defect-detector.git
%cd /kaggle/working/pcb-defect-detector
!python run_kaggle.py
```

### Option B : Étape par étape

```python
# Cellule 1 : Clone le repo
!git clone https://github.com/alainpaluku/pcb-defect-detector.git
%cd /kaggle/working/pcb-defect-detector

# Cellule 2 : Installe les dépendances
!pip install -q tf2onnx onnx onnxruntime

# Cellule 3 : Lance l'entraînement
!python run_kaggle.py
```

## 🔍 Vérifier que le Dataset est Chargé

Avant de lancer l'entraînement, vérifie que le dataset est présent :

```python
import os
from pathlib import Path

# Vérifie la structure
kaggle_input = Path("/kaggle/input")
print("📂 Datasets disponibles:")
for item in kaggle_input.iterdir():
    print(f"   - {item.name}")

# Vérifie le dataset PCB
pcb_path = Path("/kaggle/input/pcb-defects")
if pcb_path.exists():
    print(f"\n✅ Dataset PCB trouvé!")
    print(f"   Structure:")
    for item in list(pcb_path.iterdir())[:10]:
        print(f"      - {item.name}")
else:
    print("\n❌ Dataset PCB NON TROUVÉ!")
    print("   👉 Ajoute-le via '+ Add Input' → 'akhatova/pcb-defects'")
```

## 📊 Résultats Attendus

Après ~45 minutes d'entraînement (avec GPU), tu devrais obtenir :

- **Accuracy** : ~85%
- **Precision** : ~87%
- **Recall** : ~83%
- **F1 Score** : ~85%

## 📁 Fichiers Générés

Dans `/kaggle/working/` :

- `pcb_model.keras` - Modèle principal (14 MB)
- `pcb_model.h5` - Format legacy
- `pcb_model.onnx` - Format cross-platform
- `pcb_model.tflite` - Format mobile/edge
- `training_history.png` - Courbes d'entraînement
- `confusion_matrix.png` - Matrice de confusion
- `roc_curves.png` - Courbes ROC
- `classification_report.txt` - Rapport détaillé

## 🐛 Dépannage

### Erreur : "No class folders found"

**Cause** : Le dataset n'est pas ajouté ou mal placé

**Solution** :
1. Vérifie que tu as bien ajouté `akhatova/pcb-defects` dans les inputs
2. Redémarre le kernel : **Kernel** → **Restart & Run All**

### Erreur : "Out of Memory"

**Cause** : Batch size trop grand ou pas de GPU

**Solution** :
1. Active le GPU (voir section 2 ci-dessus)
2. Ou réduis le batch size dans `src/config.py` :
   ```python
   BATCH_SIZE = 16  # Au lieu de 32
   ```

### Accuracy reste à 0%

**Cause** : Aucune image n'a été chargée

**Solution** :
1. Exécute le code de vérification ci-dessus
2. Assure-toi que le dataset est bien dans `/kaggle/input/pcb-defects/`
3. Vérifie que les dossiers de classes existent (Missing_hole, Mouse_bite, etc.)

## 💡 Conseils

- **Temps d'entraînement** : ~30-45 min avec GPU, ~3-4h sans GPU
- **Sauvegarde** : Les modèles sont sauvegardés dans `/kaggle/working/`
- **Téléchargement** : Clique sur les fichiers dans l'explorateur pour les télécharger
- **Versions** : Kaggle sauvegarde automatiquement les versions de ton notebook

## 📞 Support

Si tu rencontres des problèmes :
1. Vérifie que le dataset est bien ajouté
2. Vérifie que le GPU est activé
3. Consulte les logs d'erreur complets
4. Ouvre une issue sur GitHub : https://github.com/alainpaluku/pcb-defect-detector/issues
