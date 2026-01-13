# Guide d'utilisation sur Kaggle

## Étapes détaillées

### 1. Créer un nouveau Notebook

- Allez sur [kaggle.com](https://www.kaggle.com)
- Cliquez sur **"Create"** → **"New Notebook"**

### 2. Ajouter le Dataset (IMPORTANT !)

**C'est l'étape la plus importante - sans cela, le code ne fonctionnera pas !**

1. Dans votre notebook, cherchez le bouton **"+ Add Data"** (en haut à droite)
2. Dans la barre de recherche, tapez : `pcb-defects` ou `akhatova/pcb-defects`
3. Cliquez sur le dataset **"PCB Defects"** par **akhatova**
4. Cliquez sur **"Add"**
5. Attendez que le dataset soit monté (vous verrez `/kaggle/input/pcb-defects` dans le panneau de droite)

### 3. Activer le GPU

1. Cliquez sur les **3 points** en haut à droite
2. Sélectionnez **"Accelerator"**
3. Choisissez **"GPU T4 x2"** ou **"GPU P100"**
4. Cliquez **"Save"**

### 4. Exécuter le code

**Option A : Télécharger et exécuter**
```python
!wget https://raw.githubusercontent.com/alainpaluku/pcb-defect-detector/main/kaggle_script.py
%run kaggle_script.py
```

**Option B : Copier-coller**
1. Ouvrez [kaggle_script.py](https://github.com/alainpaluku/pcb-defect-detector/blob/main/kaggle_script.py)
2. Copiez tout le contenu
3. Collez dans une cellule de votre notebook Kaggle
4. Exécutez la cellule

### 5. Résultats

Après l'exécution (environ 30-45 minutes), vous trouverez :

- **Modèle entraîné** : `/kaggle/working/pcb_model.keras`
- **Graphiques** :
  - `/kaggle/working/results/confusion_matrix.png`
  - `/kaggle/working/results/training_curves.png`
- **Rapport** : `/kaggle/working/results/report.txt`

### Visualiser les résultats

Ajoutez cette cellule à la fin pour afficher les graphiques :

```python
from IPython.display import Image, display

# Afficher la matrice de confusion
display(Image('/kaggle/working/results/confusion_matrix.png'))

# Afficher les courbes d'entraînement
display(Image('/kaggle/working/results/training_curves.png'))
```

## Dépannage

### Erreur : "FileNotFoundError: /kaggle/input/pcb-defects"

**Cause** : Le dataset n'a pas été ajouté au notebook.

**Solution** : Suivez l'étape 2 ci-dessus pour ajouter le dataset.

### Erreur : "ResourceExhaustedError" (Out of Memory)

**Cause** : Batch size trop grand ou GPU non activé.

**Solution** : 
1. Activez le GPU (étape 3)
2. Ou réduisez le batch size dans le script :
```python
BATCH_SIZE = 16  # au lieu de 32
```

### Le training est trop lent

**Cause** : GPU non activé.

**Solution** : Vérifiez que le GPU est activé (étape 3). Vous devriez voir "Using 1 GPU(s)" dans les logs.

### Modifier les hyperparamètres

Éditez ces lignes au début du script :

```python
EPOCHS = 25              # Nombre d'époques phase 1
FINE_TUNE_EPOCHS = 15    # Nombre d'époques phase 2
BATCH_SIZE = 32          # Taille des batchs
LEARNING_RATE = 1e-3     # Learning rate phase 1
FINE_TUNE_LR = 1e-5      # Learning rate phase 2
```

## Temps d'exécution estimé

- **Avec GPU T4 x2** : ~30-40 minutes
- **Avec GPU P100** : ~25-35 minutes
- **Sans GPU (CPU)** : ~4-6 heures (non recommandé)

## Support

Si vous rencontrez des problèmes :
1. Vérifiez que le dataset est bien ajouté
2. Vérifiez que le GPU est activé
3. Ouvrez une issue sur [GitHub](https://github.com/alainpaluku/pcb-defect-detector/issues)
