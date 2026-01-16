#!/usr/bin/env python3
"""
PCB Defect Detection with YOLOv8 - Kaggle Runner

Usage on Kaggle:
    !pip install ultralytics -q
    !rm -rf /kaggle/working/pcb-defect-detector
    !git clone https://github.com/alainpaluku/pcb-defect-detector.git
    %cd /kaggle/working/pcb-defect-detector
    !python run_kaggle.py
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_environment() -> None:
    """Configure l'environnement d'exécution."""
    # Change de répertoire si nécessaire
    target_dir = Path("/kaggle/working/pcb-defect-detector")
    if Path.cwd().name != "pcb-defect-detector" and target_dir.exists():
        os.chdir(target_dir)
    
    # Ajoute le répertoire courant au path
    sys.path.insert(0, str(Path.cwd()))


def install_dependencies() -> None:
    """Installe les dépendances requises."""
    print("Installation de ultralytics...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "ultralytics", "-q"],
        check=True
    )


def debug_dataset_structure() -> None:
    """Affiche la structure du dataset pour le debug."""
    print("\n" + "=" * 60)
    print("DEBUG: Structure du dataset Kaggle")
    print("=" * 60)
    
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        print("Pas dans l'environnement Kaggle")
        return
    
    # Lister les datasets disponibles
    print(f"\nDatasets dans {kaggle_input}:")
    for item in kaggle_input.iterdir():
        print(f"  📁 {item.name}")
        if item.is_dir():
            # Afficher les sous-dossiers (2 niveaux)
            for sub in item.iterdir():
                prefix = "    📁" if sub.is_dir() else "    📄"
                print(f"{prefix} {sub.name}")
                if sub.is_dir():
                    # Compter les fichiers
                    files = list(sub.iterdir())
                    if len(files) <= 10:
                        for f in files:
                            prefix2 = "      📁" if f.is_dir() else "      📄"
                            print(f"{prefix2} {f.name}")
                    else:
                        print(f"      ... ({len(files)} éléments)")
    print("=" * 60 + "\n")


def run_training(epochs: int = 100) -> dict:
    """Exécute l'entraînement et retourne les métriques."""
    from src.trainer import TrainingManager
    from src.utils import print_section_header
    
    print("\n" + "=" * 60)
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT PCB DEFECT DETECTION")
    print("=" * 60)
    print(f"   Époques: {epochs}")
    print(f"   GPU: {'✅ Disponible' if is_gpu_available() else '❌ Non disponible'}")
    print("=" * 60 + "\n")
    
    trainer = TrainingManager()
    metrics = trainer.run_pipeline(epochs=epochs)
    
    return metrics


def is_gpu_available() -> bool:
    """Vérifie si un GPU est disponible."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def main() -> None:
    """Point d'entrée principal."""
    setup_environment()
    install_dependencies()
    
    # Debug: afficher la structure du dataset
    debug_dataset_structure()
    
    # Entraînement - 100 époques avec early stopping
    run_training(epochs=100)


if __name__ == "__main__":
    main()
