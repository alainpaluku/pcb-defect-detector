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


def run_training(epochs: int = 50) -> dict:
    """Exécute l'entraînement et retourne les métriques."""
    from src.trainer import TrainingManager
    from src.utils import format_metrics, print_section_header
    
    trainer = TrainingManager()
    metrics = trainer.run_pipeline(epochs=epochs)
    
    print_section_header("RÉSULTATS FINAUX")
    print(format_metrics(metrics))
    
    return metrics


def main() -> None:
    """Point d'entrée principal."""
    setup_environment()
    install_dependencies()
    run_training(epochs=50)


if __name__ == "__main__":
    main()
