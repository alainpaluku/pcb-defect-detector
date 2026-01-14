"""Utility functions for PCB Defect Detection."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import IMAGE_EXTENSIONS

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré."""
    return logging.getLogger(name)


def count_images(
    directory: Path,
    recursive: bool = False
) -> int:
    """Compte les images dans un répertoire."""
    pattern = "**/*" if recursive else "*"
    return sum(
        1 for f in directory.glob(pattern)
        if f.suffix.lower() in [ext.lower() for ext in IMAGE_EXTENSIONS]
    )


def get_all_images(
    directory: Path,
    recursive: bool = False
) -> List[Path]:
    """Récupère tous les fichiers images d'un répertoire."""
    pattern = "**/*" if recursive else "*"
    return [
        f for f in directory.glob(pattern)
        if f.suffix.lower() in [ext.lower() for ext in IMAGE_EXTENSIONS]
    ]


def find_image_file(
    base_name: str,
    search_dirs: List[Path],
    subdirs: Optional[List[str]] = None
) -> Optional[Path]:
    """Recherche un fichier image par son nom de base."""
    subdirs = subdirs or [""]
    
    for search_dir in search_dirs:
        if not search_dir or not search_dir.exists():
            continue
        
        for subdir in subdirs:
            check_dir = search_dir / subdir if subdir else search_dir
            if not check_dir.exists():
                continue
                
            for ext in IMAGE_EXTENSIONS:
                candidate = check_dir / f"{base_name}{ext}"
                if candidate.exists():
                    return candidate
    
    return None


def format_bytes(size_bytes: int) -> str:
    """Formate les bytes en chaîne lisible."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def print_section_header(title: str, width: int = 60) -> None:
    """Affiche un en-tête de section formaté."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_detection_results(detections: List[Dict]) -> None:
    """Affiche les résultats de détection de manière formatée."""
    if not detections:
        print("Aucun défaut détecté.")
        return
    
    print(f"\n{len(detections)} défaut(s) détecté(s):")
    for i, det in enumerate(detections, 1):
        bbox = det["bbox"]
        print(f"  {i}. {det['class_name']}: {det['confidence']:.2%}")
        print(f"     Box: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) - ({bbox['x2']:.0f}, {bbox['y2']:.0f})")


def format_metrics(metrics: Dict[str, float]) -> str:
    """Formate les métriques pour l'affichage."""
    lines = [
        f"mAP@50:     {metrics.get('mAP50', 0):.4f}",
        f"mAP@50-95:  {metrics.get('mAP50-95', 0):.4f}",
        f"Precision:  {metrics.get('precision', 0):.4f}",
        f"Recall:     {metrics.get('recall', 0):.4f}",
    ]
    return "\n".join(lines)
