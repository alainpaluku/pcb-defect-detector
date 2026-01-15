"""Data ingestion and conversion for PCB Defect Detection with YOLOv8."""

import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from src.config import Config, IMAGE_EXTENSIONS
from src.utils import find_image_file, get_logger

logger = get_logger(__name__)


@dataclass
class ImageItem:
    """Représente une image avec ses métadonnées."""
    image_path: Path
    annotation_path: Optional[Path] = None
    class_name: Optional[str] = None
    source_type: str = "unknown"  # "xml" ou "class_folder"


class VOCConverter:
    """Convertisseur d'annotations VOC XML vers YOLO."""
    
    @staticmethod
    def convert(
        xml_path: Path,
        img_width: int,
        img_height: int
    ) -> List[str]:
        """Convertit une annotation VOC XML en format YOLO.
        
        Args:
            xml_path: Chemin vers le fichier XML
            img_width: Largeur de l'image
            img_height: Hauteur de l'image
        
        Returns:
            Liste de lignes au format YOLO
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in Config.CLASS_MAP:
                logger.warning(f"Classe inconnue ignorée: {class_name}")
                continue
            
            class_id = Config.CLASS_MAP[class_name]
            bbox = obj.find("bndbox")
            
            # Extraction et normalisation des coordonnées
            xmin = VOCConverter._clamp(float(bbox.find("xmin").text), 0, img_width)
            ymin = VOCConverter._clamp(float(bbox.find("ymin").text), 0, img_height)
            xmax = VOCConverter._clamp(float(bbox.find("xmax").text), 0, img_width)
            ymax = VOCConverter._clamp(float(bbox.find("ymax").text), 0, img_height)
            
            # Conversion en format YOLO (centre normalisé + dimensions)
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            if width > 0 and height > 0:
                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
        
        return yolo_lines
    
    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Limite une valeur entre min et max."""
        return max(min_val, min(value, max_val))


class DataIngestion:
    """Gère le chargement et la conversion des données au format YOLO."""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else Config.get_data_path()
        self.yolo_path = Config.get_yolo_dataset_path()
        self.images_dir: Optional[Path] = None
        self.annot_dir: Optional[Path] = None
        self.all_images: List[ImageItem] = []
    
    def find_data_structure(self) -> bool:
        """Recherche les images et annotations dans le dataset."""
        logger.info(f"Recherche dans: {self.data_path}")
        
        self._find_annotations_dir()
        self._find_images_dir()
        
        logger.info(f"Répertoire images: {self.images_dir}")
        logger.info(f"Répertoire annotations: {self.annot_dir}")
        
        return self.images_dir is not None
    
    def _find_annotations_dir(self) -> None:
        """Recherche le répertoire des annotations."""
        # Recherche récursive des dossiers Annotations
        candidates = [
            self.data_path / "Annotations",
            self.data_path / "PCB_DATASET" / "Annotations",
        ]
        
        # Recherche récursive si non trouvé
        if not any(c.exists() for c in candidates):
            for annot_dir in self.data_path.rglob("Annotations"):
                if annot_dir.is_dir() and list(annot_dir.glob("*.xml")):
                    candidates.insert(0, annot_dir)
                    break
        
        for candidate in candidates:
            if candidate.exists() and list(candidate.glob("*.xml")):
                self.annot_dir = candidate
                return
    
    def _find_images_dir(self) -> None:
        """Recherche le répertoire des images."""
        search_dirs = [
            self.data_path,
            self.data_path / "PCB_DATASET",
            self.data_path / "PCB_DATASET" / "images",
            self.data_path / "images",
        ]
        
        # Ajouter les sous-dossiers potentiels
        if self.data_path.exists():
            for subdir in self.data_path.iterdir():
                if subdir.is_dir() and subdir not in search_dirs:
                    search_dirs.append(subdir)
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            # Vérifie les dossiers de classes (insensible à la casse)
            has_class_folders = self._has_class_folders(search_dir)
            
            if has_class_folders:
                self.images_dir = search_dir
                return
            
            # Vérifie les images directes
            if any(search_dir.glob(f"*{ext}") for ext in IMAGE_EXTENSIONS):
                self.images_dir = search_dir
                return
        
        # Recherche récursive en dernier recours
        if self.images_dir is None:
            self._find_images_recursive()
    
    def _has_class_folders(self, search_dir: Path) -> bool:
        """Vérifie si le dossier contient des sous-dossiers de classes."""
        if not search_dir.exists():
            return False
        
        class_names_lower = {name.lower() for name in Config.CLASS_NAMES}
        
        for subdir in search_dir.iterdir():
            if subdir.is_dir():
                # Vérification insensible à la casse
                subdir_name = subdir.name.lower().replace("-", "_").replace(" ", "_")
                if subdir_name in class_names_lower:
                    return True
        return False
    
    def _find_images_recursive(self) -> None:
        """Recherche récursive des images dans le dataset."""
        logger.info("Recherche récursive des images...")
        
        # Chercher des dossiers qui ressemblent à des classes de défauts
        class_names_lower = {name.lower() for name in Config.CLASS_NAMES}
        
        for folder in self.data_path.rglob("*"):
            if folder.is_dir():
                folder_name = folder.name.lower().replace("-", "_").replace(" ", "_")
                if folder_name in class_names_lower:
                    # Trouvé un dossier de classe, le parent est le dossier images
                    self.images_dir = folder.parent
                    logger.info(f"Trouvé dossier de classes: {self.images_dir}")
                    return
    
    def collect_images(self) -> List[ImageItem]:
        """Collecte toutes les images avec leurs annotations."""
        self.all_images = []
        seen_images = set()
        
        # Collecte via annotations XML
        if self.annot_dir and self.annot_dir.exists():
            self._collect_from_xml(seen_images)
        
        # Collecte via dossiers de classes
        if self.images_dir:
            self._collect_from_class_folders(seen_images)
        
        logger.info(f"Total images collectées: {len(self.all_images)}")
        return self.all_images
    
    def _collect_from_xml(self, seen_images: set) -> None:
        """Collecte les images depuis les annotations XML."""
        xml_files = list(self.annot_dir.glob("*.xml"))
        logger.info(f"Trouvé {len(xml_files)} annotations XML")
        
        search_dirs = [d for d in [self.images_dir, self.data_path] if d]
        class_subdirs = list(Config.CLASS_MAP.keys())
        
        for xml_file in xml_files:
            img_path = find_image_file(
                xml_file.stem,
                search_dirs,
                subdirs=[""] + class_subdirs
            )
            
            if img_path:
                self.all_images.append(ImageItem(
                    image_path=img_path,
                    annotation_path=xml_file,
                    source_type="xml"
                ))
                seen_images.add(img_path)
    
    def _collect_from_class_folders(self, seen_images: set) -> None:
        """Collecte les images depuis les dossiers de classes."""
        if not self.images_dir or not self.images_dir.exists():
            return
        
        # Créer un mapping insensible à la casse
        class_name_map = {}
        for cls_name in Config.CLASS_NAMES:
            class_name_map[cls_name.lower()] = cls_name
            class_name_map[cls_name.lower().replace("_", "-")] = cls_name
            class_name_map[cls_name.lower().replace("_", " ")] = cls_name
        
        # Parcourir les sous-dossiers
        for subdir in self.images_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            # Normaliser le nom du dossier
            normalized_name = subdir.name.lower().replace("-", "_").replace(" ", "_")
            
            # Trouver le nom de classe correspondant
            cls_name = None
            if normalized_name in class_name_map:
                cls_name = class_name_map[normalized_name]
            else:
                # Essayer de matcher partiellement
                for key, value in class_name_map.items():
                    if key in normalized_name or normalized_name in key:
                        cls_name = value
                        break
            
            if cls_name is None:
                continue
            
            logger.info(f"Collecte depuis dossier: {subdir.name} -> classe: {cls_name}")
            
            for ext in IMAGE_EXTENSIONS:
                for img_path in subdir.glob(f"*{ext}"):
                    if img_path not in seen_images:
                        self.all_images.append(ImageItem(
                            image_path=img_path,
                            class_name=cls_name,
                            source_type="class_folder"
                        ))
                        seen_images.add(img_path)
    
    def create_yolo_dataset(self) -> Tuple[int, int]:
        """Crée le dataset au format YOLO."""
        logger.info("Création de la structure YOLO...")
        
        # Création des répertoires
        for split in ["train", "val"]:
            (self.yolo_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Mélange et split
        random.seed(Config.data.random_seed)
        shuffled = self.all_images.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1 - Config.data.val_split))
        train_images = shuffled[:split_idx]
        val_images = shuffled[split_idx:]
        
        logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}")
        
        # Traitement des images
        train_count = self._process_split(train_images, "train")
        val_count = self._process_split(val_images, "val")
        
        logger.info(f"Traité - Train: {train_count}, Val: {val_count}")
        
        # Création du fichier YAML
        self._create_yaml_config()
        
        return train_count, val_count
    
    def _process_split(self, image_list: List[ImageItem], split: str) -> int:
        """Traite les images pour un split (train/val)."""
        count = 0
        
        for item in image_list:
            # Vérifier que le fichier existe
            if not item.image_path.exists():
                logger.warning(f"Fichier non trouvé: {item.image_path}")
                continue
            
            try:
                img = Image.open(item.image_path)
                img.verify()  # Vérifier l'intégrité de l'image
                img = Image.open(item.image_path)  # Réouvrir après verify()
                img_width, img_height = img.size
            except Exception as e:
                logger.warning(f"Image invalide {item.image_path}: {e}")
                continue
            
            # Copie de l'image
            dst_img = self.yolo_path / "images" / split / item.image_path.name
            shutil.copy(item.image_path, dst_img)
            
            # Création du label
            label_path = self.yolo_path / "labels" / split / f"{item.image_path.stem}.txt"
            
            if item.source_type == "xml" and item.annotation_path:
                yolo_lines = VOCConverter.convert(
                    item.annotation_path, img_width, img_height
                )
                label_path.write_text("\n".join(yolo_lines))
            else:
                # Dossier de classe - utilise l'image entière comme bbox
                cls_id = Config.CLASS_MAP.get(item.class_name, 0)
                label_path.write_text(f"{cls_id} 0.5 0.5 1.0 1.0")
            
            count += 1
        
        return count
    
    def _create_yaml_config(self) -> Path:
        """Crée la configuration YAML du dataset YOLO."""
        yaml_content = f"""path: {self.yolo_path}
train: images/train
val: images/val

names:
{chr(10).join(f'  {i}: {name}' for i, name in enumerate(Config.CLASS_NAMES))}

nc: {Config.NUM_CLASSES}
"""
        yaml_path = self.yolo_path / "dataset.yaml"
        yaml_path.write_text(yaml_content)
        
        logger.info(f"Config dataset sauvegardée: {yaml_path}")
        return yaml_path
    
    def get_yaml_path(self) -> Path:
        """Retourne le chemin vers la config YAML du dataset."""
        return self.yolo_path / "dataset.yaml"
    
    def get_stats(self) -> Dict[str, int]:
        """Retourne les statistiques du dataset."""
        return {
            "total_images": len(self.all_images),
            "with_xml": sum(1 for item in self.all_images if item.source_type == "xml"),
            "from_folders": sum(1 for item in self.all_images if item.source_type == "class_folder"),
        }
