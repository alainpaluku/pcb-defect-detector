"""Data ingestion and conversion for PCB Defect Detection with YOLOv8."""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from src.config import Config


class DataIngestion:
    """Handles data loading and conversion to YOLO format."""
    
    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else Config.get_data_path()
        self.yolo_path = Config.get_yolo_dataset_path()
        self.images_dir = None
        self.annot_dir = None
        self.all_images = []
        
    def find_data_structure(self):
        """Find images and annotations in the dataset."""
        print(f"Searching in: {self.data_path}")
        
        # Search for Annotations folder
        for subdir in ["", "PCB_DATASET", "Annotations"]:
            candidate = self.data_path / subdir / "Annotations" if subdir != "Annotations" else self.data_path / subdir
            if candidate.exists() and list(candidate.glob("*.xml")):
                self.annot_dir = candidate
                break
        
        # Search for images
        search_dirs = [
            self.data_path,
            self.data_path / "PCB_DATASET",
            self.data_path / "PCB_DATASET" / "images",
            self.data_path / "images",
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            # Check for class folders
            class_folders = [c for c in Config.CLASS_MAP.keys() if (search_dir / c).exists()]
            if class_folders:
                self.images_dir = search_dir
                break
            
            # Check for direct images
            if list(search_dir.glob("*.jpg")) or list(search_dir.glob("*.JPG")):
                self.images_dir = search_dir
                break
        
        print(f"Images directory: {self.images_dir}")
        print(f"Annotations directory: {self.annot_dir}")
        
        return self.images_dir is not None
    
    def collect_images(self):
        """Collect all images with their annotations."""
        self.all_images = []
        
        if self.annot_dir and self.annot_dir.exists():
            # Use XML annotations
            xml_files = list(self.annot_dir.glob("*.xml"))
            print(f"Found {len(xml_files)} XML annotations")
            
            for xml_file in xml_files:
                img_path = self._find_image_for_xml(xml_file)
                if img_path:
                    self.all_images.append({
                        "image": img_path,
                        "annotation": xml_file,
                        "type": "xml"
                    })
        
        # Also collect from class folders
        if self.images_dir:
            for cls_name in Config.CLASS_MAP.keys():
                cls_dir = self.images_dir / cls_name
                if cls_dir.exists():
                    for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG"]:
                        for img_path in cls_dir.glob(ext):
                            # Check if already added via XML
                            if not any(item["image"] == img_path for item in self.all_images):
                                self.all_images.append({
                                    "image": img_path,
                                    "class": cls_name,
                                    "type": "class_folder"
                                })
        
        print(f"Total images collected: {len(self.all_images)}")
        return self.all_images
    
    def _find_image_for_xml(self, xml_path):
        """Find the corresponding image for an XML annotation."""
        img_name = xml_path.stem
        
        search_dirs = [self.images_dir] if self.images_dir else []
        search_dirs.extend([
            self.data_path,
            self.data_path / "PCB_DATASET",
            self.data_path / "PCB_DATASET" / "images",
        ])
        
        for search_dir in search_dirs:
            if not search_dir or not search_dir.exists():
                continue
            
            for ext in [".jpg", ".JPG", ".png", ".PNG"]:
                # Direct path
                candidate = search_dir / (img_name + ext)
                if candidate.exists():
                    return candidate
                
                # In class subfolders
                for cls_name in Config.CLASS_MAP.keys():
                    candidate = search_dir / cls_name / (img_name + ext)
                    if candidate.exists():
                        return candidate
        
        return None
    
    def convert_voc_to_yolo(self, xml_path, img_width, img_height):
        """Convert VOC XML annotation to YOLO format."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in Config.CLASS_MAP:
                continue
            
            class_id = Config.CLASS_MAP[class_name]
            bbox = obj.find("bndbox")
            
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            
            # Clamp values
            xmin = max(0, min(xmin, img_width))
            xmax = max(0, min(xmax, img_width))
            ymin = max(0, min(ymin, img_height))
            ymax = max(0, min(ymax, img_height))
            
            # Convert to YOLO format (normalized center_x, center_y, width, height)
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            if width > 0 and height > 0:
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return yolo_lines
    
    def create_yolo_dataset(self):
        """Create YOLO formatted dataset."""
        print("\nCreating YOLO dataset structure...")
        
        # Create directories
        for split in ["train", "val"]:
            (self.yolo_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Shuffle and split
        random.seed(Config.RANDOM_SEED)
        random.shuffle(self.all_images)
        
        split_idx = int(len(self.all_images) * (1 - Config.VAL_SPLIT))
        train_images = self.all_images[:split_idx]
        val_images = self.all_images[split_idx:]
        
        print(f"Train: {len(train_images)}, Val: {len(val_images)}")
        
        # Process images
        train_count = self._process_split(train_images, "train")
        val_count = self._process_split(val_images, "val")
        
        print(f"Processed - Train: {train_count}, Val: {val_count}")
        
        # Create YAML config
        self._create_yaml_config()
        
        return train_count, val_count
    
    def _process_split(self, image_list, split):
        """Process images for a split (train/val)."""
        count = 0
        
        for item in image_list:
            img_path = item["image"]
            
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                continue
            
            # Copy image
            dst_img = self.yolo_path / "images" / split / img_path.name
            shutil.copy(img_path, dst_img)
            
            # Create label
            label_path = self.yolo_path / "labels" / split / (img_path.stem + ".txt")
            
            if item["type"] == "xml":
                yolo_lines = self.convert_voc_to_yolo(item["annotation"], img_width, img_height)
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_lines))
            else:
                # Class folder - use full image as bounding box
                cls_name = item["class"]
                cls_id = Config.CLASS_MAP.get(cls_name, 0)
                with open(label_path, "w") as f:
                    f.write(f"{cls_id} 0.5 0.5 1.0 1.0")
            
            count += 1
        
        return count
    
    def _create_yaml_config(self):
        """Create YOLO dataset YAML configuration."""
        yaml_content = f"""path: {self.yolo_path}
train: images/train
val: images/val

names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper

nc: {Config.NUM_CLASSES}
"""
        yaml_path = self.yolo_path / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        
        print(f"Dataset config saved: {yaml_path}")
        return yaml_path
    
    def get_yaml_path(self):
        """Return path to dataset YAML config."""
        return self.yolo_path / "dataset.yaml"
    
    def get_stats(self):
        """Return dataset statistics."""
        stats = {
            "total_images": len(self.all_images),
            "with_xml": sum(1 for item in self.all_images if item["type"] == "xml"),
            "from_folders": sum(1 for item in self.all_images if item["type"] == "class_folder"),
        }
        return stats
