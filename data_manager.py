"""
Kaggle Data Manager module for PCB Defect Detector.

Handles authentication, dataset downloading, and directory parsing.
"""

import os
import json
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional
from getpass import getpass

from config import DataConfig, Environment, detect_environment


class KaggleDataManager:
    """Manages Kaggle authentication and dataset operations."""
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_path: Optional[Path] = None
        self.class_names: List[str] = []
        self.environment = detect_environment()
        
    def authenticate(self, api_token: Optional[str] = None) -> None:
        """Authenticate with Kaggle API."""
        # Skip auth on Kaggle (already authenticated)
        if self.environment == Environment.KAGGLE:
            self.logger.info("Running on Kaggle - authentication not required.")
            return
        
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if kaggle_json.exists():
            self.logger.info("Kaggle credentials found.")
            return
            
        token = api_token or os.environ.get('KAGGLE_API_TOKEN')
        
        if not token:
            self.logger.info("Enter Kaggle API token (JSON format):")
            token = getpass("Token: ")
        
        try:
            credentials = json.loads(token)
            if 'username' not in credentials or 'key' not in credentials:
                raise ValueError("Token must contain 'username' and 'key'.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(token)
        kaggle_json.chmod(0o600)
        
        os.environ['KAGGLE_USERNAME'] = credentials['username']
        os.environ['KAGGLE_KEY'] = credentials['key']
        self.logger.info("Kaggle credentials saved.")

    def download_dataset(self, force: bool = False) -> Path:
        """Download and extract dataset from Kaggle."""
        # On Kaggle, dataset is pre-mounted
        if self.environment == Environment.KAGGLE:
            self.dataset_path = self._find_kaggle_dataset()
            self.logger.info(f"Using Kaggle dataset at {self.dataset_path}")
            return self.dataset_path
        
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        existing = self._find_dataset_root()
        if existing and not force:
            self.dataset_path = existing
            self.logger.info(f"Dataset exists at {self.dataset_path}")
            return self.dataset_path
        
        self.logger.info(f"Downloading {self.config.dataset_name}...")
        
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            self.config.dataset_name,
            path=str(self.config.data_dir),
            unzip=False
        )
        
        # Extract zip
        zip_files = list(self.config.data_dir.glob("*.zip"))
        if not zip_files:
            raise RuntimeError("No zip file found after download.")
        
        zip_path = zip_files[0]
        self.logger.info(f"Extracting {zip_path}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.config.data_dir)
        
        zip_path.unlink()
        
        self.dataset_path = self._find_dataset_root()
        if not self.dataset_path:
            raise RuntimeError("Could not find dataset after extraction.")
        
        self.logger.info(f"Dataset ready at {self.dataset_path}")
        return self.dataset_path
    
    def _find_kaggle_dataset(self) -> Path:
        """Find dataset in Kaggle input directory."""
        kaggle_input = Path("/kaggle/input/pcb-defects")
        
        # Check common structures
        candidates = [
            kaggle_input / "PCB_DATASET",
            kaggle_input / "pcb_defects", 
            kaggle_input
        ]
        
        for path in candidates:
            if path.exists() and self._has_image_subdirs(path):
                return path
        
        # Search subdirectories
        for item in kaggle_input.iterdir():
            if item.is_dir() and self._has_image_subdirs(item):
                return item
        
        raise FileNotFoundError(f"Dataset not found in {kaggle_input}")
    
    def _find_dataset_root(self) -> Optional[Path]:
        """Find dataset root directory."""
        if not self.config.data_dir.exists():
            return None
        
        candidates = [
            self.config.data_dir / "PCB_DATASET",
            self.config.data_dir / "pcb_defects",
            self.config.data_dir
        ]
        
        for path in candidates:
            if path.exists() and self._has_image_subdirs(path):
                return path
        
        for item in self.config.data_dir.iterdir():
            if item.is_dir() and self._has_image_subdirs(item):
                return item
        
        return None
    
    def _has_image_subdirs(self, path: Path) -> bool:
        """Check if path contains subdirectories with images."""
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if not subdirs:
            return False
        
        for subdir in subdirs[:3]:
            images = [f for f in subdir.iterdir() 
                     if f.suffix.lower() in self.VALID_EXTENSIONS]
            if images:
                return True
        return False
    
    def parse_directory_structure(self) -> Dict[str, List[Path]]:
        """Parse dataset directory to identify classes and images."""
        if self.dataset_path is None:
            raise ValueError("Dataset not loaded. Call download_dataset() first.")
        
        class_images: Dict[str, List[Path]] = {}
        
        self.logger.info(f"Parsing {self.dataset_path}")
        
        for class_dir in sorted(self.dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            images = [
                img for img in class_dir.iterdir()
                if img.suffix.lower() in self.VALID_EXTENSIONS
            ]
            
            if images:
                class_images[class_dir.name] = images
                self.logger.info(f"  {class_dir.name}: {len(images)} images")
        
        if not class_images:
            raise ValueError(f"No image classes found in {self.dataset_path}")
        
        self.class_names = sorted(class_images.keys())
        total = sum(len(v) for v in class_images.values())
        self.logger.info(f"Total: {len(self.class_names)} classes, {total} images")
        
        return class_images
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        if not self.class_names:
            raise ValueError("Call parse_directory_structure() first.")
        return self.class_names
