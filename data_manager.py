"""
Kaggle Data Manager module for PCB Defect Detector.

This module handles authentication with Kaggle API, dataset downloading,
and directory structure parsing for the PCB defects dataset.
"""

import os
import json
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional
from getpass import getpass

from config import DataConfig


class KaggleDataManager:
    """Manages Kaggle authentication and dataset operations.
    
    This class handles secure authentication with Kaggle API using
    environment variables, downloads datasets, and parses the resulting
    directory structure for training data preparation.
    
    Attributes:
        config: Data configuration object.
        logger: Logger instance for this class.
        dataset_path: Path to the extracted dataset.
        class_names: List of detected class names from dataset.
    
    Example:
        >>> config = DataConfig()
        >>> manager = KaggleDataManager(config)
        >>> manager.authenticate()
        >>> manager.download_dataset()
        >>> structure = manager.parse_directory_structure()
    """
    
    def __init__(self, config: DataConfig) -> None:
        """Initialize KaggleDataManager with configuration.
        
        Args:
            config: DataConfig object containing dataset parameters.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_path: Optional[Path] = None
        self.class_names: List[str] = []
        
    def authenticate(self, api_token: Optional[str] = None) -> None:
        """Authenticate with Kaggle API using environment variable or user input.
        
        This method sets up Kaggle credentials either from an environment
        variable, a provided token, or by prompting the user securely.
        
        Args:
            api_token: Optional JSON string containing Kaggle credentials.
                      If not provided, will check environment variable or
                      prompt user for input.
        
        Raises:
            ValueError: If authentication credentials are invalid.
            PermissionError: If unable to create Kaggle config directory.
        """
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        # Check if already authenticated
        if kaggle_json.exists():
            self.logger.info("Kaggle credentials already configured.")
            return
            
        # Try to get token from various sources
        token = api_token or os.environ.get('KAGGLE_API_TOKEN')
        
        if not token:
            self.logger.info("No Kaggle API token found. Please enter your credentials.")
            print("\nEnter your Kaggle API token (JSON format):")
            print('Example: {"username":"your_username","key":"your_api_key"}')
            token = getpass("Token: ")
        
        # Validate token format
        try:
            credentials = json.loads(token)
            if 'username' not in credentials or 'key' not in credentials:
                raise ValueError("Token must contain 'username' and 'key' fields.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format for Kaggle token: {e}")
        
        # Create kaggle directory and save credentials
        try:
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            kaggle_json.write_text(token)
            kaggle_json.chmod(0o600)  # Secure file permissions
            self.logger.info("Kaggle credentials saved successfully.")
        except OSError as e:
            raise PermissionError(f"Failed to save Kaggle credentials: {e}")
        
        # Set environment variables for current session
        os.environ['KAGGLE_USERNAME'] = credentials['username']
        os.environ['KAGGLE_KEY'] = credentials['key']

    def download_dataset(self, force: bool = False) -> Path:
        """Download and extract the PCB defects dataset from Kaggle.
        
        Downloads the dataset if not already present, or if force=True.
        Automatically extracts the zip file after download.
        
        Args:
            force: If True, re-download even if dataset exists.
        
        Returns:
            Path to the extracted dataset directory.
        
        Raises:
            RuntimeError: If download or extraction fails.
        """
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset already exists
        expected_path = self.config.data_dir / "PCB_DATASET"
        if expected_path.exists() and not force:
            self.logger.info(f"Dataset already exists at {expected_path}")
            self.dataset_path = expected_path
            return self.dataset_path
        
        self.logger.info(f"Downloading dataset: {self.config.dataset_name}")
        
        try:
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            api.dataset_download_files(
                self.config.dataset_name,
                path=str(self.config.data_dir),
                unzip=False
            )
            
            # Find and extract zip file
            zip_files = list(self.config.data_dir.glob("*.zip"))
            if not zip_files:
                raise RuntimeError("No zip file found after download.")
            
            zip_path = zip_files[0]
            self.logger.info(f"Extracting {zip_path}...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.data_dir)
            
            # Clean up zip file
            zip_path.unlink()
            self.logger.info("Dataset extracted successfully.")
            
            # Find the actual dataset directory
            self.dataset_path = self._find_dataset_root()
            return self.dataset_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download/extract dataset: {e}")
    
    def _find_dataset_root(self) -> Path:
        """Find the root directory of the extracted dataset.
        
        Returns:
            Path to the dataset root directory.
        
        Raises:
            FileNotFoundError: If dataset directory cannot be found.
        """
        # Common patterns for PCB dataset structure
        possible_paths = [
            self.config.data_dir / "PCB_DATASET",
            self.config.data_dir / "pcb_defects",
            self.config.data_dir / "images",
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                return path
        
        # Search for directory with image subdirectories
        for item in self.config.data_dir.iterdir():
            if item.is_dir():
                subdirs = list(item.iterdir())
                if subdirs and all(d.is_dir() for d in subdirs[:5]):
                    return item
        
        raise FileNotFoundError(
            f"Could not find dataset root in {self.config.data_dir}"
        )
    
    def parse_directory_structure(self) -> Dict[str, List[Path]]:
        """Parse the dataset directory structure to identify classes and images.
        
        Scans the dataset directory to identify class folders and their
        associated image files. Supports common image formats.
        
        Returns:
            Dictionary mapping class names to lists of image paths.
        
        Raises:
            ValueError: If no valid image classes are found.
        """
        if self.dataset_path is None:
            raise ValueError("Dataset not downloaded. Call download_dataset() first.")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        class_images: Dict[str, List[Path]] = {}
        
        self.logger.info(f"Parsing directory structure at {self.dataset_path}")
        
        for class_dir in sorted(self.dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            images = [
                img for img in class_dir.iterdir()
                if img.suffix.lower() in valid_extensions
            ]
            
            if images:
                class_images[class_name] = images
                self.logger.info(f"  Class '{class_name}': {len(images)} images")
        
        if not class_images:
            raise ValueError(f"No valid image classes found in {self.dataset_path}")
        
        self.class_names = sorted(class_images.keys())
        self.logger.info(f"Found {len(self.class_names)} classes with "
                        f"{sum(len(v) for v in class_images.values())} total images")
        
        return class_images
    
    def get_class_names(self) -> List[str]:
        """Get the list of class names in the dataset.
        
        Returns:
            Sorted list of class names.
        
        Raises:
            ValueError: If directory structure hasn't been parsed yet.
        """
        if not self.class_names:
            raise ValueError("Directory not parsed. Call parse_directory_structure() first.")
        return self.class_names
