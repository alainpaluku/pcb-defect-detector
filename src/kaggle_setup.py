"""Kaggle API setup for dataset download."""

import os
import json
import getpass
from pathlib import Path


class KaggleSetup:
    """Manages Kaggle API authentication and dataset downloads."""
    
    def __init__(self):
        self.kaggle_dir = Path.home() / ".kaggle"
        self.credentials_file = self.kaggle_dir / "kaggle.json"
    
    def setup_credentials(self):
        """Set up Kaggle API credentials."""
        if self.credentials_file.exists():
            return True
        
        if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
            return True
        
        print("Kaggle credentials not found.")
        print("Get them from: https://www.kaggle.com/settings/account")
        
        username = input("Username: ")
        api_key = getpass.getpass("API Key: ")
        
        self.kaggle_dir.mkdir(exist_ok=True)
        with open(self.credentials_file, 'w') as f:
            json.dump({"username": username, "key": api_key}, f)
        
        try:
            os.chmod(self.credentials_file, 0o600)
        except Exception:
            pass
        
        return True
    
    def download_dataset(self, dataset_name, download_path="data"):
        """Download dataset from Kaggle."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.setup_credentials()
            
            api = KaggleApi()
            api.authenticate()
            
            download_dir = Path(download_path)
            download_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading: {dataset_name}")
            api.dataset_download_files(dataset_name, path=download_dir, unzip=True)
            print(f"Downloaded to: {download_dir}")
            return download_dir
            
        except ImportError:
            print("Error: pip install kaggle")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
