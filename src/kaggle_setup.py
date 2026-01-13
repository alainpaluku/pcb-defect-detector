"""
Kaggle API Setup Module.

Handles secure authentication and dataset download for local development.
Uses environment variables and getpass for credential security.
"""

import os
import json
import getpass
from pathlib import Path


class KaggleSetup:
    """
    Manages Kaggle API authentication and dataset downloads.
    
    This class ensures secure credential handling without hardcoding tokens,
    following security best practices for production environments.
    """
    
    def __init__(self):
        """Initialize Kaggle setup with credential validation."""
        self.kaggle_dir = Path.home() / ".kaggle"
        self.credentials_file = self.kaggle_dir / "kaggle.json"
    
    def setup_credentials(self):
        """
        Set up Kaggle API credentials securely.
        
        Checks for existing credentials, prompts user if needed,
        and validates the setup.
        
        Returns:
            bool: True if credentials are set up successfully
        """
        # Check if credentials already exist
        if self.credentials_file.exists():
            print("✓ Kaggle credentials found")
            return True
        
        # Check environment variables
        if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
            print("✓ Kaggle credentials found in environment variables")
            return True
        
        # Prompt user for credentials
        print("Kaggle credentials not found. Please provide your API credentials.")
        print("(Get them from: https://www.kaggle.com/settings/account)")
        
        username = input("Kaggle Username: ")
        api_key = getpass.getpass("Kaggle API Key: ")
        
        # Create .kaggle directory
        self.kaggle_dir.mkdir(exist_ok=True)
        
        # Save credentials
        credentials = {
            "username": username,
            "key": api_key
        }
        
        with open(self.credentials_file, 'w') as f:
            json.dump(credentials, f)
        
        # Set proper permissions (Unix-like systems)
        try:
            os.chmod(self.credentials_file, 0o600)
        except Exception:
            pass  # Windows doesn't support chmod
        
        print("✓ Kaggle credentials saved successfully")
        return True
    
    def download_dataset(self, dataset_name, download_path="data"):
        """
        Download dataset from Kaggle.
        
        Args:
            dataset_name (str): Kaggle dataset identifier (e.g., 'akhatova/pcb-defects')
            download_path (str): Local directory to download dataset
        
        Returns:
            Path: Path to downloaded dataset
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Ensure credentials are set up
            self.setup_credentials()
            
            # Initialize API
            api = KaggleApi()
            api.authenticate()
            
            # Create download directory
            download_dir = Path(download_path)
            download_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading dataset: {dataset_name}")
            api.dataset_download_files(
                dataset_name,
                path=download_dir,
                unzip=True
            )
            
            print(f"✓ Dataset downloaded to: {download_dir}")
            return download_dir
            
        except ImportError:
            print("Error: kaggle package not installed. Run: pip install kaggle")
            return None
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    @staticmethod
    def set_environment_variables(username, api_key):
        """
        Set Kaggle credentials as environment variables.
        
        Args:
            username (str): Kaggle username
            api_key (str): Kaggle API key
        """
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = api_key
        print("✓ Environment variables set")
