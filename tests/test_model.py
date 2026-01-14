"""Tests for PCB Defect Detection."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


class TestConfig(unittest.TestCase):
    """Test configuration."""
    
    def test_class_names(self):
        """Test class names are defined."""
        self.assertEqual(len(Config.CLASS_NAMES), 6)
        self.assertIn("missing_hole", Config.CLASS_NAMES)
        self.assertIn("short", Config.CLASS_NAMES)
    
    def test_class_map(self):
        """Test class mapping handles different cases."""
        self.assertEqual(Config.CLASS_MAP["missing_hole"], 0)
        self.assertEqual(Config.CLASS_MAP["Missing_hole"], 0)
        self.assertEqual(Config.CLASS_MAP["short"], 3)
        self.assertEqual(Config.CLASS_MAP["Short"], 3)
    
    def test_num_classes(self):
        """Test number of classes."""
        self.assertEqual(Config.NUM_CLASSES, 6)
    
    def test_model_settings(self):
        """Test model settings are valid."""
        self.assertGreater(Config.IMG_SIZE, 0)
        self.assertGreater(Config.BATCH_SIZE, 0)
        self.assertGreater(Config.EPOCHS, 0)
    
    def test_output_path(self):
        """Test output path is valid."""
        output = Config.get_output_path()
        self.assertIsInstance(output, Path)


class TestDataIngestion(unittest.TestCase):
    """Test data ingestion."""
    
    def test_import(self):
        """Test module can be imported."""
        from src.data_ingestion import DataIngestion
        self.assertIsNotNone(DataIngestion)
    
    def test_init(self):
        """Test initialization."""
        from src.data_ingestion import DataIngestion
        data = DataIngestion(data_path="test_data")
        self.assertEqual(data.data_path, Path("test_data"))


class TestModel(unittest.TestCase):
    """Test model module."""
    
    def test_import(self):
        """Test module can be imported."""
        from src.model import PCBDetector
        self.assertIsNotNone(PCBDetector)


class TestDetector(unittest.TestCase):
    """Test detector module."""
    
    def test_import(self):
        """Test module can be imported."""
        from src.detector import PCBInspector
        self.assertIsNotNone(PCBInspector)


class TestTrainer(unittest.TestCase):
    """Test trainer module."""
    
    def test_import(self):
        """Test module can be imported."""
        from src.trainer import TrainingManager
        self.assertIsNotNone(TrainingManager)


if __name__ == "__main__":
    unittest.main()
