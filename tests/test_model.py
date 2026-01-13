"""Unit tests for PCB Defect Detection."""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.model import PCBClassifier


class TestConfig:
    """Tests for Config class."""
    
    def test_config_values(self):
        """Test default configuration values."""
        assert Config.IMG_SIZE == (224, 224)
        assert Config.BATCH_SIZE > 0
        assert 0 < Config.VALIDATION_SPLIT < 1
        assert Config.NUM_CLASSES == 6
        assert Config.LEARNING_RATE > 0
    
    def test_defect_classes(self):
        """Test defect class definitions."""
        assert len(Config.DEFECT_CLASSES) == 6
        expected_classes = [
            "missing_hole", "mouse_bite", "open_circuit",
            "short", "spur", "spurious_copper"
        ]
        for cls in expected_classes:
            assert cls in Config.DEFECT_CLASSES
    
    def test_paths(self):
        """Test path methods return Path objects."""
        assert isinstance(Config.get_data_path(), Path)
        assert isinstance(Config.get_output_path(), Path)
    
    def test_is_kaggle(self):
        """Test Kaggle environment detection."""
        result = Config.is_kaggle()
        assert isinstance(result, bool)
    
    def test_device_info(self):
        """Test device info retrieval."""
        info = Config.get_device_info()
        assert 'tensorflow_version' in info
        assert 'gpu_available' in info
        assert 'environment' in info
    
    def test_fine_tuning_config(self):
        """Test fine-tuning configuration."""
        assert Config.FINE_TUNE_EPOCHS > 0
        assert Config.FINE_TUNE_LAYERS > 0
        assert Config.FINE_TUNE_LR < Config.LEARNING_RATE


class TestPCBClassifier:
    """Tests for PCBClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create a classifier instance."""
        return PCBClassifier(num_classes=6)
    
    def test_init(self, classifier):
        """Test classifier initialization."""
        assert classifier.num_classes == 6
        assert classifier.img_size == (224, 224)
        assert classifier.model is None
    
    def test_build_model(self, classifier):
        """Test model building."""
        classifier.build_model()
        
        assert classifier.model is not None
        assert classifier.base_model is not None
        assert classifier.model.input_shape == (None, 224, 224, 3)
        assert classifier.model.output_shape == (None, 6)
    
    def test_compile_model(self, classifier):
        """Test model compilation."""
        classifier.build_model()
        classifier.compile_model()
        
        assert classifier.model.optimizer is not None
        metric_names = [m.name for m in classifier.model.metrics]
        assert 'accuracy' in metric_names
    
    def test_predict_shape(self, classifier):
        """Test prediction output shape."""
        classifier.build_model()
        classifier.compile_model()
        
        x = np.random.rand(1, 224, 224, 3).astype(np.float32)
        pred = classifier.model.predict(x, verbose=0)
        
        assert pred.shape == (1, 6)
    
    def test_predict_softmax(self, classifier):
        """Test that predictions sum to 1 (softmax)."""
        classifier.build_model()
        classifier.compile_model()
        
        x = np.random.rand(1, 224, 224, 3).astype(np.float32)
        pred = classifier.model.predict(x, verbose=0)
        
        assert np.allclose(pred.sum(), 1.0, atol=1e-5)
    
    def test_batch_predict(self, classifier):
        """Test batch prediction."""
        classifier.build_model()
        classifier.compile_model()
        
        x = np.random.rand(4, 224, 224, 3).astype(np.float32)
        pred = classifier.model.predict(x, verbose=0)
        
        assert pred.shape == (4, 6)
        assert np.allclose(pred.sum(axis=1), np.ones(4), atol=1e-5)
    
    def test_fine_tuning_increases_trainable_params(self, classifier):
        """Test that fine-tuning increases trainable parameters."""
        classifier.build_model()
        classifier.compile_model()
        
        trainable_before = sum(
            tf.keras.backend.count_params(w) 
            for w in classifier.model.trainable_weights
        )
        
        classifier.enable_fine_tuning(num_layers=20, learning_rate=1e-5)
        
        trainable_after = sum(
            tf.keras.backend.count_params(w) 
            for w in classifier.model.trainable_weights
        )
        
        assert trainable_after > trainable_before
    
    def test_save_load_keras(self, classifier, tmp_path):
        """Test model save/load in Keras format."""
        classifier.build_model()
        classifier.compile_model()
        
        # Save
        path = tmp_path / "model.keras"
        classifier.model.save(path)
        
        # Load
        loaded = tf.keras.models.load_model(path)
        
        # Compare predictions
        x = np.random.rand(1, 224, 224, 3).astype(np.float32)
        pred_original = classifier.model.predict(x, verbose=0)
        pred_loaded = loaded.predict(x, verbose=0)
        
        assert np.allclose(pred_original, pred_loaded, atol=1e-5)
    
    def test_model_layers(self, classifier):
        """Test model has expected layers."""
        classifier.build_model()
        
        layer_names = [layer.name for layer in classifier.model.layers]
        
        assert 'input_image' in layer_names
        assert 'global_pool' in layer_names
        assert 'predictions' in layer_names
    
    def test_custom_num_classes(self):
        """Test classifier with custom number of classes."""
        classifier = PCBClassifier(num_classes=10)
        classifier.build_model()
        
        assert classifier.model.output_shape == (None, 10)
    
    def test_custom_img_size(self):
        """Test classifier with custom image size."""
        classifier = PCBClassifier(num_classes=6, img_size=(128, 128))
        classifier.build_model()
        
        assert classifier.model.input_shape == (None, 128, 128, 3)


class TestModelPerformance:
    """Performance-related tests."""
    
    @pytest.fixture
    def trained_classifier(self):
        """Create and compile a classifier."""
        classifier = PCBClassifier(num_classes=6)
        classifier.build_model()
        classifier.compile_model()
        return classifier
    
    def test_inference_time(self, trained_classifier):
        """Test that inference is reasonably fast."""
        import time
        
        x = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Warm up
        trained_classifier.model.predict(x, verbose=0)
        
        # Measure
        start = time.time()
        for _ in range(10):
            trained_classifier.model.predict(x, verbose=0)
        elapsed = (time.time() - start) / 10
        
        # Should be under 100ms per inference
        assert elapsed < 0.1, f"Inference too slow: {elapsed*1000:.1f}ms"
    
    def test_model_size(self, trained_classifier, tmp_path):
        """Test that model size is reasonable for edge deployment."""
        path = tmp_path / "model.keras"
        trained_classifier.model.save(path)
        
        size_mb = path.stat().st_size / (1024 * 1024)
        
        # Should be under 20MB
        assert size_mb < 20, f"Model too large: {size_mb:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
