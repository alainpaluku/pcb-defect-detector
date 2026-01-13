"""
Unit tests for PCB Defect Detection Model.

Run with: python -m pytest tests/
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.model import PCBClassifier
from src.data_ingestion import DataIngestion


class TestConfig:
    """Test configuration class"""
    
    def test_config_values(self):
        """Test that config values are valid"""
        assert Config.IMG_SIZE == (224, 224)
        assert Config.BATCH_SIZE > 0
        assert Config.EPOCHS > 0
        assert 0 < Config.VALIDATION_SPLIT < 1
        assert Config.LEARNING_RATE > 0
    
    def test_path_detection(self):
        """Test automatic path detection"""
        data_path = Config.get_data_path()
        output_path = Config.get_output_path()
        
        assert isinstance(data_path, Path)
        assert isinstance(output_path, Path)
    
    def test_kaggle_detection(self):
        """Test Kaggle environment detection"""
        is_kaggle = Config.is_kaggle_environment()
        assert isinstance(is_kaggle, bool)


class TestPCBClassifier:
    """Test PCB classifier model"""
    
    @pytest.fixture
    def model(self):
        """Create a test model"""
        return PCBClassifier(num_classes=6)
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model.num_classes == 6
        assert model.img_size == (224, 224)
        assert model.model is None
    
    def test_model_building(self, model):
        """Test model building"""
        model.build_model()
        assert model.model is not None
        assert isinstance(model.model, tf.keras.Model)
    
    def test_model_input_shape(self, model):
        """Test model input shape"""
        model.build_model()
        expected_shape = (None, 224, 224, 3)
        assert model.model.input_shape == expected_shape
    
    def test_model_output_shape(self, model):
        """Test model output shape"""
        model.build_model()
        expected_shape = (None, 6)
        assert model.model.output_shape == expected_shape
    
    def test_model_compilation(self, model):
        """Test model compilation"""
        model.build_model()
        model.compile_model()
        assert model.model.optimizer is not None
        assert model.model.loss is not None
    
    def test_model_prediction(self, model):
        """Test model prediction"""
        model.build_model()
        model.compile_model()
        
        # Create dummy input
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Predict
        predictions = model.model.predict(dummy_input, verbose=0)
        
        # Check output shape and values
        assert predictions.shape == (1, 6)
        assert np.allclose(predictions.sum(), 1.0, atol=1e-5)  # Softmax sum = 1
        assert np.all(predictions >= 0) and np.all(predictions <= 1)
    
    def test_fine_tuning(self, model):
        """Test fine-tuning functionality"""
        model.build_model()
        model.compile_model()
        
        # Count trainable params before
        trainable_before = sum([tf.keras.backend.count_params(w) 
                               for w in model.model.trainable_weights])
        
        # Enable fine-tuning
        model.enable_fine_tuning(num_layers=20)
        
        # Count trainable params after
        trainable_after = sum([tf.keras.backend.count_params(w) 
                              for w in model.model.trainable_weights])
        
        # Should have more trainable params after fine-tuning
        assert trainable_after > trainable_before


class TestDataIngestion:
    """Test data ingestion pipeline"""
    
    @pytest.fixture
    def data_ingestion(self, tmp_path):
        """Create test data ingestion with temporary path"""
        # Create dummy dataset structure
        for class_name in ['class1', 'class2', 'class3']:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            
            # Create dummy images
            for i in range(10):
                img_path = class_dir / f"image_{i}.jpg"
                # Create a small dummy image
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                from PIL import Image
                Image.fromarray(img).save(img_path)
        
        return DataIngestion(data_path=tmp_path)
    
    def test_initialization(self, data_ingestion):
        """Test data ingestion initialization"""
        assert data_ingestion.data_path is not None
        assert data_ingestion.img_size == (224, 224)
        assert data_ingestion.batch_size > 0
    
    def test_dataset_analysis(self, data_ingestion):
        """Test dataset analysis"""
        stats = data_ingestion.analyze_dataset()
        
        assert 'total_images' in stats
        assert 'num_classes' in stats
        assert 'class_distribution' in stats
        assert 'imbalance_ratio' in stats
        
        assert stats['total_images'] == 30  # 3 classes * 10 images
        assert stats['num_classes'] == 3
    
    def test_class_weights_computation(self, data_ingestion):
        """Test class weights computation"""
        weights = data_ingestion.compute_class_weights()
        
        assert isinstance(weights, dict)
        assert len(weights) == 3  # 3 classes
        assert all(w > 0 for w in weights.values())


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_dummy(self):
        """Test end-to-end with dummy data"""
        # Create model
        model = PCBClassifier(num_classes=3)
        model.build_model()
        model.compile_model()
        
        # Create dummy data
        x_train = np.random.rand(10, 224, 224, 3).astype(np.float32)
        y_train = tf.keras.utils.to_categorical(
            np.random.randint(0, 3, 10), 
            num_classes=3
        )
        
        # Train for 1 epoch
        history = model.model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=2,
            verbose=0
        )
        
        # Check that training worked
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        assert len(history.history['loss']) == 1
    
    def test_model_save_load(self, tmp_path):
        """Test model saving and loading"""
        # Create and train model
        model = PCBClassifier(num_classes=3)
        model.build_model()
        model.compile_model()
        
        # Save model
        save_path = tmp_path / "test_model.h5"
        model.model.save(save_path)
        
        # Load model
        loaded_model = tf.keras.models.load_model(save_path)
        
        # Test prediction
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        pred1 = model.model.predict(dummy_input, verbose=0)
        pred2 = loaded_model.predict(dummy_input, verbose=0)
        
        # Predictions should be identical
        assert np.allclose(pred1, pred2)


class TestPerformance:
    """Performance tests"""
    
    def test_inference_speed(self):
        """Test inference speed"""
        import time
        
        # Create model
        model = PCBClassifier(num_classes=6)
        model.build_model()
        model.compile_model()
        
        # Warm up
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        model.model.predict(dummy_input, verbose=0)
        
        # Measure inference time
        start = time.time()
        for _ in range(10):
            model.model.predict(dummy_input, verbose=0)
        end = time.time()
        
        avg_time = (end - start) / 10
        
        # Should be less than 100ms on CPU
        assert avg_time < 0.1, f"Inference too slow: {avg_time:.3f}s"
    
    def test_memory_usage(self):
        """Test memory usage"""
        # Create model
        model = PCBClassifier(num_classes=6)
        model.build_model()
        
        # Count parameters
        total_params = model.model.count_params()
        
        # Should be around 3-4 million parameters
        assert 2_000_000 < total_params < 5_000_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
