"""
Model Builder module for PCB Defect Detector.

This module handles the construction of transfer learning models
using pretrained base models with custom classification heads.
"""

import logging
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2, ResNet50

from config import ModelConfig, DataConfig


class PCBModelBuilder:
    """Builds transfer learning models for PCB defect classification.
    
    This class constructs CNN models using pretrained base models
    (MobileNetV2 or ResNet50) with frozen weights and custom
    classification heads optimized for PCB defect detection.
    
    Attributes:
        model_config: Model architecture configuration.
        data_config: Data configuration for input shape.
        num_classes: Number of output classes.
        logger: Logger instance for this class.
        model: The constructed Keras model.
    
    Example:
        >>> model_config = ModelConfig()
        >>> data_config = DataConfig()
        >>> builder = PCBModelBuilder(model_config, data_config, num_classes=6)
        >>> model = builder.build()
        >>> model.summary()
    """
    
    SUPPORTED_MODELS = {
        'MobileNetV2': MobileNetV2,
        'ResNet50': ResNet50
    }
    
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        num_classes: int
    ) -> None:
        """Initialize PCBModelBuilder with configurations.
        
        Args:
            model_config: ModelConfig object containing architecture parameters.
            data_config: DataConfig object containing input shape parameters.
            num_classes: Number of classes for classification.
        
        Raises:
            ValueError: If specified base model is not supported.
        """
        self.model_config = model_config
        self.data_config = data_config
        self.num_classes = num_classes
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model: Optional[Model] = None
        
        if model_config.base_model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported base model: {model_config.base_model}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
    
    def build(self) -> Model:
        """Build the complete transfer learning model.
        
        Constructs a model with:
        - Pretrained base model (frozen or trainable)
        - Global Average Pooling layer
        - Dense layer with ReLU activation
        - Dropout for regularization
        - Softmax output layer
        
        Returns:
            Compiled Keras Model ready for training.
        """
        self.logger.info(f"Building model with {self.model_config.base_model} base...")
        
        input_shape = (*self.data_config.image_size, 3)
        
        # Create base model
        base_model = self._create_base_model(input_shape)
        
        # Freeze base model if specified
        if self.model_config.freeze_base:
            base_model.trainable = False
            self.logger.info("Base model layers frozen.")
        else:
            self.logger.info("Base model layers trainable.")
        
        # Build custom head
        self.model = self._add_classification_head(base_model, input_shape)
        
        # Log model summary
        trainable_params = sum(
            tf.keras.backend.count_params(w) for w in self.model.trainable_weights
        )
        non_trainable_params = sum(
            tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights
        )
        
        self.logger.info(f"Model built successfully:")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Non-trainable parameters: {non_trainable_params:,}")
        self.logger.info(f"  Total parameters: {trainable_params + non_trainable_params:,}")
        
        return self.model
    
    def _create_base_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Create the pretrained base model.
        
        Args:
            input_shape: Input tensor shape (height, width, channels).
        
        Returns:
            Pretrained Keras Model without top layers.
        """
        base_model_class = self.SUPPORTED_MODELS[self.model_config.base_model]
        
        base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        self.logger.info(f"Loaded {self.model_config.base_model} with ImageNet weights.")
        return base_model
    
    def _add_classification_head(
        self,
        base_model: Model,
        input_shape: Tuple[int, int, int]
    ) -> Model:
        """Add custom classification head to base model.
        
        Args:
            base_model: Pretrained base model.
            input_shape: Input tensor shape.
        
        Returns:
            Complete model with classification head.
        """
        # Input layer
        inputs = layers.Input(shape=input_shape, name='input_image')
        
        # Base model
        x = base_model(inputs, training=False)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Dense layer
        x = layers.Dense(
            self.model_config.dense_units,
            activation='relu',
            name='dense_features'
        )(x)
        
        # Dropout for regularization
        x = layers.Dropout(
            self.model_config.dropout_rate,
            name='dropout'
        )(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='pcb_defect_classifier')
        return model
    
    def unfreeze_base_layers(self, num_layers: Optional[int] = None) -> None:
        """Unfreeze base model layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the top.
                       If None, unfreezes all layers.
        
        Raises:
            ValueError: If model hasn't been built yet.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        # Find the base model layer
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, Model):
                base_model = layer
                break
        
        if base_model is None:
            self.logger.warning("Could not find base model layer.")
            return
        
        base_model.trainable = True
        
        if num_layers is not None:
            # Freeze all layers except the last num_layers
            for layer in base_model.layers[:-num_layers]:
                layer.trainable = False
            self.logger.info(f"Unfroze last {num_layers} layers of base model.")
        else:
            self.logger.info("Unfroze all base model layers.")
    
    def get_model(self) -> Model:
        """Get the built model.
        
        Returns:
            The Keras Model.
        
        Raises:
            ValueError: If model hasn't been built yet.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        return self.model
