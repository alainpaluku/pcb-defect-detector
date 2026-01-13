"""
Model Builder module for PCB Defect Detector.

This module handles the construction of transfer learning models
using pretrained base models with custom classification heads.
"""

import logging
from typing import Optional, Tuple, Dict

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0

from config import ModelConfig, DataConfig


class PCBModelBuilder:
    """Builds transfer learning models for PCB defect classification.
    
    Supports MobileNetV2, ResNet50, and EfficientNetB0 as base models
    with customizable classification heads.
    """
    
    SUPPORTED_MODELS: Dict[str, type] = {
        'MobileNetV2': MobileNetV2,
        'ResNet50': ResNet50,
        'EfficientNetB0': EfficientNetB0
    }
    
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        num_classes: int
    ) -> None:
        """Initialize PCBModelBuilder.
        
        Args:
            model_config: Model architecture configuration.
            data_config: Data configuration for input shape.
            num_classes: Number of output classes.
        """
        self.model_config = model_config
        self.data_config = data_config
        self.num_classes = num_classes
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model: Optional[Model] = None
        self.base_model: Optional[Model] = None
        
        if model_config.base_model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_config.base_model}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )
    
    def build(self) -> Model:
        """Build the complete transfer learning model.
        
        Returns:
            Keras Model ready for training.
        """
        self.logger.info(f"Building model with {self.model_config.base_model}...")
        
        input_shape = (*self.data_config.image_size, 3)
        
        # Create base model
        self.base_model = self._create_base_model(input_shape)
        
        # Freeze base model if specified
        if self.model_config.freeze_base:
            self.base_model.trainable = False
            self.logger.info("Base model frozen.")
        
        # Build complete model
        self.model = self._build_model(input_shape)
        
        # Log parameters
        trainable = sum(tf.keras.backend.count_params(w) for w in self.model.trainable_weights)
        total = sum(tf.keras.backend.count_params(w) for w in self.model.weights)
        
        self.logger.info(f"Trainable params: {trainable:,}")
        self.logger.info(f"Total params: {total:,}")
        
        return self.model
    
    def _create_base_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Create the pretrained base model."""
        base_class = self.SUPPORTED_MODELS[self.model_config.base_model]
        
        base_model = base_class(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        self.logger.info(f"Loaded {self.model_config.base_model} with ImageNet weights.")
        return base_model
    
    def _build_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Build the complete model with classification head."""
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Base model
        x = self.base_model(inputs, training=False)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        
        # Batch normalization
        if self.model_config.use_batch_norm:
            x = layers.BatchNormalization(name='bn1')(x)
        
        # Dense layer with regularization
        x = layers.Dense(
            self.model_config.dense_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.model_config.l2_regularization),
            name='dense'
        )(x)
        
        # Dropout
        x = layers.Dropout(self.model_config.dropout_rate, name='dropout')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        return Model(inputs=inputs, outputs=outputs, name='pcb_classifier')
    
    def unfreeze_layers(self, num_layers: Optional[int] = None) -> None:
        """Unfreeze base model layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end.
                       If None, unfreezes all layers.
        """
        if self.base_model is None:
            raise ValueError("Model not built yet.")
        
        self.base_model.trainable = True
        
        if num_layers is not None:
            for layer in self.base_model.layers[:-num_layers]:
                layer.trainable = False
            self.logger.info(f"Unfroze last {num_layers} layers.")
        else:
            self.logger.info("Unfroze all base layers.")
        
        # Log new trainable count
        trainable = sum(tf.keras.backend.count_params(w) for w in self.model.trainable_weights)
        self.logger.info(f"New trainable params: {trainable:,}")
    
    def get_model(self) -> Model:
        """Get the built model."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        return self.model
