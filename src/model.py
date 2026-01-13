"""
Model Architecture Module for PCB Defect Detection.

Implements MobileNetV2-based transfer learning for lightweight,
production-ready defect classification suitable for edge deployment.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from src.config import Config


class PCBClassifier:
    """
    PCB Defect Classification Model using MobileNetV2.
    
    Why MobileNetV2?
    ----------------
    1. Lightweight: ~14MB model size - deployable on edge devices
    2. Fast Inference: <50ms per image on CPU - suitable for real-time AOI
    3. Efficient: Inverted residual blocks reduce computational cost
    4. Proven: State-of-the-art accuracy on ImageNet with minimal parameters
    5. Industrial Ready: Can run on Raspberry Pi, NVIDIA Jetson, or factory PLCs
    
    Architecture:
    - Base: MobileNetV2 (pretrained on ImageNet)
    - Custom Head: Global pooling + Dense layers + Dropout
    - Output: Softmax for multi-class defect classification
    """
    
    def __init__(self, num_classes, img_size=Config.IMG_SIZE):
        """
        Initialize PCB classifier model.
        
        Args:
            num_classes (int): Number of defect classes to predict
            img_size (tuple): Input image dimensions (height, width)
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        
    def build_model(self, trainable_base_layers=0, use_dropout=True):
        """
        Build MobileNetV2-based classification model.
        
        Args:
            trainable_base_layers (int): Number of base layers to fine-tune.
                                        0 = freeze all base layers (faster training)
                                        >0 = fine-tune top N layers (better accuracy)
            use_dropout (bool): Whether to use dropout layers for regularization
        
        Returns:
            tf.keras.Model: Compiled model ready for training
        """
        # Load pretrained MobileNetV2 (without top classification layer)
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Optionally unfreeze top layers for fine-tuning
        if trainable_base_layers > 0:
            base_model.trainable = True
            # Freeze all layers except the last N
            for layer in base_model.layers[:-trainable_base_layers]:
                layer.trainable = False
        
        # Build custom classification head
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        # Preprocessing (MobileNetV2 expects [-1, 1] range)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom head for defect classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        # Dense layers with optional dropout for regularization
        x = layers.Dense(512, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        if use_dropout:
            x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        if use_dropout:
            x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        return self.model
    
    def compile_model(self, learning_rate=Config.LEARNING_RATE):
        """
        Compile model with optimizer, loss, and metrics.
        
        Args:
            learning_rate (float): Initial learning rate for Adam optimizer
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        print("Model compiled successfully")
        print(f"Learning rate: {learning_rate}")
    
    def get_model_summary(self):
        """
        Print model architecture summary.
        
        Returns:
            str: Model summary string
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        self.model.summary()
        
        # Calculate model size
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) 
                               for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print("\n" + "="*60)
        print("MODEL STATISTICS")
        print("="*60)
        print(f"Total Parameters:        {total_params:,}")
        print(f"Trainable Parameters:    {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        print(f"Estimated Model Size:    ~{total_params * 4 / (1024**2):.2f} MB")
        print("="*60 + "\n")
    
    def enable_fine_tuning(self, num_layers=20, learning_rate=1e-5):
        """
        Enable fine-tuning of base model layers.
        
        Call this after initial training to improve accuracy by unfreezing
        and training some base layers with a lower learning rate.
        
        Args:
            num_layers (int): Number of top layers to unfreeze
            learning_rate (float): Lower learning rate for fine-tuning
        """
        # Unfreeze the base model
        self.model.layers[2].trainable = True  # MobileNetV2 base
        
        # Freeze all layers except the last num_layers
        for layer in self.model.layers[2].layers[:-num_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=learning_rate)
        
        print(f"Fine-tuning enabled for top {num_layers} layers")
        print(f"New learning rate: {learning_rate}")
