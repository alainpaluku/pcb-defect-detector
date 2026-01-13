"""Model architecture for PCB Defect Detection using MobileNetV2."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from src.config import Config


class PCBClassifier:
    """MobileNetV2-based classifier for PCB defect detection.
    
    Architecture optimized for:
    - Small dataset (~1400 images)
    - 6 defect classes
    - Edge deployment (~14MB model)
    """
    
    def __init__(self, num_classes=Config.NUM_CLASSES, img_size=Config.IMG_SIZE):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.base_model = None
        
    def build_model(self, dropout_rate=0.5, l2_reg=0.01):
        """Build MobileNetV2-based classification model.
        
        Args:
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained MobileNetV2
        self.base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet',
            alpha=1.0  # Width multiplier
        )
        
        # Freeze base model initially
        self.base_model.trainable = False
        
        # Build classification head
        inputs = layers.Input(shape=(*self.img_size, 3), name='input_image')
        
        # MobileNetV2 preprocessing
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Feature extraction
        x = self.base_model(x, training=False)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        
        # Batch normalization
        x = layers.BatchNormalization(name='bn_head')(x)
        
        # Dense layers with stronger regularization to reduce overfitting
        x = layers.Dense(
            128,  # Réduit de 256 à 128
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='dense_1'
        )(x)
        x = layers.Dropout(dropout_rate, name='dropout_1')(x)
        
        x = layers.Dense(
            64,  # Réduit de 128 à 64
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='dense_2'
        )(x)
        x = layers.Dropout(dropout_rate * 0.8, name='dropout_2')(x)  # Plus de dropout
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax',
            name='predictions'
        )(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs, name='PCB_Classifier')
        
        return self.model
    
    def compile_model(self, learning_rate=Config.LEARNING_RATE):
        """Compile model with optimizer and metrics.
        
        Args:
            learning_rate: Initial learning rate
        """
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc', curve='ROC'),
                tf.keras.metrics.AUC(name='auc_pr', curve='PR')  # Precision-Recall AUC
            ]
        )
        
        print(f"✅ Model compiled with learning rate: {learning_rate}")
    
    def enable_fine_tuning(self, num_layers=Config.FINE_TUNE_LAYERS, 
                           learning_rate=Config.FINE_TUNE_LR):
        """Enable fine-tuning of base model layers.
        
        Args:
            num_layers: Number of layers to unfreeze from the end
            learning_rate: Learning rate for fine-tuning (should be low)
        """
        # Unfreeze base model
        self.base_model.trainable = True
        
        # Freeze all layers except the last `num_layers`
        for layer in self.base_model.layers[:-num_layers]:
            layer.trainable = False
        
        # Count trainable parameters
        trainable_count = sum(
            tf.keras.backend.count_params(w) 
            for w in self.model.trainable_weights
        )
        non_trainable_count = sum(
            tf.keras.backend.count_params(w) 
            for w in self.model.non_trainable_weights
        )
        
        print(f"\n🔓 Fine-tuning enabled:")
        print(f"   Unfrozen layers: {num_layers}")
        print(f"   Trainable params: {trainable_count:,}")
        print(f"   Non-trainable params: {non_trainable_count:,}")
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=learning_rate)
    
    def get_model_summary(self):
        """Print model summary with layer details."""
        print("\n" + "=" * 60)
        print("🏗️  MODEL ARCHITECTURE")
        print("=" * 60)
        self.model.summary()
        
        # Count parameters
        total_params = self.model.count_params()
        trainable_params = sum(
            tf.keras.backend.count_params(w) 
            for w in self.model.trainable_weights
        )
        
        print(f"\n📊 Parameter Summary:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        print(f"   Non-trainable: {total_params - trainable_params:,}")
        print(f"   Estimated size: {total_params * 4 / (1024**2):.2f} MB")
        print("=" * 60 + "\n")
    
    def predict_single(self, image_path, class_names):
        """Predict class for a single image.
        
        Args:
            image_path: Path to image file
            class_names: List of class names
            
        Returns:
            dict with prediction results
        """
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.img_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, 0) / 255.0
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top prediction
        top_idx = np.argmax(predictions)
        
        return {
            'class': class_names[top_idx],
            'confidence': float(predictions[top_idx]),
            'all_probabilities': {
                name: float(prob) 
                for name, prob in zip(class_names, predictions)
            }
        }
    
    def save(self, path, format='keras'):
        """Save model to file.
        
        Args:
            path: Output path
            format: 'keras', 'h5', or 'savedmodel'
        """
        from pathlib import Path
        path = Path(path)
        
        if format == 'keras':
            self.model.save(path.with_suffix('.keras'))
        elif format == 'h5':
            self.model.save(path.with_suffix('.h5'))
        elif format == 'savedmodel':
            self.model.save(path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"💾 Model saved: {path}")
    
    @classmethod
    def load(cls, path):
        """Load model from file.
        
        Args:
            path: Path to saved model
            
        Returns:
            PCBClassifier instance
        """
        model = tf.keras.models.load_model(path)
        
        instance = cls(
            num_classes=model.output_shape[-1],
            img_size=model.input_shape[1:3]
        )
        instance.model = model
        
        return instance
