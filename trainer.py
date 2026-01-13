"""
Trainer module for PCB Defect Detector.

This module handles model compilation, callback management,
and the training loop execution.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam

from config import TrainingConfig


class Trainer:
    """Manages model training with callbacks and optimization.
    
    This class handles the complete training workflow including
    model compilation, callback setup, and training execution
    with support for class weights and learning rate scheduling.
    
    Attributes:
        config: Training configuration object.
        model: Keras model to train.
        logger: Logger instance for this class.
        history: Training history after fit.
        callbacks: List of training callbacks.
    
    Example:
        >>> config = TrainingConfig()
        >>> trainer = Trainer(config, model)
        >>> trainer.compile()
        >>> history = trainer.train(train_ds, val_ds, class_weights)
    """
    
    def __init__(self, config: TrainingConfig, model: Model) -> None:
        """Initialize Trainer with configuration and model.
        
        Args:
            config: TrainingConfig object containing training parameters.
            model: Keras Model to train.
        """
        self.config = config
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history: Optional[tf.keras.callbacks.History] = None
        self.callbacks: List[Callback] = []
        
        # Ensure checkpoint directory exists
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def compile(self, learning_rate: Optional[float] = None) -> None:
        """Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Optional learning rate override.
                          Uses config value if not specified.
        """
        lr = learning_rate or self.config.learning_rate
        
        optimizer = Adam(learning_rate=lr)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info(f"Model compiled with Adam optimizer (lr={lr})")
    
    def setup_callbacks(self) -> List[Callback]:
        """Set up training callbacks.
        
        Creates and configures:
        - EarlyStopping: Stops training when validation loss stops improving
        - ModelCheckpoint: Saves best model based on validation loss
        - ReduceLROnPlateau: Reduces learning rate when validation loss plateaus
        
        Returns:
            List of configured Keras callbacks.
        """
        self.logger.info("Setting up training callbacks...")
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        self.logger.info(f"  EarlyStopping: patience={self.config.early_stopping_patience}")
        
        # Model checkpoint
        checkpoint_path = self.config.checkpoint_dir / "best_model.keras"
        model_checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        self.logger.info(f"  ModelCheckpoint: {checkpoint_path}")
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_learning_rate,
            verbose=1
        )
        self.logger.info(f"  ReduceLROnPlateau: factor={self.config.reduce_lr_factor}, "
                        f"patience={self.config.reduce_lr_patience}")
        
        self.callbacks = [early_stopping, model_checkpoint, reduce_lr]
        return self.callbacks

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        class_weights: Optional[Dict[int, float]] = None,
        epochs: Optional[int] = None
    ) -> tf.keras.callbacks.History:
        """Execute the training loop.
        
        Args:
            train_dataset: TensorFlow Dataset for training.
            val_dataset: TensorFlow Dataset for validation.
            class_weights: Optional dictionary of class weights for
                          handling imbalanced data.
            epochs: Optional number of epochs override.
        
        Returns:
            Keras History object containing training metrics.
        """
        num_epochs = epochs or self.config.epochs
        
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        
        if class_weights:
            self.logger.info("Using class weights for imbalanced data handling.")
        
        # Setup callbacks if not already done
        if not self.callbacks:
            self.setup_callbacks()
        
        try:
            self.history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=num_epochs,
                callbacks=self.callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            self.logger.info("Training completed successfully.")
            self._log_training_summary()
            
            return self.history
            
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user.")
            raise
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _log_training_summary(self) -> None:
        """Log a summary of training results."""
        if self.history is None:
            return
        
        final_epoch = len(self.history.history['loss'])
        
        self.logger.info("Training Summary:")
        self.logger.info(f"  Total epochs: {final_epoch}")
        self.logger.info(f"  Final train loss: {self.history.history['loss'][-1]:.4f}")
        self.logger.info(f"  Final train accuracy: {self.history.history['accuracy'][-1]:.4f}")
        self.logger.info(f"  Final val loss: {self.history.history['val_loss'][-1]:.4f}")
        self.logger.info(f"  Final val accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        
        # Find best epoch
        best_epoch = min(
            range(len(self.history.history['val_loss'])),
            key=lambda i: self.history.history['val_loss'][i]
        )
        self.logger.info(f"  Best epoch: {best_epoch + 1}")
        self.logger.info(f"  Best val loss: {self.history.history['val_loss'][best_epoch]:.4f}")
        self.logger.info(f"  Best val accuracy: {self.history.history['val_accuracy'][best_epoch]:.4f}")
    
    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save the trained model.
        
        Args:
            path: Optional path for saving. Uses default if not specified.
        
        Returns:
            Path where model was saved.
        """
        save_path = path or (self.config.checkpoint_dir / "final_model.keras")
        self.model.save(str(save_path))
        self.logger.info(f"Model saved to {save_path}")
        return save_path
    
    def load_best_model(self) -> Model:
        """Load the best model from checkpoint.
        
        Returns:
            Loaded Keras Model.
        
        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        checkpoint_path = self.config.checkpoint_dir / "best_model.keras"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        self.model = tf.keras.models.load_model(str(checkpoint_path))
        self.logger.info(f"Loaded best model from {checkpoint_path}")
        return self.model
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history as dictionary.
        
        Returns:
            Dictionary containing training metrics history.
        
        Raises:
            ValueError: If training hasn't been run yet.
        """
        if self.history is None:
            raise ValueError("No training history. Call train() first.")
        return dict(self.history.history)
