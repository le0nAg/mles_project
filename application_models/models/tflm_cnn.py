"""
tflm_cnn.py - TFLM-optimized quantized CNN for RPI Pico

Designed for TensorFlow Lite for Microcontrollers:
- INT8 quantization-aware training
- Simple architecture for easy TFLM deployment
- Early exit support optional
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Try to import QAT toolkit (may not be available)
try:
    import tensorflow_model_optimization as tfmot
    HAS_TFMOT = True
except ImportError:
    HAS_TFMOT = False
    tfmot = None


def tflm_cnn_base(
    input_shape=(28, 28, 1),
    num_classes: int = 10,
    filters: tuple = (8, 16, 32),
    dense_units: int = 32,
    dropout: float = 0.0
) -> keras.Model:
    """
    Simple CNN optimized for TFLM deployment.
    
    Uses standard Conv2D (no depthwise separable) for better TFLM compatibility.
    """
    inputs = keras.Input(shape=input_shape, name="input")
    
    x = inputs
    
    # Conv blocks with pooling
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, 3, padding='same', use_bias=False, name=f"conv{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn{i+1}")(x)
        x = layers.ReLU(name=f"relu{i+1}")(x)
        if i < len(filters) - 1:  # Pool after all but last conv
            x = layers.MaxPooling2D(2, name=f"pool{i+1}")(x)
    
    # Classifier
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)
    
    x = layers.Dense(dense_units, activation='relu', name="fc")(x)
    outputs = layers.Dense(num_classes, name="output")(x)
    
    return keras.Model(inputs, outputs, name="tflm_cnn")


def tflm_cnn_tiny(input_shape=(28, 28, 1), num_classes: int = 10) -> keras.Model:
    """Smallest variant: ~15K params, target <20KB INT8."""
    return tflm_cnn_base(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=(8, 16),
        dense_units=24
    )


def tflm_cnn_small(input_shape=(28, 28, 1), num_classes: int = 10) -> keras.Model:
    """Small variant: ~25K params, target <30KB INT8."""
    return tflm_cnn_base(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=(8, 16, 32),
        dense_units=32
    )


def tflm_cnn_with_early_exit(
    input_shape=(28, 28, 1),
    num_classes: int = 10
) -> keras.Model:
    """
    TFLM CNN with early exit branch after first conv block.
    """
    inputs = keras.Input(shape=input_shape, name="input")
    
    # Block 1: 28x28 -> 14x14
    x = layers.Conv2D(8, 3, padding='same', use_bias=False, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    
    # Early exit branch
    early = layers.GlobalAveragePooling2D(name="early_gap")(x)
    early_out = layers.Dense(num_classes, name="early_exit")(early)
    
    # Block 2: 14x14 -> 7x7
    x = layers.Conv2D(16, 3, padding='same', use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    
    # Block 3: 7x7 -> 7x7
    x = layers.Conv2D(32, 3, padding='same', use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    
    # Final classifier
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(32, activation='relu', name="fc")(x)
    final_out = layers.Dense(num_classes, name="output")(x)
    
    outputs = {"early_exit": early_out, "output": final_out}
    
    return keras.Model(inputs, outputs, name="tflm_cnn_early_exit")


def apply_quantization_aware_training(model: keras.Model) -> keras.Model:
    """
    Apply quantization-aware training to model.
    
    Uses TensorFlow Model Optimization Toolkit.
    """
    if not HAS_TFMOT:
        print("Warning: tensorflow-model-optimization not available, skipping QAT")
        return model
    
    # Apply QAT to entire model
    qat_model = tfmot.quantization.keras.quantize_model(model)
    
    return qat_model


def apply_selective_qat(
    model: keras.Model,
    skip_layers: list = None
) -> keras.Model:
    """
    Apply QAT selectively, skipping certain layers.
    
    Useful for keeping first/last layers in higher precision.
    """
    if not HAS_TFMOT:
        print("Warning: tensorflow-model-optimization not available, skipping QAT")
        return model
    
    if skip_layers is None:
        skip_layers = []
    
    def apply_quantization_to_layer(layer):
        # Skip input layers
        if isinstance(layer, keras.layers.InputLayer):
            return layer
        
        # Skip specified layers
        if layer.name in skip_layers:
            return layer
        
        # Apply quantization to Dense and Conv2D layers
        if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        
        return layer
    
    # Clone and annotate
    annotated_model = keras.models.clone_model(
        model,
        clone_function=apply_quantization_to_layer
    )
    
    # Apply quantization
    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    
    return qat_model


class QuantizationAwareEarlyExitLoss(keras.losses.Loss):
    """Loss function for QAT with early exit."""
    
    def __init__(
        self,
        early_weight: float = 0.3,
        final_weight: float = 1.0,
        label_smoothing: float = 0.1,
        name: str = "qat_early_exit_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.early_weight = early_weight
        self.final_weight = final_weight
        self.ce_loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=keras.losses.Reduction.NONE
        )
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        if isinstance(y_pred, dict):
            early_loss = tf.reduce_mean(self.ce_loss(y_true, y_pred["early_exit"]))
            final_loss = tf.reduce_mean(self.ce_loss(y_true, y_pred["output"]))
            return self.early_weight * early_loss + self.final_weight * final_loss
        else:
            return tf.reduce_mean(self.ce_loss(y_true, y_pred))


def create_qat_model(
    model_type: str = "tiny",
    with_early_exit: bool = False,
    input_shape: tuple = (28, 28, 1),
    num_classes: int = 10
) -> keras.Model:
    """
    Factory function to create QAT-ready model.
    
    Args:
        model_type: "tiny", "small", or "early_exit"
        with_early_exit: Add early exit branch
        input_shape: Input shape
        num_classes: Number of classes
    
    Returns:
        QAT-wrapped Keras model
    """
    if with_early_exit or model_type == "early_exit":
        base_model = tflm_cnn_with_early_exit(input_shape, num_classes)
    elif model_type == "tiny":
        base_model = tflm_cnn_tiny(input_shape, num_classes)
    else:
        base_model = tflm_cnn_small(input_shape, num_classes)
    
    # For models with dict outputs, we need special handling
    if isinstance(base_model.output, dict):
        # QAT doesn't directly support dict outputs
        # Return base model - quantization applied during export
        return base_model
    
    # Apply QAT
    qat_model = apply_quantization_aware_training(base_model)
    
    return qat_model


if __name__ == "__main__":
    print("Testing TFLM CNN models...")
    
    # Test tiny model
    model = tflm_cnn_tiny()
    model.summary()
    print(f"\nTiny model params: {model.count_params()}")
    
    # Test small model
    print("\n" + "="*50)
    model = tflm_cnn_small()
    model.summary()
    print(f"\nSmall model params: {model.count_params()}")
    
    # Test early exit model
    print("\n" + "="*50)
    model = tflm_cnn_with_early_exit()
    model.summary()
    
    # Test forward pass
    dummy = np.random.rand(1, 28, 28, 1).astype(np.float32)
    outputs = model(dummy)
    print(f"\nEarly exit shape: {outputs['early_exit'].shape}")
    print(f"Final output shape: {outputs['output'].shape}")
    
    # Test QAT
    print("\n" + "="*50)
    print("Testing QAT...")
    try:
        qat_model = create_qat_model("tiny")
        qat_model.summary()
        print("QAT model created successfully")
    except Exception as e:
        print(f"QAT requires tensorflow-model-optimization: {e}")
