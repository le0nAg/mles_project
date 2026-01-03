"""
mcunet_tiny.py - Ultra-tiny CNN optimized for RPI Pico

Architecture based on MCUNet principles:
- Depthwise separable convolutions for efficiency
- Early exit branches for confidence-based inference
- Designed for <50KB INT8 model size
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class EarlyExitBranch(layers.Layer):
    """Early exit branch with confidence-based decision."""
    
    def __init__(self, num_classes: int = 10, name: str = "early_exit", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.gap = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes, name=f"{name}_logits")
    
    def call(self, x):
        x = self.gap(x)
        return self.classifier(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


def depthwise_separable_block(x, filters, stride=1, name="dsconv"):
    """Depthwise separable convolution block."""
    # Depthwise
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding='same',
        use_bias=False,
        name=f"{name}_dw"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)
    
    # Pointwise
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=f"{name}_pw"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(name=f"{name}_pw_relu")(x)
    
    return x


def mcunet_tiny(
    input_shape=(28, 28, 1),
    num_classes: int = 10,
    with_early_exit: bool = True
) -> keras.Model:
    """
    MCUNet-style tiny model for digit recognition.
    
    Target specs:
    - Parameters: <30K
    - INT8 size: <35KB
    - Optimized for ARM Cortex-M0+
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        with_early_exit: Include early exit branch
    
    Returns:
        Keras model (functional API)
    """
    inputs = keras.Input(shape=input_shape, name="input")
    
    # Initial conv: 28x28x1 -> 28x28x8
    x = layers.Conv2D(8, 3, padding='same', use_bias=False, name="conv1")(inputs)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.ReLU(name="conv1_relu")(x)
    
    # Block 1: 28x28x8 -> 14x14x16
    x = depthwise_separable_block(x, 16, stride=2, name="block1")
    
    outputs = {}
    
    # Early exit after block 1 (optional)
    if with_early_exit:
        early_logits = EarlyExitBranch(num_classes, name="early_exit")(x)
        outputs["early_exit"] = early_logits
    
    # Block 2: 14x14x16 -> 7x7x32
    x = depthwise_separable_block(x, 32, stride=2, name="block2")
    
    # Block 3: 7x7x32 -> 7x7x32 (no stride)
    x = depthwise_separable_block(x, 32, stride=1, name="block3")
    
    # Final classification
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(24, name="fc1")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    final_logits = layers.Dense(num_classes, name="output")(x)
    
    outputs["output"] = final_logits
    
    if with_early_exit:
        model = keras.Model(inputs, outputs, name="mcunet_tiny_early_exit")
    else:
        model = keras.Model(inputs, final_logits, name="mcunet_tiny")
    
    return model


def mcunet_tiny_v2(
    input_shape=(28, 28, 1),
    num_classes: int = 10
) -> keras.Model:
    """
    Even smaller MCUNet variant - single output, no early exit.
    Target: <20KB INT8 model.
    """
    inputs = keras.Input(shape=input_shape, name="input")
    
    # Conv 1: 28x28x1 -> 14x14x8
    x = layers.Conv2D(8, 3, strides=2, padding='same', use_bias=False, name="conv1")(inputs)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.ReLU(name="conv1_relu")(x)
    
    # Conv 2: 14x14x8 -> 7x7x16
    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False, name="conv2")(inputs)
    x = layers.BatchNormalization(name="conv2_bn")(x)
    x = layers.ReLU(name="conv2_relu")(x)
    
    # Global pooling + classifier
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(16, activation='relu', name="fc1")(x)
    outputs = layers.Dense(num_classes, name="output")(x)
    
    return keras.Model(inputs, outputs, name="mcunet_tiny_v2")


def mcunet_inference_model(trained_model: keras.Model) -> keras.Model:
    """
    Create inference-only model from trained early-exit model.
    Outputs only final logits (for simpler deployment).
    """
    inputs = trained_model.input
    
    # Get final output only
    if isinstance(trained_model.output, dict):
        outputs = trained_model.output["output"]
    else:
        outputs = trained_model.output
    
    return keras.Model(inputs, outputs, name="mcunet_inference")


class EarlyExitLoss(keras.losses.Loss):
    """
    Combined loss for early exit training.
    
    Weights early exit loss lower to encourage using full network
    when accuracy matters, but allow early exit when confident.
    """
    
    def __init__(
        self,
        early_weight: float = 0.3,
        final_weight: float = 1.0,
        name: str = "early_exit_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.early_weight = early_weight
        self.final_weight = final_weight
        self.ce_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    def call(self, y_true, y_pred):
        """
        y_pred is dict with 'early_exit' and 'output' keys.
        """
        if isinstance(y_pred, dict):
            early_loss = self.ce_loss(y_true, y_pred["early_exit"])
            final_loss = self.ce_loss(y_true, y_pred["output"])
            return self.early_weight * early_loss + self.final_weight * final_loss
        else:
            return self.ce_loss(y_true, y_pred)


def get_early_exit_metrics():
    """Get metrics for early exit model training."""
    return {
        "output": keras.metrics.SparseCategoricalAccuracy(name="final_acc"),
        "early_exit": keras.metrics.SparseCategoricalAccuracy(name="early_acc")
    }


if __name__ == "__main__":
    # Test model creation
    print("Testing MCUNet Tiny models...")
    
    # With early exit
    model_ee = mcunet_tiny(with_early_exit=True)
    model_ee.summary()
    
    # Test forward pass
    dummy = np.random.rand(1, 28, 28, 1).astype(np.float32)
    outputs = model_ee(dummy)
    print(f"\nEarly exit output shape: {outputs['early_exit'].shape}")
    print(f"Final output shape: {outputs['output'].shape}")
    
    # Without early exit
    print("\n" + "="*50)
    model = mcunet_tiny(with_early_exit=False)
    model.summary()
    
    # Count parameters
    params = model.count_params()
    print(f"\nTotal parameters: {params}")
    print(f"Estimated INT8 size: ~{params / 1024:.1f} KB")
