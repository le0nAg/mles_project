"""
train.py - Training pipeline for PIN digit recognition

Features:
- Quantization-aware training (QAT)
- Early exit support
- Combined MNIST + custom samples
- Hyperparameter configuration
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import load_combined_dataset, create_tf_dataset
from models.mcunet_tiny import mcunet_tiny, EarlyExitLoss
from models.tflm_cnn import tflm_cnn_tiny, tflm_cnn_small, tflm_cnn_with_early_exit

# Try to import QAT toolkit
try:
    import tensorflow_model_optimization as tfmot
    HAS_TFMOT = True
except ImportError:
    HAS_TFMOT = False
    print("Warning: tensorflow-model-optimization not installed. QAT disabled.")


# Default configuration
DEFAULT_CONFIG = {
    "model": "mcunet_tiny",  # mcunet_tiny, tflm_tiny, tflm_small
    "early_exit": True,
    "qat": True,
    "epochs": 15,
    "batch_size": 64,
    "learning_rate": 0.001,
    "lr_schedule": "cosine",  # constant, step, cosine
    "optimizer": "adam",
    "early_exit_weight": 0.3,
    "label_smoothing": 0.1,
    "samples_dir": "../samples",
    "augment_custom": True,
    "augmentations_per_sample": 10,
    "custom_weight": 3.0,
    "output_dir": "checkpoints"
}


def get_lr_schedule(config, steps_per_epoch):
    """Create learning rate schedule."""
    initial_lr = config["learning_rate"]
    total_steps = config["epochs"] * steps_per_epoch
    
    if config["lr_schedule"] == "cosine":
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps,
            alpha=0.01  # Minimum LR as fraction of initial
        )
    elif config["lr_schedule"] == "step":
        boundaries = [int(total_steps * 0.5), int(total_steps * 0.75)]
        values = [initial_lr, initial_lr * 0.1, initial_lr * 0.01]
        return keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    else:
        return initial_lr


def get_optimizer(config, lr_schedule):
    """Create optimizer."""
    if config["optimizer"] == "adam":
        return keras.optimizers.Adam(learning_rate=lr_schedule)
    elif config["optimizer"] == "sgd":
        return keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    elif config["optimizer"] == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr_schedule)
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")


def create_model(config):
    """Create model based on config."""
    model_name = config["model"]
    early_exit = config["early_exit"]
    
    if model_name == "mcunet_tiny":
        model = mcunet_tiny(with_early_exit=early_exit)
    elif model_name == "tflm_tiny":
        if early_exit:
            model = tflm_cnn_with_early_exit()
        else:
            model = tflm_cnn_tiny()
    elif model_name == "tflm_small":
        if early_exit:
            model = tflm_cnn_with_early_exit()
        else:
            model = tflm_cnn_small()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def apply_qat(model, config):
    """Apply quantization-aware training if enabled and available."""
    if not config["qat"]:
        return model, False
    
    if not HAS_TFMOT:
        print("QAT requested but tensorflow-model-optimization not available")
        return model, False
    
    # Check if model has dict outputs (early exit)
    if isinstance(model.output, dict):
        print("Note: QAT with dict outputs - quantization applied during export")
        return model, False
    
    try:
        qat_model = tfmot.quantization.keras.quantize_model(model)
        print("QAT applied successfully")
        return qat_model, True
    except Exception as e:
        print(f"QAT failed: {e}")
        return model, False


def get_loss_and_metrics(config):
    """Create loss function and metrics."""
    if config["early_exit"]:
        # For multi-output models, use dict of losses
        loss = {
            "output": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "early_exit": keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        }
        loss_weights = {
            "output": 1.0,
            "early_exit": config["early_exit_weight"]
        }
        metrics = {
            "output": keras.metrics.SparseCategoricalAccuracy(name="acc"),
            "early_exit": keras.metrics.SparseCategoricalAccuracy(name="acc")
        }
        return loss, metrics, loss_weights
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
        return loss, metrics, None


def train(config):
    """Main training function."""
    print("="*60)
    print("PIN Digit Recognition - Model Training")
    print("="*60)
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Setup paths
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples_dir = Path(config["samples_dir"])
    if not samples_dir.exists():
        print(f"Samples directory not found: {samples_dir}")
        samples_dir = None
    
    # Load data
    print("\n--- Loading Data ---")
    data = load_combined_dataset(
        samples_dir=samples_dir,
        augment_custom=config["augment_custom"],
        augmentations_per_sample=config["augmentations_per_sample"],
        custom_weight=config["custom_weight"]
    )
    
    train_x, train_y = data["train"]
    val_x, val_y = data["val"]
    test_x, test_y = data["test"]
    
    # Add channel dimension
    train_x = np.expand_dims(train_x, -1)
    val_x = np.expand_dims(val_x, -1)
    test_x = np.expand_dims(test_x, -1)
    
    print(f"Train: {train_x.shape}, Val: {val_x.shape}, Test: {test_x.shape}")
    
    # Create datasets
    if config["early_exit"]:
        # Multi-output model needs labels for each output
        train_ds = tf.data.Dataset.from_tensor_slices(
            (train_x, {"output": train_y, "early_exit": train_y})
        )
        val_ds = tf.data.Dataset.from_tensor_slices(
            (val_x, {"output": val_y, "early_exit": val_y})
        )
        test_ds = tf.data.Dataset.from_tensor_slices(
            (test_x, {"output": test_y, "early_exit": test_y})
        )
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    
    train_ds = train_ds.shuffle(10000).batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(config["batch_size"])
    
    steps_per_epoch = len(train_x) // config["batch_size"]
    
    # Create model
    print("\n--- Creating Model ---")
    model = create_model(config)
    model.summary()
    
    # Apply QAT
    model, qat_applied = apply_qat(model, config)
    
    # Compile
    lr_schedule = get_lr_schedule(config, steps_per_epoch)
    optimizer = get_optimizer(config, lr_schedule)
    loss, metrics, loss_weights = get_loss_and_metrics(config)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config['model']}_{'ee' if config['early_exit'] else 'no_ee'}"
    
    # Monitor metric depends on model type
    if config["early_exit"]:
        monitor_metric = "val_output_acc"
        monitor_mode = "max"
    else:
        monitor_metric = "val_loss"
        monitor_mode = "min"
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"{model_name}_best.keras"),
            monitor=monitor_metric,
            save_best_only=True,
            mode=monitor_mode
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=5,
            restore_best_weights=True,
            mode=monitor_mode
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train
    print("\n--- Training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n--- Evaluation ---")
    test_results = model.evaluate(test_ds, verbose=1)
    
    if config["early_exit"]:
        print(f"Test results: {dict(zip(model.metrics_names, test_results))}")
    else:
        print(f"Test Loss: {test_results[0]:.4f}, Test Accuracy: {test_results[1]:.4f}")
    
    # Save final model
    final_path = output_dir / f"{model_name}_final.keras"
    model.save(final_path)
    print(f"\nSaved model to: {final_path}")
    
    # Save training history
    history_path = output_dir / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    
    # Save config
    config_path = output_dir / f"{model_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Model params: {model.count_params()}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train PIN digit recognition model")
    parser.add_argument("--model", choices=["mcunet_tiny", "tflm_tiny", "tflm_small"],
                        default="mcunet_tiny", help="Model architecture")
    parser.add_argument("--no-early-exit", action="store_true", help="Disable early exit")
    parser.add_argument("--no-qat", action="store_true", help="Disable QAT")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--samples-dir", type=str, default="../samples", help="Custom samples dir")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config["model"] = args.model
    config["early_exit"] = not args.no_early_exit
    config["qat"] = not args.no_qat
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["samples_dir"] = args.samples_dir
    config["output_dir"] = args.output_dir
    
    train(config)


if __name__ == "__main__":
    main()
