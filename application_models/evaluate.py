"""
evaluate.py - Evaluate and benchmark trained models

Compares:
- Model accuracy on MNIST and custom samples
- Inference time benchmarks
- Memory footprint analysis
- Early exit effectiveness
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent))
from data.data_loader import load_combined_dataset, load_custom_samples


def evaluate_keras_model(model, test_x, test_y, batch_size=32):
    """Evaluate Keras model accuracy."""
    if len(test_x.shape) == 3:
        test_x = np.expand_dims(test_x, -1)
    
    predictions = model.predict(test_x, batch_size=batch_size, verbose=0)
    
    # Handle early exit models
    if isinstance(predictions, dict):
        final_preds = np.argmax(predictions["output"], axis=1)
        early_preds = np.argmax(predictions["early_exit"], axis=1)
        
        final_acc = np.mean(final_preds == test_y)
        early_acc = np.mean(early_preds == test_y)
        
        return {"final_accuracy": final_acc, "early_accuracy": early_acc}
    else:
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": np.mean(preds == test_y)}


def evaluate_per_digit(model, test_x, test_y, batch_size=32):
    """Evaluate accuracy per digit class."""
    if len(test_x.shape) == 3:
        test_x = np.expand_dims(test_x, -1)
    
    predictions = model.predict(test_x, batch_size=batch_size, verbose=0)
    
    if isinstance(predictions, dict):
        predictions = predictions["output"]
    
    preds = np.argmax(predictions, axis=1)
    
    results = {}
    for digit in range(10):
        mask = test_y == digit
        if np.sum(mask) > 0:
            digit_acc = np.mean(preds[mask] == test_y[mask])
            results[digit] = {
                "accuracy": float(digit_acc),
                "samples": int(np.sum(mask))
            }
    
    return results


def benchmark_inference(model, input_shape=(28, 28, 1), runs=200, warmup=20):
    """Benchmark Keras model inference time."""
    dummy = np.random.rand(1, *input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        _ = model.predict(dummy, verbose=0)
    
    # Benchmark
    start = time.time()
    for _ in range(runs):
        _ = model.predict(dummy, verbose=0)
    elapsed = time.time() - start
    
    return (elapsed / runs) * 1000  # ms


def benchmark_tflite_inference(tflite_path, runs=200, warmup=20):
    """Benchmark TFLite model inference time."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    input_idx = input_details[0]["index"]
    input_dtype = input_details[0]["dtype"]
    input_shape = input_details[0]["shape"][1:]
    
    if input_dtype == np.uint8:
        dummy = np.random.randint(0, 256, (1, *input_shape), dtype=np.uint8)
    else:
        dummy = np.random.rand(1, *input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_idx, dummy)
        interpreter.invoke()
    
    # Benchmark
    start = time.time()
    for _ in range(runs):
        interpreter.set_tensor(input_idx, dummy)
        interpreter.invoke()
    elapsed = time.time() - start
    
    return (elapsed / runs) * 1000  # ms


def evaluate_early_exit_effectiveness(model, test_x, test_y, confidence_threshold=0.9):
    """
    Evaluate early exit effectiveness.
    
    Returns stats on how often early exit is sufficient.
    """
    if len(test_x.shape) == 3:
        test_x = np.expand_dims(test_x, -1)
    
    predictions = model.predict(test_x, batch_size=32, verbose=0)
    
    if not isinstance(predictions, dict):
        return {"early_exit_available": False}
    
    early_logits = predictions["early_exit"]
    final_logits = predictions["output"]
    
    # Convert to probabilities
    early_probs = tf.nn.softmax(early_logits).numpy()
    final_probs = tf.nn.softmax(final_logits).numpy()
    
    early_confidence = np.max(early_probs, axis=1)
    early_preds = np.argmax(early_probs, axis=1)
    final_preds = np.argmax(final_probs, axis=1)
    
    # How often early exit is confident enough
    early_exit_mask = early_confidence >= confidence_threshold
    early_exit_rate = np.mean(early_exit_mask)
    
    # Accuracy when taking early exit
    early_exit_correct = np.sum((early_preds == test_y) & early_exit_mask)
    early_exit_acc = early_exit_correct / np.sum(early_exit_mask) if np.sum(early_exit_mask) > 0 else 0
    
    # Combined accuracy (early when confident, final otherwise)
    combined_preds = np.where(early_exit_mask, early_preds, final_preds)
    combined_acc = np.mean(combined_preds == test_y)
    
    # Final-only accuracy
    final_acc = np.mean(final_preds == test_y)
    
    return {
        "early_exit_available": True,
        "confidence_threshold": confidence_threshold,
        "early_exit_rate": float(early_exit_rate),
        "early_exit_accuracy": float(early_exit_acc),
        "final_accuracy": float(final_acc),
        "combined_accuracy": float(combined_acc),
        "potential_speedup": f"{1/(1-early_exit_rate*0.4):.2f}x" if early_exit_rate > 0 else "1.0x"
    }


def compare_models(model_paths: list, samples_dir=None, output_path=None):
    """Compare multiple models."""
    print("="*60)
    print("Model Comparison")
    print("="*60)
    
    # Load data
    data = load_combined_dataset(
        Path(samples_dir) if samples_dir else None,
        augment_custom=False
    )
    test_x, test_y = data["test"]
    
    # Load custom samples separately for focused evaluation
    custom_x, custom_y = None, None
    if samples_dir and Path(samples_dir).exists():
        custom_x, custom_y = load_custom_samples(Path(samples_dir))
    
    results = []
    
    for model_path in model_paths:
        model_path = Path(model_path)
        print(f"\n--- {model_path.name} ---")
        
        if model_path.suffix == ".tflite":
            # TFLite evaluation
            size = model_path.stat().st_size
            inference_ms = benchmark_tflite_inference(model_path)
            
            result = {
                "model": model_path.name,
                "type": "tflite",
                "size_bytes": size,
                "inference_ms": inference_ms,
            }
        else:
            # Keras evaluation
            model = keras.models.load_model(model_path, compile=False)
            
            acc_results = evaluate_keras_model(model, test_x, test_y)
            per_digit = evaluate_per_digit(model, test_x, test_y)
            inference_ms = benchmark_inference(model)
            early_exit = evaluate_early_exit_effectiveness(model, test_x, test_y)
            
            result = {
                "model": model_path.name,
                "type": "keras",
                "params": model.count_params(),
                "inference_ms": inference_ms,
                **acc_results,
                "early_exit": early_exit
            }
            
            # Custom samples accuracy
            if custom_x is not None:
                custom_acc = evaluate_keras_model(model, custom_x, custom_y)
                result["custom_samples"] = custom_acc
        
        results.append(result)
        print(json.dumps(result, indent=2))
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("models", nargs="+", help="Model paths to evaluate")
    parser.add_argument("--samples-dir", type=str, default="../samples")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    
    args = parser.parse_args()
    
    compare_models(
        model_paths=args.models,
        samples_dir=args.samples_dir,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
