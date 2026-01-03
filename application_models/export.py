"""
export.py - Export trained models to TFLite INT8 for RPI Pico

Outputs:
- .tflite file (INT8 quantized)
- .h C header file for embedding
- Model metadata JSON
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

sys.path.insert(0, str(Path(__file__).parent))
from data.data_loader import load_combined_dataset

# Import custom layers for model loading
from models.mcunet_tiny import EarlyExitBranch


def load_representative_data(samples_dir=None, num_samples=200):
    """Load representative data for quantization calibration."""
    data = load_combined_dataset(
        samples_dir=samples_dir,
        augment_custom=False,
        mnist_train_samples=1000
    )
    
    train_x = data["train"][0][:num_samples]
    train_x = np.expand_dims(train_x, -1).astype(np.float32)
    
    def representative_dataset():
        for i in range(len(train_x)):
            yield [train_x[i:i+1]]
    
    return representative_dataset


def strip_early_exit(model):
    """Convert early-exit model to single-output for deployment."""
    if not isinstance(model.output, dict):
        return model
    
    # Get only the final output
    inputs = model.input
    outputs = model.output["output"]
    
    return keras.Model(inputs, outputs, name=f"{model.name}_inference")


def convert_to_tflite_int8(
    model,
    representative_dataset,
    output_path: Path,
    optimize_for_size: bool = True
):
    """Convert Keras model to INT8 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # INT8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8
    
    # Representative dataset for calibration
    converter.representative_dataset = representative_dataset
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    output_path.write_bytes(tflite_model)
    
    return len(tflite_model)


def convert_to_tflite_float(model, output_path: Path):
    """Convert to float TFLite (for comparison)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return len(tflite_model)


def generate_c_header(tflite_path: Path, output_path: Path, array_name: str = "model_data"):
    """Generate C header file from TFLite model."""
    data = tflite_path.read_bytes()
    
    header = f"""// Auto-generated TFLite model header
// Generated: {datetime.now().isoformat()}
// Model: {tflite_path.name}
// Size: {len(data)} bytes

#ifndef {array_name.upper()}_H
#define {array_name.upper()}_H

#include <stdint.h>

const unsigned int {array_name}_len = {len(data)};

alignas(8) const unsigned char {array_name}[] = {{
"""
    
    # Convert bytes to C array
    bytes_per_line = 12
    lines = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i + bytes_per_line]
        hex_values = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"    {hex_values},")
    
    header += "\n".join(lines)
    header += "\n};\n\n#endif  // " + array_name.upper() + "_H\n"
    
    output_path.write_text(header)


def evaluate_tflite(tflite_path: Path, test_data, is_int8: bool = True):
    """Evaluate TFLite model accuracy."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_idx = input_details[0]["index"]
    output_idx = output_details[0]["index"]
    input_dtype = input_details[0]["dtype"]
    
    test_x, test_y = test_data
    if len(test_x.shape) == 3:
        test_x = np.expand_dims(test_x, -1)
    
    correct = 0
    total = len(test_x)
    
    for i in range(total):
        sample = test_x[i:i+1]
        
        # Convert to correct dtype
        if input_dtype == np.uint8:
            sample = (sample * 255).astype(np.uint8)
        else:
            sample = sample.astype(np.float32)
        
        interpreter.set_tensor(input_idx, sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_idx)
        
        pred = np.argmax(output)
        if pred == test_y[i]:
            correct += 1
    
    accuracy = correct / total
    return accuracy


def measure_inference_time(tflite_path: Path, runs: int = 100):
    """Measure TFLite inference time."""
    import time
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    input_idx = input_details[0]["index"]
    input_dtype = input_details[0]["dtype"]
    
    # Dummy input
    if input_dtype == np.uint8:
        dummy = np.random.randint(0, 256, (1, 28, 28, 1), dtype=np.uint8)
    else:
        dummy = np.random.rand(1, 28, 28, 1).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_idx, dummy)
        interpreter.invoke()
    
    # Measure
    start = time.time()
    for _ in range(runs):
        interpreter.set_tensor(input_idx, dummy)
        interpreter.invoke()
    elapsed = time.time() - start
    
    return (elapsed / runs) * 1000  # ms


def export_model(
    model_path: str,
    output_dir: str,
    samples_dir: str = None,
    model_name: str = None
):
    """Main export function."""
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if model_name is None:
        model_name = model_path.stem
    
    print("="*60)
    print("Model Export to TFLite INT8")
    print("="*60)
    print(f"Input: {model_path}")
    print(f"Output: {output_dir}")
    
    # Load model
    print("\n--- Loading Model ---")
    model = keras.models.load_model(
        model_path, 
        compile=False,
        custom_objects={"EarlyExitBranch": EarlyExitBranch}
    )
    model.summary()
    
    # Strip early exit if present
    if isinstance(model.output, dict):
        print("\nStripping early exit for inference model...")
        model = strip_early_exit(model)
        model.summary()
    
    # Load data
    print("\n--- Loading Data ---")
    samples_path = Path(samples_dir) if samples_dir else None
    data = load_combined_dataset(samples_path, augment_custom=False)
    rep_dataset = load_representative_data(samples_path)
    
    # Export float TFLite
    print("\n--- Exporting Float TFLite ---")
    float_path = output_dir / f"{model_name}_float.tflite"
    float_size = convert_to_tflite_float(model, float_path)
    print(f"Float TFLite: {float_path} ({float_size / 1024:.1f} KB)")
    
    # Export INT8 TFLite
    print("\n--- Exporting INT8 TFLite ---")
    int8_path = output_dir / f"{model_name}_int8.tflite"
    int8_size = convert_to_tflite_int8(model, rep_dataset, int8_path)
    print(f"INT8 TFLite: {int8_path} ({int8_size / 1024:.1f} KB)")
    
    # Generate C header
    print("\n--- Generating C Header ---")
    header_path = output_dir / f"{model_name}_model.h"
    generate_c_header(int8_path, header_path, f"{model_name}_model")
    print(f"C Header: {header_path}")
    
    # Evaluate accuracy
    print("\n--- Evaluating Models ---")
    test_data = data["test"]
    
    float_acc = evaluate_tflite(float_path, test_data, is_int8=False)
    print(f"Float TFLite Accuracy: {float_acc:.4f}")
    
    int8_acc = evaluate_tflite(int8_path, test_data, is_int8=True)
    print(f"INT8 TFLite Accuracy: {int8_acc:.4f}")
    
    # Measure inference time
    float_ms = measure_inference_time(float_path)
    int8_ms = measure_inference_time(int8_path)
    
    print(f"\nFloat inference: {float_ms:.3f} ms")
    print(f"INT8 inference: {int8_ms:.3f} ms")
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "keras_params": model.count_params(),
        "float_tflite_size": float_size,
        "int8_tflite_size": int8_size,
        "float_accuracy": float(float_acc),
        "int8_accuracy": float(int8_acc),
        "float_inference_ms": float(float_ms),
        "int8_inference_ms": float(int8_ms),
        "export_date": datetime.now().isoformat()
    }
    
    meta_path = output_dir / f"{model_name}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    print(f"Float size:    {float_size / 1024:.1f} KB")
    print(f"INT8 size:     {int8_size / 1024:.1f} KB")
    print(f"Size reduction: {(1 - int8_size/float_size) * 100:.1f}%")
    print(f"Float acc:     {float_acc:.4f}")
    print(f"INT8 acc:      {int8_acc:.4f}")
    print(f"Accuracy drop: {(float_acc - int8_acc) * 100:.2f}%")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Export model to TFLite INT8")
    parser.add_argument("model", type=str, help="Path to trained Keras model")
    parser.add_argument("--output-dir", type=str, default="exports", help="Output directory")
    parser.add_argument("--samples-dir", type=str, default="../samples", help="Custom samples")
    parser.add_argument("--name", type=str, default=None, help="Model name for outputs")
    
    args = parser.parse_args()
    
    export_model(
        model_path=args.model,
        output_dir=args.output_dir,
        samples_dir=args.samples_dir,
        model_name=args.name
    )


if __name__ == "__main__":
    main()
