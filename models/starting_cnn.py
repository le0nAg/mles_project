# mles_project/models/cnn.py
import os
import time
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# -----------------------------
# Prepare dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
print("Dataset loaded.")

# -----------------------------
# Build YOUR baseline model
# -----------------------------
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(80, activation='elu'),
    keras.layers.Dense(60, activation='elu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)  # logits
])

model.summary()

# -----------------------------
# Initial test (untrained)
# -----------------------------
predictions = model(x_train[:1]).numpy()
predictions_prob = tf.nn.softmax(predictions).numpy()
print('Probabilities (untrained):', predictions_prob)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
initial_loss = loss_fn(y_train[:1], predictions).numpy()
print('Initial loss (untrained):', initial_loss)

# -----------------------------
# Train model
# -----------------------------
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# -----------------------------
# Helper: measure inference speed for Keras
# -----------------------------
def measure_inference_speed_keras(model, runs=200):
    dummy = np.random.rand(1, 28, 28).astype('float32')

    # warm-up
    for _ in range(10):
        _ = model.predict(dummy, verbose=0)

    t0 = time.time()
    for _ in range(runs):
        _ = model.predict(dummy, verbose=0)
    t1 = time.time()

    avg_ms = (t1 - t0) / runs * 1000
    return avg_ms

keras_ms = measure_inference_speed_keras(model)
print(f"Keras inference speed: {keras_ms:.3f} ms/sample")

# -----------------------------
# Save Keras model
# -----------------------------
out = Path("out_fc_baseline")
out.mkdir(exist_ok=True)

keras_path = out / "baseline_fc_model.h5"
model.save(keras_path)
keras_size = keras_path.stat().st_size
print(f"Saved Keras model: {keras_path} ({keras_size} bytes)")

# -----------------------------
# Convert to TFLite
# -----------------------------
def convert_to_tflite(model, output_dir:Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = output_dir / "model.tflite"
    tflite_path.write_bytes(tflite_model)
    return tflite_path, tflite_path.stat().st_size

tflite_path, tflite_size = convert_to_tflite(model, out)
print(f"Saved TFLite model: {tflite_path} ({tflite_size} bytes)")

# -----------------------------
# Measure TFLite inference speed
# -----------------------------
def measure_tflite_inference_speed(tflite_path, runs=200):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy = np.random.rand(1, 28, 28).astype('float32')

    # warm-up
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()

    t0 = time.time()
    for _ in range(runs):
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
    t1 = time.time()

    avg_ms = (t1 - t0) / runs * 1000
    return avg_ms

tflite_ms = measure_tflite_inference_speed(tflite_path)
print(f"TFLite inference speed: {tflite_ms:.3f} ms/sample")

# -----------------------------
# Save summary for Step 3
# -----------------------------
summary = out / "summary.csv"
with open(summary, 'w') as f:
    f.write("keras_size,tflite_size,test_acc,keras_ms,tflite_ms\n")
    f.write(f"{keras_size},{tflite_size},{test_acc:.6f},{keras_ms:.6f},{tflite_ms:.6f}")

print(f"Summary written to {summary}")
