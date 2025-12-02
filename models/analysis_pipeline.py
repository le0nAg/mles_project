"""
analysis_pipeline.py

Run experiments for:
- baseline FC (your original architecture)
- baseline CNN
- optimized CNN (smaller)
- SqueezeNet (small implementation)
- MobileNetV2 (1-channel adapted)
- MCUNet proxy (tiny CNN)

Outputs:
- out/results.csv
- out/plots/optimized_vs_baseline.png
- out/plots/architectures_mem_vs_speed.png

Usage:
    python analysis_pipeline.py
"""

import os
import time
import math
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

OUT = Path("out_analysis")
PLOTS = OUT / "plots"
OUT.mkdir(exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def prepare_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    # validation split 10% from training
    val_idx = int(len(x_train) * 0.9)
    x_val, y_val = x_train[val_idx:], y_train[val_idx:]
    x_train, y_train = x_train[:val_idx], y_train[:val_idx]
    # Expand channel dim for conv models
    x_train_c = np.expand_dims(x_train, -1)
    x_val_c = np.expand_dims(x_val, -1)
    x_test_c = np.expand_dims(x_test, -1)
    return (x_train, y_train, x_train_c), (x_val, y_val, x_val_c), (x_test, y_test, x_test_c)

def save_keras_and_get_size(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path, include_optimizer=False)
    return path.stat().st_size

def convert_tflite_and_get_size(model, path: Path, quantize=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tflite)
    return path.stat().st_size

def measure_keras_inference_ms(model, input_shape, runs=200, batch_size=1):
    dummy = np.random.rand(batch_size, *input_shape).astype("float32")
    for _ in range(10):
        model.predict(dummy, verbose=0)
    t0 = time.time()
    for _ in range(runs):
        model.predict(dummy, verbose=0)
    t1 = time.time()
    return (t1 - t0) / runs * 1000.0 / batch_size

def measure_tflite_inference_ms(tflite_path, input_shape, runs=200):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    idx = input_details[0]['index']
    dtype = input_details[0]['dtype']
    # create dummy respecting dtype and shape (batch=1)
    dummy = (np.random.rand(1, *input_shape) * 255).astype(dtype) if dtype == np.uint8 else np.random.rand(1, *input_shape).astype(dtype)
    for _ in range(10):
        interpreter.set_tensor(idx, dummy)
        interpreter.invoke()
    t0 = time.time()
    for _ in range(runs):
        interpreter.set_tensor(idx, dummy)
        interpreter.invoke()
    t1 = time.time()
    return (t1 - t0) / runs * 1000.0

# -------------------------
# Model factories
# -------------------------
def fc_baseline(input_shape=(28,28), num_classes=10):
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(80, activation='elu'),
        layers.Dense(60, activation='elu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes)  # logits
    ])
    return model

def simple_cnn(input_shape=(28,28,1), num_classes=10, activation='relu', dropout=0.0):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Flatten()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(128)(x)
    x = layers.Activation(activation)(x)
    out = layers.Dense(num_classes)(x)
    return keras.Model(inp, out, name='simple_cnn')

def optimized_cnn(input_shape=(28,28,1), num_classes=10, activation='relu', dropout=0.0):
    # smaller variant: fewer filters and smaller dense
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Flatten()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(64)(x)
    x = layers.Activation(activation)(x)
    out = layers.Dense(num_classes)(x)
    return keras.Model(inp, out, name='optimized_cnn')

def squeezenet_small(input_shape=(28,28,1), num_classes=10):
    # tiny SqueezeNet-like architecture (not full SqueezeNet, but uses fire modules)
    def fire_module(x, squeeze, expand):
        x = layers.Conv2D(squeeze, 1, activation='relu', padding='same')(x)
        left = layers.Conv2D(expand, 1, activation='relu', padding='same')(x)
        right = layers.Conv2D(expand, 3, activation='relu', padding='same')(x)
        return layers.Concatenate()([left, right])

    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, strides=1, padding='same', activation='relu')(inp)
    x = fire_module(x, 8, 16)
    x = layers.MaxPool2D(2)(x)
    x = fire_module(x, 8, 16)
    x = layers.MaxPool2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(num_classes)(x)
    return keras.Model(inp, out, name='squeezenet_small')

def mobilenet_v2_small(input_shape=(28,28,1), num_classes=10):
    # MobileNetV2 requires at least 32x32 → upscale MNIST
    inp = keras.Input(shape=input_shape)

    # upscale to 32x32
    x = layers.Resizing(32, 32, interpolation="bilinear")(inp)

    # replicate channel to 3
    x = layers.Concatenate()([x, x, x])

    base = keras.applications.MobileNetV2(
        input_shape=(32,32,3),
        include_top=False,
        weights=None,
        pooling='avg'
    )

    x = base(x)
    out = layers.Dense(num_classes)(x)
    return keras.Model(inp, out, name='mobilenet_v2_small')


def mcunet_proxy(input_shape=(28,28,1), num_classes=10):
    # ultra tiny model as proxy for MCUNet tiny configs
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(8, 3, padding='same', activation='relu')(inp)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(num_classes)(x)
    return keras.Model(inp, out, name='mcunet_proxy')

# -------------------------
# Experiment settings
# -------------------------
optimizers = {
    'adam': lambda lr: keras.optimizers.Adam(learning_rate=lr),
    'sgd': lambda lr: keras.optimizers.SGD(learning_rate=lr),
    'rmsprop': lambda lr: keras.optimizers.RMSprop(learning_rate=lr)
}
activations = ['relu', 'elu', 'swish']  # swish is available in TF2
epochs_list = [5, 10, 20]
lrs = [0.001, 0.01, 0.05]
batch_size = 128

# NOTE:
# The full grid (3*3*3*3) can be very long. We'll run the "per-dimension" experiments you asked:
# 1) 3 optimizers (same baseline lr/epochs/activation)
# 2) 3 activations (same baseline optimizer/lr/epochs)
# 3) 3 epochs (same baseline optimizer/activation/lr)
# 4) 3 lrs (same baseline optimizer/activation/epochs)
#
# Baseline defaults:
base_optimizer = 'adam'
base_activation = 'relu'
base_epochs = 10
base_lr = 0.01

# -------------------------
# Data
# -------------------------
(x_train_flat, y_train, x_train_c), (x_val_flat, y_val, x_val_c), (x_test_flat, y_test, x_test_c) = prepare_mnist()
print("Shapes:", x_train_flat.shape, x_val_flat.shape, x_test_flat.shape)

# -------------------------
# Helper: train / evaluate model and collect metrics
# -------------------------
def run_and_measure(model_builder, name, input_type='conv', optimizer='adam', lr=0.01,
                    activation='relu', epochs=10, dropout=0.0, batch_size=128, quantize=False):
    print(f"\n--- Running {name} | opt={optimizer} lr={lr} act={activation} epochs={epochs} ---")
    if input_type == 'conv':
        x_tr, x_va, x_te = x_train_c, x_val_c, x_test_c
        input_shape = (28,28,1)
    else:
        x_tr, x_va, x_te = x_train_flat, x_val_flat, x_test_flat
        input_shape = (28,28)
    model = model_builder() if callable(model_builder) and model_builder.__code__.co_argcount==0 else model_builder(input_shape=input_shape, activation=activation, dropout=dropout)
    opt = optimizers[optimizer](lr)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
    history = model.fit(x_tr, y_train, validation_data=(x_va, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    test_loss, test_acc = model.evaluate(x_te, y_test, verbose=0)
    keras_ms = measure_keras_inference_ms(model, input_shape if input_type=='conv' else (28,28), runs=200, batch_size=1)

    # save keras model (temp path)
    keras_path = OUT / f"{name}.h5"
    if keras_path.exists():
        keras_path.unlink()
    s_keras = save_keras_and_get_size(model, keras_path)

    # convert to tflite
    tflite_path = OUT / f"{name}.tflite"
    if tflite_path.exists():
        tflite_path.unlink()
    s_tflite = convert_tflite_and_get_size(model, tflite_path, quantize=quantize)
    tflite_ms = measure_tflite_inference_ms(tflite_path, input_shape if input_type=='conv' else (28,28))
    # params count
    params = model.count_params()

    result = {
        'name': name,
        'params': params,
        'keras_size_bytes': s_keras,
        'tflite_size_bytes': s_tflite,
        'test_accuracy': float(test_acc),
        'keras_ms': float(keras_ms),
        'tflite_ms': float(tflite_ms),
        'history': history.history
    }
    print(f"Result {name}: acc={test_acc:.4f} params={params} keras_size={s_keras} tflite_size={s_tflite} keras_ms={keras_ms:.3f} tflite_ms={tflite_ms:.3f}")
    return result

# -------------------------
# Phase A: Baseline runs (FC baseline + CNN baseline + optimized CNN)
# -------------------------
results = []

# FC baseline (your original)
res_fc = run_and_measure(lambda: fc_baseline(), name='fc_baseline', input_type='flat', optimizer=base_optimizer, lr=base_lr, activation=base_activation, epochs=base_epochs, batch_size=batch_size)
results.append(res_fc)

# CNN baseline
res_cnn = run_and_measure(lambda: simple_cnn(), name='cnn_baseline', input_type='conv', optimizer=base_optimizer, lr=base_lr, activation=base_activation, epochs=base_epochs, batch_size=batch_size)
results.append(res_cnn)

# Optimized CNN
res_opt = run_and_measure(lambda: optimized_cnn(), name='cnn_optimized', input_type='conv', optimizer=base_optimizer, lr=base_lr, activation=base_activation, epochs=base_epochs, batch_size=batch_size)
results.append(res_opt)

# -------------------------
# Phase B: Optimizer comparison on baseline CNN (3 optimizers)
# -------------------------
for opt_name in ['adam', 'sgd', 'rmsprop']:
    r = run_and_measure(lambda: simple_cnn(), name=f'cnn_opt_{opt_name}', input_type='conv', optimizer=opt_name, lr=base_lr, activation=base_activation, epochs=base_epochs, batch_size=batch_size)
    results.append(r)

# -------------------------
# Phase C: Activation functions comparison on baseline CNN
# -------------------------
for act in activations:
    r = run_and_measure(lambda: simple_cnn(), name=f'cnn_act_{act}', input_type='conv', optimizer=base_optimizer, lr=base_lr, activation=act, epochs=base_epochs, batch_size=batch_size)
    results.append(r)

# -------------------------
# Phase D: Epochs comparison on baseline CNN
# -------------------------
for e in epochs_list:
    r = run_and_measure(lambda: simple_cnn(), name=f'cnn_epochs_{e}', input_type='conv', optimizer=base_optimizer, lr=base_lr, activation=base_activation, epochs=e, batch_size=batch_size)
    results.append(r)

# -------------------------
# Phase E: Learning rate comparison on baseline CNN
# -------------------------
for lr in lrs:
    r = run_and_measure(lambda: simple_cnn(), name=f'cnn_lr_{lr}', input_type='conv', optimizer=base_optimizer, lr=lr, activation=base_activation, epochs=base_epochs, batch_size=batch_size)
    results.append(r)

# -------------------------
# Phase F: Other architectures (SqueezeNet, MobileNetV2, MCUNet proxy)
# -------------------------
res_sqn = run_and_measure(lambda: squeezenet_small(), name='squeezenet_small', input_type='conv', optimizer=base_optimizer, lr=base_lr, activation=base_activation, epochs=base_epochs, batch_size=batch_size)
results.append(res_sqn)

res_mn = run_and_measure(lambda: mobilenet_v2_small(), name='mobilenet_v2', input_type='conv', optimizer=base_optimizer, lr=base_lr, activation=base_activation, epochs=base_epochs, batch_size=batch_size)
results.append(res_mn)

res_mc = run_and_measure(lambda: mcunet_proxy(), name='mcunet_proxy', input_type='conv', optimizer=base_optimizer, lr=base_lr, activation=base_activation, epochs=base_epochs, batch_size=batch_size)
results.append(res_mc)

# -------------------------
# Persist results
# -------------------------
df_rows = []
for r in results:
    df_rows.append({
        'name': r['name'],
        'params': r['params'],
        'keras_size_bytes': r['keras_size_bytes'],
        'tflite_size_bytes': r['tflite_size_bytes'],
        'test_accuracy': r['test_accuracy'],
        'keras_ms': r['keras_ms'],
        'tflite_ms': r['tflite_ms']
    })

df = pd.DataFrame(df_rows)
csv_path = OUT / "results.csv"
df.to_csv(csv_path, index=False)
print(f"Saved results to {csv_path}")

# -------------------------------------
# Save CSVs for plotting (for the report)
# -------------------------------------

# Save baseline vs optimized CNN comparison data
df_baseline_vs_opt = df[df['name'].isin(['cnn_baseline', 'cnn_optimized'])]
df_baseline_vs_opt.to_csv(OUT / "plot_baseline_vs_optimized.csv", index=False)

# Save architecture-wide memory vs speed data
df_arch = df[df['name'].isin([
    'cnn_baseline',
    'cnn_optimized',
    'fc_baseline',
    'squeezenet_small',
    'mobilenet_v2',
    'mcunet_proxy'
])]
df_arch.to_csv(OUT / "plot_architectures_mem_vs_speed.csv", index=False)


# -------------------------
# Plot 1: optimized CNN vs baseline CNN
#  - bar chart: validation/test accuracy (test_accuracy)
#  - bar chart: keras_ms (inference)
# -------------------------
def plot_optimized_vs_baseline(df, outpath):
    names = ['cnn_baseline', 'cnn_optimized']
    sel = df[df['name'].isin(names)].set_index('name')

    # Save CSV for report reproducibility
    sel.to_csv(outpath.with_suffix(".csv"))

    fig, axes = plt.subplots(1,2, figsize=(10,4))

    axes[0].bar(sel.index, sel['test_accuracy'])
    axes[0].set_title('Test accuracy')
    axes[0].set_ylim(0,1)

    axes[1].bar(sel.index, sel['keras_ms'])
    axes[1].set_title('Keras inference time (ms/sample)')

    plt.suptitle('Optimized CNN vs Baseline CNN')
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close()


plot_optimized_vs_baseline(df, PLOTS / 'optimized_vs_baseline.png')
print(f"Wrote {PLOTS / 'optimized_vs_baseline.png'}")

# -------------------------
# Plot 2: memory size vs inference speed for all architectures
#  - x: tflite_size_bytes (kB)
#  - y: tflite_ms
#  - annotate with name and accuracy
# -------------------------
def plot_mem_vs_speed(df, outpath):
    df.to_csv(outpath.with_suffix(".csv"))

    fig, ax = plt.subplots(figsize=(8,6))
    x = df['tflite_size_bytes'] / 1024.0
    y = df['tflite_ms']
    sc = ax.scatter(x, y, s=60)

    ax.set_xlabel('TFLite Model Size (KB)')
    ax.set_ylabel('TFLite Inference Time (ms/sample)')
    ax.set_title('Model size vs inference speed')

    for i, row in df.iterrows():
        ax.annotate(
            f"{row['name']} ({row['test_accuracy']:.2f})",
            (x.iat[i], y.iat[i]),
            textcoords="offset points", xytext=(5,3),
            ha='left', fontsize=7
        )

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close()


plot_mem_vs_speed(df, PLOTS / 'architectures_mem_vs_speed.png')
print(f"Wrote {PLOTS / 'architectures_mem_vs_speed.png'}")

print("Done.")
