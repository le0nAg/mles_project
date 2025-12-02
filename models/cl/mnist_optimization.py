"""
MNIST Optimization Framework
Comprehensive analysis of different architectures and hyperparameters
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# Data Loading
# -----------------------------
class DataLoader:
    """Handle MNIST dataset loading and preprocessing"""
    
    @staticmethod
    def load_mnist():
        """Load and normalize MNIST dataset"""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        print(f"Dataset loaded: Train={x_train.shape}, Test={x_test.shape}")
        return (x_train, y_train), (x_test, y_test)
    
    @staticmethod
    def load_custom_data(data_path: Path):
        """Load custom handwritten digit data (Step 2)"""
        # This will load 10 custom images from the Kaggle dataset
        # Format: 28x28 grayscale images
        custom_images = []
        custom_labels = []
        
        if data_path.exists():
            # Assuming images are named as digit_0.png, digit_1.png, etc.
            for i in range(10):
                img_path = data_path / f"digit_{i}.png"
                if img_path.exists():
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, color_mode='grayscale', target_size=(28, 28)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array.squeeze() / 255.0
                    custom_images.append(img_array)
                    custom_labels.append(i)
        
        if custom_images:
            return np.array(custom_images), np.array(custom_labels)
        else:
            print("Warning: No custom data found. Using test set samples instead.")
            return None, None


# -----------------------------
# Model Architectures
# -----------------------------
class ModelArchitectures:
    """Collection of different neural network architectures"""
    
    @staticmethod
    def baseline_fc(input_shape=(28, 28)):
        """Baseline fully connected model from your code"""
        model = keras.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(80, activation='elu'),
            layers.Dense(60, activation='elu'),
            layers.Dropout(0.2),
            layers.Dense(10)
        ], name='baseline_fc')
        return model
    
    @staticmethod
    def simple_cnn(input_shape=(28, 28, 1)):
        """Simple CNN architecture (Step 1)"""
        model = keras.Sequential([
            layers.Reshape((28, 28, 1), input_shape=(28, 28)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10)
        ], name='simple_cnn')
        return model
    
    @staticmethod
    def optimized_cnn(input_shape=(28, 28, 1), activation='relu', dropout_rate=0.2):
        """Optimized CNN with configurable parameters"""
        model = keras.Sequential([
            layers.Reshape((28, 28, 1), input_shape=(28, 28)),
            layers.Conv2D(16, (3, 3), activation=activation, padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation=activation, padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation=activation),
            layers.Dropout(dropout_rate),
            layers.Dense(10)
        ], name='optimized_cnn')
        return model
    
    @staticmethod
    def squeezenet_like(input_shape=(28, 28, 1)):
        """SqueezeNet-inspired architecture for MNIST"""
        def fire_module(x, squeeze_filters, expand_filters):
            squeeze = layers.Conv2D(squeeze_filters, (1, 1), activation='relu', padding='same')(x)
            expand_1x1 = layers.Conv2D(expand_filters, (1, 1), activation='relu', padding='same')(squeeze)
            expand_3x3 = layers.Conv2D(expand_filters, (3, 3), activation='relu', padding='same')(squeeze)
            return layers.Concatenate()([expand_1x1, expand_3x3])
        
        inputs = layers.Input(shape=(28, 28))
        x = layers.Reshape((28, 28, 1))(inputs)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = fire_module(x, 8, 16)
        x = fire_module(x, 8, 16)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = fire_module(x, 16, 32)
        x = layers.Dropout(0.2)(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='squeezenet_like')
        return model
    
    @staticmethod
    def mobilenet_like(input_shape=(28, 28, 1)):
        """MobileNet-inspired architecture for MNIST"""
        def depthwise_separable_conv(x, filters, stride=1):
            x = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters, (1, 1), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            return x
        
        inputs = layers.Input(shape=(28, 28))
        x = layers.Reshape((28, 28, 1))(inputs)
        x = layers.Conv2D(8, (3, 3), strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = depthwise_separable_conv(x, 16, stride=1)
        x = depthwise_separable_conv(x, 32, stride=2)
        x = depthwise_separable_conv(x, 32, stride=1)
        x = depthwise_separable_conv(x, 64, stride=2)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(10)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='mobilenet_like')
        return model
    
    @staticmethod
    def mcunet_like(input_shape=(28, 28, 1)):
        """MCUNet-inspired tiny architecture for MNIST"""
        inputs = layers.Input(shape=(28, 28))
        x = layers.Reshape((28, 28, 1))(inputs)
        
        # Very lightweight architecture
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(10)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='mcunet_like')
        return model


# -----------------------------
# Performance Measurement
# -----------------------------
class PerformanceMeasurement:
    """Measure model performance metrics"""
    
    @staticmethod
    def measure_inference_speed_keras(model, input_shape=(28, 28), runs=200):
        """Measure Keras model inference speed"""
        dummy = np.random.rand(1, *input_shape).astype('float32')
        
        # Warm-up
        for _ in range(10):
            _ = model.predict(dummy, verbose=0)
        
        t0 = time.time()
        for _ in range(runs):
            _ = model.predict(dummy, verbose=0)
        t1 = time.time()
        
        avg_ms = (t1 - t0) / runs * 1000
        return avg_ms
    
    @staticmethod
    def measure_tflite_inference_speed(tflite_path, input_shape=(28, 28), runs=200):
        """Measure TFLite model inference speed"""
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        dummy = np.random.rand(1, *input_shape).astype('float32')
        
        # Warm-up
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
    
    @staticmethod
    def convert_to_tflite(model, output_path: Path, quantize=False):
        """Convert Keras model to TFLite format"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)
        
        return output_path, output_path.stat().st_size
    
    @staticmethod
    def get_model_size(model_path: Path):
        """Get model file size in bytes"""
        return model_path.stat().st_size
    
    @staticmethod
    def count_parameters(model):
        """Count trainable parameters"""
        return model.count_params()


# -----------------------------
# Experiment Manager
# -----------------------------
class ExperimentManager:
    """Manage optimization experiments"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def run_experiment(self, 
                      model_fn,
                      model_name: str,
                      x_train, y_train,
                      x_test, y_test,
                      optimizer='adam',
                      learning_rate=0.001,
                      epochs=5,
                      batch_size=32,
                      custom_data=None):
        """Run a single experiment with given configuration"""
        
        print(f"\n{'='*60}")
        print(f"Experiment: {model_name}")
        print(f"Optimizer: {optimizer}, LR: {learning_rate}, Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        # Build model
        model = model_fn()
        model.summary()
        
        # Compile model
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
        
        # Train model
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        # Evaluate on custom data if available
        custom_acc = None
        if custom_data is not None:
            x_custom, y_custom = custom_data
            predictions = model.predict(x_custom, verbose=0)
            predicted_labels = np.argmax(tf.nn.softmax(predictions), axis=1)
            custom_acc = np.mean(predicted_labels == y_custom)
        
        # Save model
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        keras_path = model_dir / "model.h5"
        model.save(keras_path)
        keras_size = keras_path.stat().st_size
        
        # Convert to TFLite
        tflite_path = model_dir / "model.tflite"
        _, tflite_size = PerformanceMeasurement.convert_to_tflite(
            model, tflite_path, quantize=False
        )
        
        # Convert to TFLite with quantization
        tflite_quant_path = model_dir / "model_quantized.tflite"
        _, tflite_quant_size = PerformanceMeasurement.convert_to_tflite(
            model, tflite_quant_path, quantize=True
        )
        
        # Measure inference speeds
        keras_ms = PerformanceMeasurement.measure_inference_speed_keras(model)
        tflite_ms = PerformanceMeasurement.measure_tflite_inference_speed(tflite_path)
        tflite_quant_ms = PerformanceMeasurement.measure_tflite_inference_speed(tflite_quant_path)
        
        # Count parameters
        num_params = PerformanceMeasurement.count_parameters(model)
        
        # Store results
        result = {
            'model_name': model_name,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'num_parameters': num_params,
            'keras_size_bytes': keras_size,
            'keras_size_kb': keras_size / 1024,
            'tflite_size_bytes': tflite_size,
            'tflite_size_kb': tflite_size / 1024,
            'tflite_quant_size_bytes': tflite_quant_size,
            'tflite_quant_size_kb': tflite_quant_size / 1024,
            'validation_accuracy': history.history['val_accuracy'][-1],
            'test_accuracy': test_acc,
            'custom_accuracy': custom_acc if custom_acc is not None else 'N/A',
            'keras_inference_ms': keras_ms,
            'tflite_inference_ms': tflite_ms,
            'tflite_quant_inference_ms': tflite_quant_ms,
            'train_history': history.history
        }
        
        self.results.append(result)
        
        # Print summary
        print(f"\n--- Results Summary ---")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Parameters: {num_params:,}")
        print(f"Keras Size: {keras_size/1024:.2f} KB")
        print(f"TFLite Size: {tflite_size/1024:.2f} KB")
        print(f"TFLite Quantized Size: {tflite_quant_size/1024:.2f} KB")
        print(f"Keras Inference: {keras_ms:.3f} ms")
        print(f"TFLite Inference: {tflite_ms:.3f} ms")
        print(f"TFLite Quantized Inference: {tflite_quant_ms:.3f} ms")
        if custom_acc is not None:
            print(f"Custom Data Accuracy: {custom_acc:.4f}")
        
        return result
    
    def save_results(self):
        """Save all results to CSV and JSON"""
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'train_history'} 
                          for r in self.results])
        
        csv_path = self.output_dir / "optimization_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        # Save full results with training history
        json_path = self.output_dir / "optimization_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Full results saved to {json_path}")
        
        return df


# -----------------------------
# Visualization
# -----------------------------
class Visualizer:
    """Create visualizations for analysis"""
    
    @staticmethod
    def plot_optimization_comparison(df: pd.DataFrame, output_dir: Path):
        """Create comprehensive comparison plots"""
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Accuracy Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Test Accuracy
        ax = axes[0, 0]
        df_sorted = df.sort_values('test_accuracy', ascending=False)
        sns.barplot(data=df_sorted.head(15), x='test_accuracy', y='model_name', ax=ax, palette='viridis')
        ax.set_title('Test Accuracy Comparison (Top 15)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Test Accuracy', fontsize=12)
        ax.set_ylabel('Model Configuration', fontsize=12)
        
        # Model Size Comparison
        ax = axes[0, 1]
        df_sorted = df.sort_values('tflite_size_kb')
        sns.barplot(data=df_sorted.head(15), x='tflite_size_kb', y='model_name', ax=ax, palette='rocket')
        ax.set_title('TFLite Model Size (Top 15 Smallest)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Size (KB)', fontsize=12)
        ax.set_ylabel('Model Configuration', fontsize=12)
        
        # Inference Speed
        ax = axes[1, 0]
        df_sorted = df.sort_values('tflite_inference_ms')
        sns.barplot(data=df_sorted.head(15), x='tflite_inference_ms', y='model_name', ax=ax, palette='mako')
        ax.set_title('TFLite Inference Speed (Top 15 Fastest)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Inference Time (ms)', fontsize=12)
        ax.set_ylabel('Model Configuration', fontsize=12)
        
        # Parameters Count
        ax = axes[1, 1]
        df_sorted = df.sort_values('num_parameters')
        sns.barplot(data=df_sorted.head(15), x='num_parameters', y='model_name', ax=ax, palette='crest')
        ax.set_title('Number of Parameters (Top 15 Smallest)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Parameters', fontsize=12)
        ax.set_ylabel('Model Configuration', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'optimization_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'optimization_comparison.png'}")
        plt.close()
        
        # 2. Scatter plots for trade-offs
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy vs Size
        ax = axes[0]
        scatter = ax.scatter(df['tflite_size_kb'], df['test_accuracy'], 
                           s=100, alpha=0.6, c=df['tflite_inference_ms'], 
                           cmap='plasma')
        ax.set_xlabel('TFLite Model Size (KB)', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Size Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Inference Time (ms)', fontsize=10)
        
        # Accuracy vs Speed
        ax = axes[1]
        scatter = ax.scatter(df['tflite_inference_ms'], df['test_accuracy'],
                           s=100, alpha=0.6, c=df['tflite_size_kb'],
                           cmap='plasma')
        ax.set_xlabel('TFLite Inference Time (ms)', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Model Size (KB)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tradeoff_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'tradeoff_analysis.png'}")
        plt.close()
        
        # 3. Architecture Comparison
        architectures = df['model_name'].str.split('_').str[0].unique()
        if len(architectures) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            metrics = [
                ('test_accuracy', 'Test Accuracy', axes[0, 0]),
                ('tflite_size_kb', 'TFLite Size (KB)', axes[0, 1]),
                ('tflite_inference_ms', 'Inference Time (ms)', axes[1, 0]),
                ('num_parameters', 'Parameters', axes[1, 1])
            ]
            
            for metric, title, ax in metrics:
                df['architecture'] = df['model_name'].str.split('_').str[0]
                sns.boxplot(data=df, x='architecture', y=metric, ax=ax, palette='Set2')
                ax.set_title(f'{title} by Architecture', fontsize=14, fontweight='bold')
                ax.set_xlabel('Architecture', fontsize=12)
                ax.set_ylabel(title, fontsize=12)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir / 'architecture_comparison.png'}")
            plt.close()
    
    @staticmethod
    def create_detailed_table(df: pd.DataFrame, output_dir: Path):
        """Create detailed comparison table"""
        
        # Select key columns
        table_df = df[[
            'model_name', 'optimizer', 'learning_rate', 'epochs',
            'num_parameters', 'tflite_size_kb', 'tflite_quant_size_kb',
            'validation_accuracy', 'test_accuracy', 'custom_accuracy',
            'tflite_inference_ms', 'tflite_quant_inference_ms'
        ]].copy()
        
        # Round numerical values
        for col in table_df.select_dtypes(include=[np.number]).columns:
            if 'accuracy' in col:
                table_df[col] = table_df[col].round(4)
            elif 'ms' in col:
                table_df[col] = table_df[col].round(3)
            elif 'kb' in col:
                table_df[col] = table_df[col].round(2)
        
        # Sort by test accuracy
        table_df = table_df.sort_values('test_accuracy', ascending=False)
        
        # Save as CSV
        table_path = output_dir / 'detailed_results_table.csv'
        table_df.to_csv(table_path, index=False)
        print(f"Saved detailed table: {table_path}")
        
        return table_df


# -----------------------------
# Main Execution
# -----------------------------
def main():
    """Main execution function"""
    
    print("="*80)
    print("MNIST OPTIMIZATION FRAMEWORK")
    print("="*80)
    
    # Setup
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = DataLoader.load_mnist()
    
    # Try to load custom data (Step 2)
    custom_data_path = Path("custom_data")
    custom_data = DataLoader.load_custom_data(custom_data_path)
    
    # Initialize experiment manager
    manager = ExperimentManager(output_dir)
    
    # ============================================
    # STEP 1 & 3: Baseline and Optimizations
    # ============================================
    
    print("\n" + "="*80)
    print("PHASE 1: BASELINE MODELS")
    print("="*80)
    
    # 1. Original baseline (fully connected)
    manager.run_experiment(
        ModelArchitectures.baseline_fc,
        "baseline_fc_original",
        x_train, y_train, x_test, y_test,
        optimizer='adam', learning_rate=0.001, epochs=5,
        custom_data=custom_data
    )
    
    # 2. Simple CNN baseline
    manager.run_experiment(
        ModelArchitectures.simple_cnn,
        "simple_cnn_baseline",
        x_train, y_train, x_test, y_test,
        optimizer='adam', learning_rate=0.001, epochs=5,
        custom_data=custom_data
    )
    
    print("\n" + "="*80)
    print("PHASE 2: OPTIMIZER VARIATIONS")
    print("="*80)
    
    # Test different optimizers
    optimizers = ['adam', 'sgd', 'rmsprop']
    for opt in optimizers:
        manager.run_experiment(
            ModelArchitectures.simple_cnn,
            f"simple_cnn_opt_{opt}",
            x_train, y_train, x_test, y_test,
            optimizer=opt, learning_rate=0.001, epochs=5,
            custom_data=custom_data
        )
    
    print("\n" + "="*80)
    print("PHASE 3: ACTIVATION FUNCTION VARIATIONS")
    print("="*80)
    
    # Test different activation functions
    activations = ['relu', 'elu', 'swish']
    for act in activations:
        manager.run_experiment(
            lambda: ModelArchitectures.optimized_cnn(activation=act),
            f"optimized_cnn_act_{act}",
            x_train, y_train, x_test, y_test,
            optimizer='adam', learning_rate=0.001, epochs=5,
            custom_data=custom_data
        )
    
    print("\n" + "="*80)
    print("PHASE 4: EPOCH VARIATIONS")
    print("="*80)
    
    # Test different epoch counts
    epoch_counts = [3, 5, 10]
    for ep in epoch_counts:
        manager.run_experiment(
            ModelArchitectures.optimized_cnn,
            f"optimized_cnn_epochs_{ep}",
            x_train, y_train, x_test, y_test,
            optimizer='adam', learning_rate=0.001, epochs=ep,
            custom_data=custom_data
        )
    
    print("\n" + "="*80)
    print("PHASE 5: LEARNING RATE VARIATIONS")
    print("="*80)
    
    # Test different learning rates
    learning_rates = [0.0001, 0.001, 0.01]
    for lr in learning_rates:
        manager.run_experiment(
            ModelArchitectures.optimized_cnn,
            f"optimized_cnn_lr_{lr}",
            x_train, y_train, x_test, y_test,
            optimizer='adam', learning_rate=lr, epochs=5,
            custom_data=custom_data
        )
    
    # ============================================
    # STEP 5: Alternative Architectures
    # ============================================
    
    print("\n" + "="*80)
    print("PHASE 6: ALTERNATIVE ARCHITECTURES")
    print("="*80)
    
    # SqueezeNet-like
    manager.run_experiment(
        ModelArchitectures.squeezenet_like,
        "squeezenet_like",
        x_train, y_train, x_test, y_test,
        optimizer='adam', learning_rate=0.001, epochs=5,
        custom_data=custom_data
    )
    
    # MobileNet-like
    manager.run_experiment(
        ModelArchitectures.mobilenet_like,
        "mobilenet_like",
        x_train, y_train, x_test, y_test,
        optimizer='adam', learning_rate=0.001, epochs=5,
        custom_data=custom_data
    )
    
    # MCUNet-like
    manager.run_experiment(
        ModelArchitectures.mcunet_like,
        "mcunet_like",
        x_train, y_train, x_test, y_test,
        optimizer='adam', learning_rate=0.001, epochs=5,
        custom_data=custom_data
    )
    
    # ============================================
    # Save and Visualize Results
    # ============================================
    
    print("\n" + "="*80)
    print("SAVING RESULTS AND GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Save results
    results_df = manager.save_results()
    
    # Create visualizations
    Visualizer.plot_optimization_comparison(results_df, output_dir)
    Visualizer.create_detailed_table(results_df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {output_dir}")
    print("\nGenerated files:")
    print("  - optimization_results.csv")
    print("  - optimization_results.json")
    print("  - detailed_results_table.csv")
    print("  - optimization_comparison.png")
    print("  - tradeoff_analysis.png")
    print("  - architecture_comparison.png")
    print("\nIndividual model files saved in subdirectories")


if __name__ == "__main__":
    main()