"""
data_loader.py - Load MNIST + custom handwritten samples for PIN digit recognition

Custom samples: 28x28 binary text files (0/1)
Split per digit: 6 train / 2 val / 1 test (from 9 samples)
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import tensorflow as tf
from tensorflow import keras


def load_custom_sample(filepath: Path) -> np.ndarray:
    """Load a 28x28 binary text file as numpy array."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse 28x28 grid of 0s and 1s
    data = []
    for line in lines[:28]:  # Take first 28 lines
        row = [int(c) for c in line.strip()[:28]]
        # Pad if needed
        while len(row) < 28:
            row.append(0)
        data.append(row)
    
    # Pad rows if needed
    while len(data) < 28:
        data.append([0] * 28)
    
    return np.array(data, dtype=np.float32)


def load_custom_samples(samples_dir: Path, samples_per_digit: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load custom digit samples from directory.
    
    Expected naming: digit_{label}_{index}.txt
    Returns: (images, labels) arrays
    """
    images = []
    labels = []
    
    for digit in range(10):
        count = 0
        idx = 0
        while count < samples_per_digit:
            # Try different naming patterns
            patterns = [
                f"digit_{digit}_{idx:02d}.txt",
                f"digit_{digit}_{idx}.txt",
            ]
            
            found = False
            for pattern in patterns:
                filepath = samples_dir / pattern
                if filepath.exists():
                    img = load_custom_sample(filepath)
                    images.append(img)
                    labels.append(digit)
                    count += 1
                    found = True
                    break
            
            idx += 1
            # Safety: don't loop forever
            if idx > 20:
                break
        
        if count < samples_per_digit:
            print(f"Warning: Only found {count} samples for digit {digit}")
    
    return np.array(images), np.array(labels)


def split_custom_samples(
    images: np.ndarray, 
    labels: np.ndarray,
    train_per_digit: int = 6,
    val_per_digit: int = 2,
    test_per_digit: int = 1
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split custom samples into train/val/test sets.
    Ensures balanced split across all digits.
    """
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    test_images, test_labels = [], []
    
    for digit in range(10):
        mask = labels == digit
        digit_images = images[mask]
        digit_labels = labels[mask]
        
        n = len(digit_images)
        if n < train_per_digit + val_per_digit + test_per_digit:
            print(f"Warning: Digit {digit} has only {n} samples")
            # Adjust split proportionally
            train_n = max(1, int(n * 0.67))
            val_n = max(1, int(n * 0.22))
            test_n = max(1, n - train_n - val_n)
        else:
            train_n = train_per_digit
            val_n = val_per_digit
            test_n = test_per_digit
        
        train_images.extend(digit_images[:train_n])
        train_labels.extend(digit_labels[:train_n])
        
        val_images.extend(digit_images[train_n:train_n + val_n])
        val_labels.extend(digit_labels[train_n:train_n + val_n])
        
        test_images.extend(digit_images[train_n + val_n:train_n + val_n + test_n])
        test_labels.extend(digit_labels[train_n + val_n:train_n + val_n + test_n])
    
    return {
        'train': (np.array(train_images), np.array(train_labels)),
        'val': (np.array(val_images), np.array(val_labels)),
        'test': (np.array(test_images), np.array(test_labels))
    }


def augment_image(image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Apply random augmentation to a single image."""
    if seed is not None:
        np.random.seed(seed)
    
    img = image.copy()
    
    # Random rotation (-15 to +15 degrees)
    angle = np.random.uniform(-15, 15)
    img = tf.keras.preprocessing.image.apply_affine_transform(
        img.reshape(28, 28, 1),
        theta=angle,
        fill_mode='constant',
        cval=0.0
    ).reshape(28, 28)
    
    # Random shift (-2 to +2 pixels)
    tx = np.random.randint(-2, 3)
    ty = np.random.randint(-2, 3)
    img = tf.keras.preprocessing.image.apply_affine_transform(
        img.reshape(28, 28, 1),
        tx=tx,
        ty=ty,
        fill_mode='constant',
        cval=0.0
    ).reshape(28, 28)
    
    # Random zoom (0.9 to 1.1)
    zoom = np.random.uniform(0.9, 1.1)
    img = tf.keras.preprocessing.image.apply_affine_transform(
        img.reshape(28, 28, 1),
        zx=zoom,
        zy=zoom,
        fill_mode='constant',
        cval=0.0
    ).reshape(28, 28)
    
    return np.clip(img, 0, 1).astype(np.float32)


def augment_dataset(
    images: np.ndarray, 
    labels: np.ndarray, 
    augmentations_per_sample: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment dataset by generating variations of each sample."""
    aug_images = list(images)  # Include originals
    aug_labels = list(labels)
    
    for i, (img, lbl) in enumerate(zip(images, labels)):
        for j in range(augmentations_per_sample):
            aug_img = augment_image(img, seed=i * 1000 + j)
            aug_images.append(aug_img)
            aug_labels.append(lbl)
    
    return np.array(aug_images), np.array(aug_labels)


def load_combined_dataset(
    samples_dir: Optional[Path] = None,
    mnist_train_samples: int = 54000,  # Use subset for faster training
    augment_custom: bool = True,
    augmentations_per_sample: int = 10,
    custom_weight: float = 3.0  # Repeat custom samples to balance with MNIST
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load combined MNIST + custom samples dataset.
    
    Returns dict with 'train', 'val', 'test' keys, each containing (images, labels).
    Images are normalized to [0, 1] with shape (N, 28, 28).
    """
    # Load MNIST
    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = keras.datasets.mnist.load_data()
    
    # Normalize MNIST
    mnist_x_train = mnist_x_train.astype(np.float32) / 255.0
    mnist_x_test = mnist_x_test.astype(np.float32) / 255.0
    
    # Split MNIST into train/val
    val_size = 6000
    mnist_x_val = mnist_x_train[-val_size:]
    mnist_y_val = mnist_y_train[-val_size:]
    mnist_x_train = mnist_x_train[:mnist_train_samples]
    mnist_y_train = mnist_y_train[:mnist_train_samples]
    
    print(f"MNIST - Train: {len(mnist_x_train)}, Val: {len(mnist_x_val)}, Test: {len(mnist_x_test)}")
    
    # Load and process custom samples if directory provided
    if samples_dir is not None and samples_dir.exists():
        custom_images, custom_labels = load_custom_samples(samples_dir)
        print(f"Custom samples loaded: {len(custom_images)}")
        
        # Split custom samples
        custom_split = split_custom_samples(custom_images, custom_labels)
        
        # Augment custom training samples
        if augment_custom:
            custom_train_x, custom_train_y = augment_dataset(
                custom_split['train'][0],
                custom_split['train'][1],
                augmentations_per_sample=augmentations_per_sample
            )
            print(f"Custom training after augmentation: {len(custom_train_x)}")
        else:
            custom_train_x, custom_train_y = custom_split['train']
        
        # Repeat custom samples to balance with MNIST
        repeat_factor = int(custom_weight)
        custom_train_x = np.tile(custom_train_x, (repeat_factor, 1, 1))
        custom_train_y = np.tile(custom_train_y, repeat_factor)
        
        # Combine datasets
        train_x = np.concatenate([mnist_x_train, custom_train_x])
        train_y = np.concatenate([mnist_y_train, custom_train_y])
        
        val_x = np.concatenate([mnist_x_val, custom_split['val'][0]])
        val_y = np.concatenate([mnist_y_val, custom_split['val'][1]])
        
        test_x = np.concatenate([mnist_x_test, custom_split['test'][0]])
        test_y = np.concatenate([mnist_y_test, custom_split['test'][1]])
        
        print(f"Combined - Train: {len(train_x)}, Val: {len(val_x)}, Test: {len(test_x)}")
    else:
        print("No custom samples directory provided, using MNIST only")
        train_x, train_y = mnist_x_train, mnist_y_train
        val_x, val_y = mnist_x_val, mnist_y_val
        test_x, test_y = mnist_x_test, mnist_y_test
    
    # Shuffle training data
    perm = np.random.permutation(len(train_x))
    train_x, train_y = train_x[perm], train_y[perm]
    
    return {
        'train': (train_x, train_y),
        'val': (val_x, val_y),
        'test': (test_x, test_y)
    }


def create_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    add_channel: bool = True
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from numpy arrays."""
    if add_channel:
        images = np.expand_dims(images, -1)
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    # Test loading
    samples_dir = Path("../samples")  # Adjust path as needed
    
    if samples_dir.exists():
        data = load_combined_dataset(samples_dir)
        print(f"\nFinal dataset shapes:")
        print(f"  Train: {data['train'][0].shape}, {data['train'][1].shape}")
        print(f"  Val: {data['val'][0].shape}, {data['val'][1].shape}")
        print(f"  Test: {data['test'][0].shape}, {data['test'][1].shape}")
    else:
        print(f"Samples directory not found: {samples_dir}")
        print("Testing with MNIST only...")
        data = load_combined_dataset(None)
