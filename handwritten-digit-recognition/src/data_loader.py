"""Load MNIST and set repeatable random seeds."""

import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


SEED = 42


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def load_mnist():
    print("Loading MNIST dataset...")
    return mnist.load_data()


def load_emnist_letters():
    """Load EMNIST Letters as upright 28x28 images with labels 0-25 (A-Z)."""
    import tensorflow_datasets as tfds

    print("Loading EMNIST Letters dataset...")
    train, test = tfds.load("emnist/letters", split=["train", "test"],
                            as_supervised=True, batch_size=-1)
    x_train, y_train = tfds.as_numpy(train)
    x_test, y_test = tfds.as_numpy(test)
    # TFDS documents EMNIST images as transposed relative to upright writing.
    x_train = x_train[..., 0].transpose(0, 2, 1)
    x_test = x_test[..., 0].transpose(0, 2, 1)
    minimum = min(int(y_train.min()), int(y_test.min()))
    maximum = max(int(y_train.max()), int(y_test.max()))
    if (minimum, maximum) == (1, 26):
        y_train = y_train - 1
        y_test = y_test - 1
    elif (minimum, maximum) != (0, 25):
        raise ValueError(f"Unexpected EMNIST Letters label range: {minimum} to {maximum}")
    return (x_train, y_train), (x_test, y_test)


def load_dataset(task):
    if task == "digit":
        return load_mnist()
    if task == "letter":
        return load_emnist_letters()
    raise ValueError("task must be 'digit' or 'letter'")
