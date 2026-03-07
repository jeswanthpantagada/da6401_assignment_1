"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split


def load_dataset(dataset_name):
    if dataset_name == "mnist":
        return mnist.load_data()
    if dataset_name == "fashion_mnist":
        return fashion_mnist.load_data()
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def preprocess_split(dataset_name, validation_split=0.1, random_state=42):
    (X_train, y_train), (X_test, y_test) = load_dataset(dataset_name)

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_split,
        random_state=random_state,
        stratify=y_train,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
