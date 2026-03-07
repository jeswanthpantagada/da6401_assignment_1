"""
Inference Script
Evaluate trained models on test sets
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_arguments():

    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--dataset", type=str, choices=["mnist", "fashion_mnist"], required=True)

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--hidden_layers", type=int, default=1)

    parser.add_argument("--num_neurons", type=int, default=128)

    parser.add_argument("--activation", type=str, choices=["relu", "sigmoid", "tanh"], default="relu")

    return parser.parse_args()



def load_model(model_path):

    model_layers = np.load(model_path, allow_pickle=True)

    return model_layers


def evaluate_model(model, X_test, y_test):

    output = X_test

    for layer in model:
        output = layer.forward(output)

    logits = output

    y_pred = np.argmax(logits, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, average="macro")

    recall = recall_score(y_test, y_pred, average="macro")

    f1 = f1_score(y_test, y_pred, average="macro")

    loss = None

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():

    args = parse_arguments()

    if args.dataset == "mnist":
        (_, _), (X_test, y_test) = mnist.load_data()

    elif args.dataset == "fashion_mnist":
        (_, _), (X_test, y_test) = fashion_mnist.load_data()

    X_test = X_test / 255.0

    X_test = X_test.reshape(X_test.shape[0], -1)

    model = load_model(args.model_path)

    results = evaluate_model(model, X_test, y_test)

    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1-score:", results["f1"])

    return results


if __name__ == '__main__':
    main()
