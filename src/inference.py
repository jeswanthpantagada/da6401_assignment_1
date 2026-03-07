"""
Inference Script
Evaluate trained models on test sets
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from .ann.neural_network import NeuralNetwork
    from .utils.data_loader import load_dataset
except ImportError:
    from ann.neural_network import NeuralNetwork
    from utils.data_loader import load_dataset

DEFAULT_MODEL_PATH = os.path.join("src", "best_model.npy")
DEFAULT_CONFIG_PATH = os.path.join("src", "best_config.json")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashion_mnist"], default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=None)
    parser.add_argument("--num_neurons", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, nargs="+", default=None)
    parser.add_argument("--activation", type=str, choices=["relu", "sigmoid", "tanh"], default=None)
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH)
    if argv is None:
        args, _ = parser.parse_known_args()
        return args
    return parser.parse_args(argv)


def load_serialized_weights(model_path):
    if not os.path.exists(model_path):
        fallback_candidates = []
        if os.path.basename(model_path) == "best_model.npy":
            fallback_candidates = ["best_model.npy", os.path.join("src", "best_model.npy")]
        for candidate in fallback_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

    payload = np.load(model_path, allow_pickle=True)

    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()

    if isinstance(payload, dict):
        if "weights" in payload:
            return payload["weights"]
        if any(key.startswith("W") for key in payload):
            return payload

    if isinstance(payload, np.ndarray):
        dense_layers = []
        for layer in payload.tolist():
            if hasattr(layer, "W") and hasattr(layer, "b"):
                dense_layers.append(layer)
        if dense_layers:
            weights = {}
            for index, layer in enumerate(dense_layers):
                weights[f"W{index}"] = layer.W.copy()
                weights[f"b{index}"] = layer.b.copy()
            return weights

    raise ValueError(f"Unsupported model format in {model_path}")


def load_config(args):
    config = {}
    config_path = args.config_path
    if config_path and not os.path.exists(config_path) and os.path.basename(config_path) == "best_config.json":
        for candidate in ["best_config.json", os.path.join("src", "best_config.json")]:
            if os.path.exists(candidate):
                config_path = candidate
                break
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
    if args.dataset is not None:
        config["dataset"] = args.dataset
    if args.activation is not None:
        config["activation"] = args.activation
    if args.hidden_layers is not None:
        config["hidden_layers"] = args.hidden_layers
    if args.num_neurons is not None:
        config["num_neurons"] = args.num_neurons
    if args.hidden_size is not None:
        config["hidden_size"] = args.hidden_size
    return config


def build_model_from_weights(weights, config):
    weight_keys = sorted([key for key in weights if key.startswith("W")], key=lambda key: int(key[1:]))
    hidden_sizes = [weights[key].shape[1] for key in weight_keys[:-1]]

    activation = config.get("activation", "relu")
    num_layers = config.get("num_layers", config.get("hidden_layers", len(hidden_sizes)))
    hidden_size = config.get("hidden_size", hidden_sizes)
    if isinstance(hidden_size, int):
        hidden_size = [hidden_size] * int(num_layers)

    model = NeuralNetwork(
        input_size=weights["W0"].shape[0],
        output_size=weights[weight_keys[-1]].shape[1],
        num_layers=len(hidden_sizes),
        hidden_size=hidden_sizes if hidden_sizes else hidden_size,
        activation=activation,
        loss=config.get("loss", "cross_entropy"),
        optimizer=config.get("optimizer", "sgd"),
        learning_rate=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 0.0),
        weight_init=config.get("weight_init", "random"),
    )
    model.set_weights(weights)
    return model


def evaluate_model(model, X_test, y_test, batch_size=256):
    logits_batches = []
    for start in range(0, X_test.shape[0], batch_size):
        end = start + batch_size
        logits_batches.append(model.forward(X_test[start:end]))

    logits = np.vstack(logits_batches)
    y_pred = np.argmax(logits, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    loss = model.compute_loss(logits, y_test)

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main(argv=None):
    args = parse_arguments(argv)
    config = load_config(args)

    dataset_name = config.get("dataset", "mnist")
    (_, _), (X_test, y_test) = load_dataset(dataset_name)
    X_test = X_test.astype(np.float64) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1)

    weights = load_serialized_weights(args.model_path)
    model = build_model_from_weights(weights, config)
    results = evaluate_model(model, X_test, y_test, batch_size=args.batch_size)

    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1-score:", results["f1"])
    return results


if __name__ == "__main__":
    main()
