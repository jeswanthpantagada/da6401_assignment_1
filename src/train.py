"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import f1_score

try:
    from .ann.neural_network import NeuralNetwork
    from .utils.data_loader import preprocess_split
except ImportError:
    from ann.neural_network import NeuralNetwork
    from utils.data_loader import preprocess_split

DEFAULT_MODEL_PATH = os.path.join("outputs", "last_train_model.npy")
DEFAULT_CONFIG_PATH = os.path.join("outputs", "last_train_config.json")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], required=True)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", type=str, choices=["cross_entropy", "mse"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128])
    parser.add_argument("-a", "--activation", type=str, choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "xavier"], default="random")

    parser.add_argument("--hidden_layers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num_neurons", type=int, nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--wandb_project", type=str, default="da6401_assignment")
    parser.add_argument("--model_save_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config_save_path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)

    if argv is None:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args(argv)

    if args.hidden_layers is not None:
        args.num_layers = args.hidden_layers
    if args.num_neurons is not None:
        args.hidden_size = args.num_neurons

    if len(args.hidden_size) == 1 and args.num_layers > 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        raise ValueError("Length of --hidden_size must match --num_layers")

    args.hidden_layers = args.num_layers
    args.num_neurons = args.hidden_size[0]
    return args


def save_model(model, model_path, config_path, config):
    model_dir = os.path.dirname(model_path)
    config_dir = os.path.dirname(config_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)

    np.save(model_path, model.get_weights(), allow_pickle=True)
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    # If the user explicitly updates the submission artifact under src/, mirror it at repo root too.
    if os.path.normpath(model_path) == os.path.normpath(os.path.join("src", "best_model.npy")):
        np.save("best_model.npy", model.get_weights(), allow_pickle=True)
    if os.path.normpath(config_path) == os.path.normpath(os.path.join("src", "best_config.json")):
        with open("best_config.json", "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)


def main(argv=None):
    args = parse_arguments(argv)
    np.random.seed(args.seed)

    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_split(args.dataset)
    model = NeuralNetwork(args)

    best_state = model.get_weights()
    best_epoch_loss = np.inf

    for epoch in range(args.epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_epoch = X_train[permutation]
        y_train_epoch = y_train[permutation]

        batch_losses = []
        for start in range(0, X_train_epoch.shape[0], args.batch_size):
            end = start + args.batch_size
            X_batch = X_train_epoch[start:end]
            y_batch = y_train_epoch[start:end]

            logits = model.forward(X_batch)
            loss = model.compute_loss(logits, y_batch)
            model.backward(y_batch, logits)
            model.update_weights()
            batch_losses.append(loss)

        epoch_loss = float(np.mean(batch_losses))
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_state = model.get_weights()

        print(f"Epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.6f}")

    model.set_weights(best_state)
    eval_batch_size = max(1024, args.batch_size)
    val_metrics = model.evaluate(X_val, y_val, batch_size=eval_batch_size)
    test_metrics = model.evaluate(X_test, y_test, batch_size=eval_batch_size)
    val_f1 = f1_score(y_val, val_metrics["predictions"], average="macro")
    test_f1 = f1_score(y_test, test_metrics["predictions"], average="macro")

    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "num_layers": args.num_layers,
        "hidden_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "num_neurons": args.hidden_size[0],
        "activation": args.activation,
        "loss": args.loss,
        "weight_init": args.weight_init,
        "input_size": 784,
        "output_size": 10,
        "val_f1": val_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_f1,
    }

    save_model(model, args.model_save_path, args.config_save_path, config)
    print(f"Validation F1: {val_f1:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print(f"Model saved to {args.model_save_path}")
    print(f"Config saved to {args.config_save_path}")
    return config


if __name__ == "__main__":
    main()
