"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
from models.mlp import NeuralNetwork
from optimizers.sgd import SGD
from optimizers.momentum import Momentum
from optimizers.nag import NAG
from optimizers.rmsprop import RMSProp
from optimizers.adam import Adam
from optimizers.nadam import Nadam
import json

def parse_arguments():

    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('--dataset', type=str, choices=['mnist','fashion_mnist'], required=True)

    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--optimizer', type=str,
                        choices=['sgd','momentum','nag','rmsprop','adam','nadam'],
                        default='sgd')

    parser.add_argument('--hidden_layers', type=int, default=1)

    parser.add_argument('--num_neurons', type=int, default=128)

    parser.add_argument('--activation', type=str,
                        choices=['relu','sigmoid','tanh'],
                        default='relu')

    parser.add_argument('--loss', type=str,
                        choices=['cross_entropy','mse'],
                        default='cross_entropy')

    parser.add_argument('--weight_init', type=str, default='random')

    parser.add_argument('--wandb_project', type=str, default='da6401_assignment')

    parser.add_argument('--model_save_path', type=str, default='best_model.npy')

    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--hidden_size', type=int)

    args = parser.parse_args()

    if args.num_layers is not None:
        args.hidden_layers = args.num_layers

    if args.hidden_size is not None:
        args.num_neurons = args.hidden_size

    return args

def load_dataset(dataset_name):

    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError("Invalid dataset")

    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, y_train, X_test, y_test):

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    num_classes = 10

    y_train_onehot = np.zeros((y_train.shape[0], num_classes))
    y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1

    y_test_onehot = np.zeros((y_test.shape[0], num_classes))
    y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1

    X_train, X_val, y_train_onehot, y_val = train_test_split(
        X_train,
        y_train_onehot,
        test_size=0.1,
        random_state=42
    )

    return X_train, y_train_onehot, X_val, y_val, X_test, y_test_onehot

def get_optimizer(name, learning_rate):

    if name == "sgd":
        return SGD(learning_rate)

    elif name == "momentum":
        return Momentum(learning_rate)

    elif name == "nag":
        return NAG(learning_rate)

    elif name == "rmsprop":
        return RMSProp(learning_rate)

    elif name == "adam":
        return Adam(learning_rate)

    elif name == "nadam":
        return Nadam(learning_rate)

    else:
        raise ValueError("Invalid optimizer")

def main():

    args = parse_arguments()

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test
    )

    model = NeuralNetwork(
    input_size=784,
    hidden_layers=args.hidden_layers,
    num_neurons=args.num_neurons,
    activation=args.activation,
    loss=args.loss,
    weight_init=args.weight_init
)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)

    for epoch in range(args.epochs):

        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(0, X_train.shape[0], args.batch_size):

            X_batch = X_train[i:i+args.batch_size]
            y_batch = y_train[i:i+args.batch_size]

            y_pred = model.forward(X_batch)

            loss = model.compute_loss(y_pred, y_batch)

            model.backward(y_pred, y_batch)

            for layer in model.layers:

                if hasattr(layer, "W"):
                    optimizer.update(layer)

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss}")
    
    np.save(args.model_save_path, model)
    print("Model saved successfully!")
    
    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "hidden_layers": args.hidden_layers,
        "num_neurons": args.num_neurons,
        "activation": args.activation,
        "loss": args.loss,
        "weight_init": args.weight_init
    }

    with open("best_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("Configuration saved successfully!")

    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Test data:", X_test.shape)

    print("Training complete!")


if __name__ == '__main__':
    main()
