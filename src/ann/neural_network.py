"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import argparse

import numpy as np

from .activations import ReLU, Sigmoid, Tanh
from .neural_layer import Dense
from .objective_functions import CrossEntropyLoss, MSELoss, softmax
from .optimizers import get_optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args=None, **kwargs):
        config = self._normalize_config(cli_args, kwargs)

        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.hidden_sizes = config["hidden_sizes"]
        self.activation_name = config["activation"]
        self.loss_name = config["loss"]
        self.weight_init = config["weight_init"]
        self.learning_rate = config["learning_rate"]
        self.optimizer_name = config["optimizer"]
        self.weight_decay = config["weight_decay"]

        self.layers = []
        self.activation_layers = []

        previous_size = self.input_size
        for hidden_size in self.hidden_sizes:
            self.layers.append(Dense(previous_size, hidden_size, self.weight_init))
            self.activation_layers.append(self._build_activation(self.activation_name))
            previous_size = hidden_size
        self.layers.append(Dense(previous_size, self.output_size, self.weight_init))

        self.loss_fn = CrossEntropyLoss() if self.loss_name == "cross_entropy" else MSELoss()
        self.optimizer = get_optimizer(self.optimizer_name, self.learning_rate)
        self.grad_W = np.empty(0, dtype=object)
        self.grad_b = np.empty(0, dtype=object)

    def _normalize_config(self, cli_args, kwargs):
        data = {}
        if cli_args is not None:
            if isinstance(cli_args, argparse.Namespace):
                data.update(vars(cli_args))
            elif isinstance(cli_args, dict):
                data.update(cli_args)
            else:
                raise TypeError("cli_args must be an argparse.Namespace or dict")
        data.update(kwargs)

        input_size = data.get("input_size", 784)
        output_size = data.get("output_size", 10)
        activation = data.get("activation", "relu")
        loss = data.get("loss", "cross_entropy")
        weight_init = data.get("weight_init", "random")
        learning_rate = data.get("learning_rate", 0.001)
        optimizer = data.get("optimizer", "sgd")
        weight_decay = data.get("weight_decay", 0.0)

        hidden_sizes = data.get("hidden_size")
        if hidden_sizes is None:
            hidden_sizes = data.get("hidden_sizes")
        num_neurons = data.get("num_neurons")
        if hidden_sizes is None and num_neurons is not None:
            num_layers = data.get("num_layers", data.get("hidden_layers", 1))
            if isinstance(num_neurons, (list, tuple, np.ndarray)):
                hidden_sizes = [int(size) for size in num_neurons]
            else:
                hidden_sizes = [int(num_neurons)] * int(num_layers)
        if hidden_sizes is None and data.get("hidden_layers") is not None and num_neurons is not None:
            if isinstance(num_neurons, (list, tuple, np.ndarray)):
                hidden_sizes = [int(size) for size in num_neurons]
            else:
                hidden_sizes = [int(num_neurons)] * int(data["hidden_layers"])
        if hidden_sizes is None:
            hidden_sizes = [128]

        if isinstance(hidden_sizes, int):
            num_layers = int(data.get("num_layers", data.get("hidden_layers", 1)))
            hidden_sizes = [int(hidden_sizes)] * num_layers
        else:
            hidden_sizes = [int(size) for size in hidden_sizes]

        num_layers = int(data.get("num_layers", data.get("hidden_layers", len(hidden_sizes))))
        if len(hidden_sizes) == 1 and num_layers > 1:
            hidden_sizes = hidden_sizes * num_layers
        elif len(hidden_sizes) != num_layers:
            raise ValueError("Length of hidden_size must match num_layers")

        return {
            "input_size": int(input_size),
            "output_size": int(output_size),
            "hidden_sizes": hidden_sizes,
            "activation": activation,
            "loss": loss,
            "weight_init": weight_init,
            "learning_rate": float(learning_rate),
            "optimizer": optimizer,
            "weight_decay": float(weight_decay),
        }

    def _build_activation(self, name):
        activation_map = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh,
        }
        if name not in activation_map:
            raise ValueError(f"Unsupported activation: {name}")
        return activation_map[name]()

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        output = X
        for layer_index, layer in enumerate(self.layers[:-1]):
            output = layer.forward(output)
            output = self.activation_layers[layer_index].forward(output)
        return self.layers[-1].forward(output)

    def compute_loss(self, y_pred, y_true):
        loss = self.loss_fn.forward(y_pred, self._ensure_one_hot(y_true))
        if self.weight_decay > 0.0:
            penalty = 0.0
            for layer in self.layers:
                penalty += np.sum(layer.W ** 2)
            loss += 0.5 * self.weight_decay * penalty
        return loss

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        y_true_one_hot = self._ensure_one_hot(y_true)
        grad_output = self.loss_fn.backward(y_pred, y_true_one_hot)

        grad_W_list = []
        grad_b_list = []

        grad_output = self.layers[-1].backward(grad_output)
        if self.weight_decay > 0.0:
            self.layers[-1].grad_W += self.weight_decay * self.layers[-1].W
        grad_W_list.append(self.layers[-1].grad_W.copy())
        grad_b_list.append(self.layers[-1].grad_b.copy())

        for layer_index in range(len(self.layers) - 2, -1, -1):
            grad_output = self.activation_layers[layer_index].backward(grad_output)
            grad_output = self.layers[layer_index].backward(grad_output)
            if self.weight_decay > 0.0:
                self.layers[layer_index].grad_W += self.weight_decay * self.layers[layer_index].W
            grad_W_list.append(self.layers[layer_index].grad_W.copy())
            grad_b_list.append(self.layers[layer_index].grad_b.copy())

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for index, (grad_W, grad_b) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[index] = grad_W
            self.grad_b[index] = grad_b
        return self.grad_W, self.grad_b

    def update_weights(self):
        for layer in self.layers:
            self.optimizer.update(layer)

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        history = {"loss": [], "accuracy": []}
        num_samples = X_train.shape[0]

        for _ in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                logits = self.forward(X_batch)
                loss = self.compute_loss(logits, y_batch)
                self.backward(y_batch, logits)
                self.update_weights()

            metrics = self.evaluate(X_train, y_train, batch_size=batch_size)
            history["loss"].append(loss)
            history["accuracy"].append(metrics["accuracy"])

        return history

    def evaluate(self, X, y, batch_size=1024):
        losses = []
        logits_batches = []
        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            batch_logits = self.forward(X[start:end])
            logits_batches.append(batch_logits)
            losses.append(self.compute_loss(batch_logits, y[start:end]))

        logits = np.vstack(logits_batches)
        probabilities = softmax(logits)
        predictions = np.argmax(probabilities, axis=1)
        targets = self._labels_from_target(y)
        accuracy = float(np.mean(predictions == targets))
        return {
            "loss": float(np.mean(losses)),
            "accuracy": accuracy,
            "logits": logits,
            "predictions": predictions,
        }

    def predict_proba(self, X):
        return softmax(self.forward(X))

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        if isinstance(weight_dict, np.ndarray) and weight_dict.shape == ():
            weight_dict = weight_dict.item()
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
                layer.grad_W = np.zeros_like(layer.W)
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
                layer.grad_b = np.zeros_like(layer.b)

    def _ensure_one_hot(self, y):
        y = np.asarray(y)
        if y.ndim == 2:
            return y
        one_hot = np.zeros((y.shape[0], self.output_size))
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1.0
        return one_hot

    def _labels_from_target(self, y):
        y = np.asarray(y)
        if y.ndim == 2:
            return np.argmax(y, axis=1)
        return y.astype(int)
