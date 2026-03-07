import numpy as np

from models.dense import Dense
from models.activations import ReLU, Sigmoid, Tanh
from models.losses import Softmax, CrossEntropyLoss, MSELoss


class NeuralNetwork:

    def __init__(self, input_size, hidden_layers, num_neurons,
                 activation="relu", loss="cross_entropy", weight_init="random"):

        self.layers = []

        activation_map = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh
        }

        activation_class = activation_map[activation]

        prev_size = input_size

        # Hidden layers
        for _ in range(hidden_layers):
            self.layers.append(Dense(prev_size, num_neurons, weight_init))
            self.layers.append(activation_class())
            prev_size = num_neurons

        # Output layer
        self.layers.append(Dense(prev_size, 10, weight_init))

        self.softmax = Softmax()

        if loss == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = MSELoss()


    def forward(self, X):

        output = X

        for layer in self.layers:
            output = layer.forward(output)

        output = self.softmax.forward(output)

        return output


    def compute_loss(self, y_pred, y_true):

        return self.loss_fn.forward(y_pred, y_true)


    def backward(self, y_pred, y_true):

        grad = self.loss_fn.backward(y_pred, y_true)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad