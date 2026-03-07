import numpy as np


class Softmax:

    def __init__(self):
        self.output = None

    def forward(self, X):

        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
        return probabilities


class CrossEntropyLoss:

    def forward(self, y_pred, y_true):

        samples = y_pred.shape[0]

        y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)

        loss = -np.sum(y_true * np.log(y_pred_clipped)) / samples

        return loss

    def backward(self, y_pred, y_true):

        samples = y_pred.shape[0]

        grad = (y_pred - y_true) / samples

        return grad


class MSELoss:

    def forward(self, y_pred, y_true):

        loss = np.mean((y_pred - y_true) ** 2)

        return loss

    def backward(self, y_pred, y_true):

        samples = y_pred.shape[0]

        grad = 2 * (y_pred - y_true) / samples

        return grad