import numpy as np


class ReLU:

    def __init__(self):
        self.input = None

    def forward(self, X):

        self.input = X
        return np.maximum(0, X)

    def backward(self, grad_output):

        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input


class Sigmoid:

    def __init__(self):
        self.output = None

    def forward(self, X):

        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward(self, grad_output):

        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input


class Tanh:

    def __init__(self):
        self.output = None

    def forward(self, X):

        self.output = np.tanh(X)
        return self.output

    def backward(self, grad_output):

        grad_input = grad_output * (1 - self.output ** 2)
        return grad_input