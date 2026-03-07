import numpy as np


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0.0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0.0] = 0.0
        return grad_input


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1.0 - self.output)


class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1.0 - self.output ** 2)
