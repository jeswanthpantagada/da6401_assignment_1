import numpy as np


class Dense:
    def __init__(self, input_size, output_size, weight_init="random"):
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == "zeros":
            self.W = np.zeros((input_size, output_size))
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01

        self.b = np.zeros((1, output_size))
        self.input = None
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x):
        self.input = x
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return x @ self.W + self.b

    def backward(self, grad_output):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            self.grad_W = self.input.T @ grad_output
            grad_input = grad_output @ self.W.T
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return grad_input
