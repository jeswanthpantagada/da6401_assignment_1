import numpy as np


class Dense:

    def __init__(self, input_size, output_size, weight_init="random"):

        if weight_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01

        self.b = np.zeros((1, output_size))

        self.input = None

        self.grad_W = None
        self.grad_b = None

    def forward(self, X):

        self.input = X

        output = np.dot(X, self.W) + self.b

        return output

    def backward(self, grad_output):

        batch_size = self.input.shape[0]

        self.grad_W = np.dot(self.input.T, grad_output) / batch_size
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True) / batch_size

        grad_input = np.dot(grad_output, self.W.T)

        return grad_input