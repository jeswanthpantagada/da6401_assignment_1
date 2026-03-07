import numpy as np


class RMSProp:

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):

        self.lr = learning_rate
        self.beta = beta
        self.eps = epsilon

        self.s_w = {}
        self.s_b = {}

    def update(self, layer):

        if id(layer) not in self.s_w:

            self.s_w[id(layer)] = np.zeros_like(layer.W)
            self.s_b[id(layer)] = np.zeros_like(layer.b)

        self.s_w[id(layer)] = self.beta * self.s_w[id(layer)] + (1 - self.beta) * (layer.grad_W ** 2)
        self.s_b[id(layer)] = self.beta * self.s_b[id(layer)] + (1 - self.beta) * (layer.grad_b ** 2)

        layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s_w[id(layer)]) + self.eps)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[id(layer)]) + self.eps)