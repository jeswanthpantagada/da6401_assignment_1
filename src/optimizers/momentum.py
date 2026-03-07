import numpy as np


class Momentum:

    def __init__(self, learning_rate=0.01, beta=0.9):

        self.lr = learning_rate
        self.beta = beta

        self.v_w = {}
        self.v_b = {}

    def update(self, layer):

        if id(layer) not in self.v_w:

            self.v_w[id(layer)] = np.zeros_like(layer.W)
            self.v_b[id(layer)] = np.zeros_like(layer.b)

        self.v_w[id(layer)] = self.beta * self.v_w[id(layer)] + (1 - self.beta) * layer.grad_W
        self.v_b[id(layer)] = self.beta * self.v_b[id(layer)] + (1 - self.beta) * layer.grad_b

        layer.W -= self.lr * self.v_w[id(layer)]
        layer.b -= self.lr * self.v_b[id(layer)]