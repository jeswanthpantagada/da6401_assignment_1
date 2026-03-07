import numpy as np


class NAG:

    def __init__(self, learning_rate=0.01, beta=0.9):

        self.lr = learning_rate
        self.beta = beta

        self.v_w = {}
        self.v_b = {}

    def update(self, layer):

        if id(layer) not in self.v_w:

            self.v_w[id(layer)] = np.zeros_like(layer.W)
            self.v_b[id(layer)] = np.zeros_like(layer.b)

        v_prev_w = self.v_w[id(layer)]
        v_prev_b = self.v_b[id(layer)]

        self.v_w[id(layer)] = self.beta * self.v_w[id(layer)] + self.lr * layer.grad_W
        self.v_b[id(layer)] = self.beta * self.v_b[id(layer)] + self.lr * layer.grad_b

        layer.W -= (-self.beta * v_prev_w + (1 + self.beta) * self.v_w[id(layer)])
        layer.b -= (-self.beta * v_prev_b + (1 + self.beta) * self.v_b[id(layer)])