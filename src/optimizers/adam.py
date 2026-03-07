import numpy as np


class Adam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}

        self.t = 0

    def update(self, layer):

        if id(layer) not in self.m_w:

            self.m_w[id(layer)] = np.zeros_like(layer.W)
            self.v_w[id(layer)] = np.zeros_like(layer.W)
            self.m_b[id(layer)] = np.zeros_like(layer.b)
            self.v_b[id(layer)] = np.zeros_like(layer.b)

        self.t += 1

        self.m_w[id(layer)] = self.beta1 * self.m_w[id(layer)] + (1 - self.beta1) * layer.grad_W
        self.v_w[id(layer)] = self.beta2 * self.v_w[id(layer)] + (1 - self.beta2) * (layer.grad_W ** 2)

        m_hat = self.m_w[id(layer)] / (1 - self.beta1 ** self.t)
        v_hat = self.v_w[id(layer)] / (1 - self.beta2 ** self.t)

        layer.W -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)