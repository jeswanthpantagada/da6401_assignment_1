import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        layer.W -= self.learning_rate * layer.grad_W
        layer.b -= self.learning_rate * layer.grad_b


class Momentum:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity_W = {}
        self.velocity_b = {}

    def update(self, layer):
        layer_id = id(layer)
        if layer_id not in self.velocity_W:
            self.velocity_W[layer_id] = np.zeros_like(layer.W)
            self.velocity_b[layer_id] = np.zeros_like(layer.b)

        self.velocity_W[layer_id] = self.beta * self.velocity_W[layer_id] + (1.0 - self.beta) * layer.grad_W
        self.velocity_b[layer_id] = self.beta * self.velocity_b[layer_id] + (1.0 - self.beta) * layer.grad_b

        layer.W -= self.learning_rate * self.velocity_W[layer_id]
        layer.b -= self.learning_rate * self.velocity_b[layer_id]


class NAG:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity_W = {}
        self.velocity_b = {}

    def update(self, layer):
        layer_id = id(layer)
        if layer_id not in self.velocity_W:
            self.velocity_W[layer_id] = np.zeros_like(layer.W)
            self.velocity_b[layer_id] = np.zeros_like(layer.b)

        previous_velocity_W = self.velocity_W[layer_id]
        previous_velocity_b = self.velocity_b[layer_id]

        self.velocity_W[layer_id] = self.beta * self.velocity_W[layer_id] - self.learning_rate * layer.grad_W
        self.velocity_b[layer_id] = self.beta * self.velocity_b[layer_id] - self.learning_rate * layer.grad_b

        layer.W += -self.beta * previous_velocity_W + (1.0 + self.beta) * self.velocity_W[layer_id]
        layer.b += -self.beta * previous_velocity_b + (1.0 + self.beta) * self.velocity_b[layer_id]


class RMSProp:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.square_W = {}
        self.square_b = {}

    def update(self, layer):
        layer_id = id(layer)
        if layer_id not in self.square_W:
            self.square_W[layer_id] = np.zeros_like(layer.W)
            self.square_b[layer_id] = np.zeros_like(layer.b)

        self.square_W[layer_id] = self.beta * self.square_W[layer_id] + (1.0 - self.beta) * (layer.grad_W ** 2)
        self.square_b[layer_id] = self.beta * self.square_b[layer_id] + (1.0 - self.beta) * (layer.grad_b ** 2)

        layer.W -= self.learning_rate * layer.grad_W / (np.sqrt(self.square_W[layer_id]) + self.epsilon)
        layer.b -= self.learning_rate * layer.grad_b / (np.sqrt(self.square_b[layer_id]) + self.epsilon)


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = {}
        self.v_W = {}
        self.m_b = {}
        self.v_b = {}
        self.time_step = 0

    def update(self, layer):
        layer_id = id(layer)
        if layer_id not in self.m_W:
            self.m_W[layer_id] = np.zeros_like(layer.W)
            self.v_W[layer_id] = np.zeros_like(layer.W)
            self.m_b[layer_id] = np.zeros_like(layer.b)
            self.v_b[layer_id] = np.zeros_like(layer.b)

        self.time_step += 1

        self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1.0 - self.beta1) * layer.grad_W
        self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1.0 - self.beta2) * (layer.grad_W ** 2)
        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1.0 - self.beta1) * layer.grad_b
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1.0 - self.beta2) * (layer.grad_b ** 2)

        m_hat_W = self.m_W[layer_id] / (1.0 - self.beta1 ** self.time_step)
        v_hat_W = self.v_W[layer_id] / (1.0 - self.beta2 ** self.time_step)
        m_hat_b = self.m_b[layer_id] / (1.0 - self.beta1 ** self.time_step)
        v_hat_b = self.v_b[layer_id] / (1.0 - self.beta2 ** self.time_step)

        layer.W -= self.learning_rate * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
        layer.b -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


class Nadam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = {}
        self.v_W = {}
        self.m_b = {}
        self.v_b = {}
        self.time_step = 0

    def update(self, layer):
        layer_id = id(layer)
        if layer_id not in self.m_W:
            self.m_W[layer_id] = np.zeros_like(layer.W)
            self.v_W[layer_id] = np.zeros_like(layer.W)
            self.m_b[layer_id] = np.zeros_like(layer.b)
            self.v_b[layer_id] = np.zeros_like(layer.b)

        self.time_step += 1

        self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1.0 - self.beta1) * layer.grad_W
        self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1.0 - self.beta2) * (layer.grad_W ** 2)
        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1.0 - self.beta1) * layer.grad_b
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1.0 - self.beta2) * (layer.grad_b ** 2)

        m_hat_W = self.m_W[layer_id] / (1.0 - self.beta1 ** self.time_step)
        v_hat_W = self.v_W[layer_id] / (1.0 - self.beta2 ** self.time_step)
        m_hat_b = self.m_b[layer_id] / (1.0 - self.beta1 ** self.time_step)
        v_hat_b = self.v_b[layer_id] / (1.0 - self.beta2 ** self.time_step)

        nesterov_W = self.beta1 * m_hat_W + ((1.0 - self.beta1) * layer.grad_W) / (1.0 - self.beta1 ** self.time_step)
        nesterov_b = self.beta1 * m_hat_b + ((1.0 - self.beta1) * layer.grad_b) / (1.0 - self.beta1 ** self.time_step)

        layer.W -= self.learning_rate * nesterov_W / (np.sqrt(v_hat_W) + self.epsilon)
        layer.b -= self.learning_rate * nesterov_b / (np.sqrt(v_hat_b) + self.epsilon)


def get_optimizer(name, learning_rate):
    optimizers = {
        "sgd": SGD,
        "momentum": Momentum,
        "nag": NAG,
        "rmsprop": RMSProp,
        "adam": Adam,
        "nadam": Nadam,
    }
    if name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {name}")
    return optimizers[name](learning_rate=learning_rate)
