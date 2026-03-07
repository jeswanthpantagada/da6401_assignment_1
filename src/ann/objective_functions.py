import numpy as np


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


class CrossEntropyLoss:
    def forward(self, logits, y_true):
        probabilities = softmax(logits)
        clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        return -np.sum(y_true * np.log(clipped)) / logits.shape[0]

    def backward(self, logits, y_true):
        probabilities = softmax(logits)
        return (probabilities - y_true) / logits.shape[0]


class MSELoss:
    def forward(self, logits, y_true):
        probabilities = softmax(logits)
        return np.mean((probabilities - y_true) ** 2)

    def backward(self, logits, y_true):
        probabilities = softmax(logits)
        grad_prob = 2.0 * (probabilities - y_true) / logits.shape[0]
        dot = np.sum(grad_prob * probabilities, axis=1, keepdims=True)
        return probabilities * (grad_prob - dot)
