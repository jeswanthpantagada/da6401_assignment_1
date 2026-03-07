class SGD:

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, layer):

        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b