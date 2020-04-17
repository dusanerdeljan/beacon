from beacon.optim.optimizer import Optimizer
from beacon.tensor import functions as fn

class Nesterov(Optimizer):

    def __init__(self, parameters, lr=0.01, momentum=0.9):
        super().__init__(parameters, lr=lr)
        self.momentum = fn.to_tensor(momentum)
        self.vs = [fn.zeros_like(p) for p in self.parameters]

    def _step(self, epoch):
        grads = [p.grad for p in self.parameters]
        for p, g, v in zip(self.parameters, grads, self.vs):
            prev = v
            v = self.momentum * v - self.lr * g
            p -= self.momentum * prev - (1.0 + self.momentum) * v
