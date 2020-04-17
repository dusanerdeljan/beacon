from beacon.optim.optimizer import Optimizer
from beacon.tensor import functions as fn

class Adagrad(Optimizer):

    def __init__(self, parameters, lr=0.01, epsilon=1e-8):
        """
        Adagrad optimizer.
        """
        super().__init__(parameters, lr=lr)
        self.epison = epsilon
        self.G = [fn.zeros_like(p) for p in self.parameters]

    def _step(self, epoch):
        grads = [p.grad for p in self.parameters]
        for p, g, gs in zip(self.parameters, grads, self.G):
            gs += fn.square(g)
            p -= self.lr * g / fn.sqrt(gs + self.epison)
