from beacon.optim.optimizer import Optimizer
from beacon.tensor import functions as fn

class RMSProp(Optimizer):

    def __init__(self, parameters, lr=0.01, beta=0.99, epsilon=1e-8):
        super().__init__(parameters, lr=lr)
        self.epsilon = epsilon
        self.beta = fn.to_tensor(beta)
        self.E = [fn.zeros_like(p) for p in self.parameters]

    def _step(self, epoch):
        grads = [p.grad for p in self.parameters]
        for p, g, e in zip(self.parameters, grads, self.E):
            e = self.beta * e + (1 - self.beta) * fn.square(g)
            p -= self.lr * g / (fn.sqrt(e) + self.epsilon)