from beacon.optim.optimizer import Optimizer
from beacon.tensor import functions as fn

class Adamax(Optimizer):

    def __init__(self, parameters, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(parameters, lr=lr)
        self.beta1 = fn.to_tensor(beta1)
        self.beta2 = fn.to_tensor(beta2)
        self.epsilon = fn.to_tensor(epsilon)
        self.ms = [fn.zeros_like(p) for p in self.parameters]
        self.vs = [fn.zeros_like(p) for p in self.parameters]

    def _step(self, epoch):
        t = fn.to_tensor(epoch)
        grads = [p.grad for p in self.parameters]
        for p, g, m, v in (zip(self.parameters, grads, self.ms, self.vs)):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = fn.maximum(self.beta2*v, fn.abs(g))
            lr_t = self.lr / (1 - fn.power(self.beta1, t))
            p -= lr_t * m / (v + self.epsilon)
