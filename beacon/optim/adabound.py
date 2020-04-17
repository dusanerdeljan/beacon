from beacon.optim.optimizer import Optimizer
from beacon.tensor import functions as fn

class Adabound(Optimizer):

    def __init__(self, parameters, lr=0.01, beta1=0.9, beta2=0.999, final_lr=0.1, gamma=1e-3, epsilon=1e-8):
        """
        Adabound optimizer.
        """
        super().__init__(parameters, lr=lr)
        self.beta1 = fn.to_tensor(beta1)
        self.beta2 = fn.to_tensor(beta2)
        self.epsilon = fn.to_tensor(epsilon)
        self.final_lr = fn.to_tensor(final_lr)
        self.gamma = fn.to_tensor(gamma)
        self.ms = [fn.zeros_like(p) for p in self.parameters]
        self.vs = [fn.zeros_like(p) for p in self.parameters]

    def _step(self, epoch):
        t = fn.to_tensor(epoch)
        step_size = self.lr * (fn.sqrt(1 - fn.power(self.beta2, t)) / (1 - fn.power(self.beta1, t)))
        lower_bound = self.final_lr * (1.0 - 1.0 / (self.gamma*t + 1))
        upper_bound = self.final_lr * (1.0 + 1.0 / (self.gamma*t))
        grads = [p.grad for p in self.parameters]
        for p, g, m, v in (zip(self.parameters, grads, self.ms, self.vs)):
            m = self.beta1*m + (1-self.beta1)*g
            v = self.beta2*v + (1-self.beta2)*fn.square(g)
            denom = fn.sqrt(v) + self.epsilon
            p -= m * fn.clip(step_size / denom, lower_bound.item(), upper_bound.item())
