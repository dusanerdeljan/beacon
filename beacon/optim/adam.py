from beacon.optim.optimizer import Optimizer
from beacon.tensor import functions as fn

class Adam(Optimizer):

    def __init__(self, parameters, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(parameters, lr=lr)
        self.beta1 = fn.to_tensor(beta1)
        self.beta2 = fn.to_tensor(beta2)
        self.epsilon = fn.to_tensor(epsilon)
        self.ms = [fn.zeros_like(p) for p in self.parameters]
        self.vs = [fn.zeros_like(p) for p in self.parameters]
        self.t = fn.to_tensor(0.0)

    def _step(self):
        self.t = self.t + 1
        grads = [p.grad for p in self.parameters]
        for p, g, m, v in (zip(self.parameters, grads, self.ms, self.vs)):
            m = self.beta1*m + (1-self.beta1)*g
            v = self.beta2*v + (1-self.beta2)*fn.square(g)
            m_hat = m / (1 - fn.power(self.beta1, self.t))
            v_hat = v / (1 - fn.power(self.beta2, self.t))
            p -= self.lr * m_hat / (fn.sqrt(v_hat) + self.epsilon)
