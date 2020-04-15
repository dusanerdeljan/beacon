from abc import ABC, abstractmethod

class Optimizer(ABC):

    def __init__(self, parameters, lr=0.01):
        super().__init__()
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()