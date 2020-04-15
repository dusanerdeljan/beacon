from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Represents abstract optimizer.
    """

    def __init__(self, parameters, lr=0.01):
        super().__init__()
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self):
        """
        Updates all the parameters in the neural network.
        """
        pass

    def zero_grad(self):
        """
        Sets gradients of all the parameters in the neural network to zero.
        """
        for parameter in self.parameters:
            parameter.zero_grad()