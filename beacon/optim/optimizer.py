from beacon.nn import beacon
from abc import ABC, abstractmethod
from beacon.tensor import Tensor

class Optimizer(ABC):
    """
    Represents abstract optimizer.
    """

    def __init__(self, parameters, lr=0.01):
        super().__init__()
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """
        Updates all the parameters in the neural network.
        """
        with beacon.no_grad():
            self._step()

    def zero_grad(self):
        """
        Sets gradients of all the parameters in the neural network to zero.
        """
        for parameter in self.parameters:
            parameter.zero_grad()

    @abstractmethod
    def _step(self):
        """
        Updates all the parameters in the neural network.
        """
        pass