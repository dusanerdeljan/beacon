from beacon.nn import beacon
from abc import ABC, abstractmethod
from beacon.tensor import Tensor
from beacon.tensor import functions as fn

class Optimizer(ABC):
    def __init__(self, parameters, lr=0.01):
        """
        Represents abstract optimizer.

        ## Parameters
        parameters: list(Parameters) - all the parameters in the neural network
        lr: `float` - learning rate (step size)
        """
        super().__init__()
        self.parameters = parameters
        self.lr = lr

    def step(self, epoch=1):
        """
        Updates all the parameters in the neural network.
        """
        with beacon.no_grad():
            self._step(epoch)

    def zero_grad(self):
        """
        Sets gradients of all the parameters in the neural network to zero.
        """
        for parameter in self.parameters:
            parameter.zero_grad()

    @abstractmethod
    def _step(self, epoch):
        """
        Updates all the parameters in the neural network.
        """
        pass