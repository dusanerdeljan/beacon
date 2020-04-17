from beacon.nn.parameter import Parameter
from abc import ABC, abstractmethod
from inspect import getmembers

class Module(ABC):

    def __init__(self):
        """
        Represents abstract module in a neural network.
        """
        super().__init__()

    def parameters(self):
        """
        Returns all the parameters from the subclass module
        """
        params = []
        for _, param in getmembers(self):
            if isinstance(param, Parameter):
                params.append(param)
            elif isinstance(param, Module):
                params.extend(param.parameters())
        return params

    def __call__(self, x):
        """
        Redefined call operator.
        """
        return self.forward(x)

    @abstractmethod
    def forward(self, x):
        """
        Abstract method which performs forward pass.
        """
        pass