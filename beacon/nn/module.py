from beacon.nn.parameter import Parameter
from abc import ABC, abstractmethod
from inspect import getmembers

class Module(ABC):
    def __init__(self):
        super().__init__()

    def parameters(self):
        params = []
        for name, value in getmembers(self):
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)

    @abstractmethod
    def forward(self, x):
        pass