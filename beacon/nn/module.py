from beacon.nn.parameter import Parameter
from beacon.tensor import functions as fn
from abc import ABC, abstractmethod
from inspect import getmembers
import pickle

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
        return self.forward(fn.to_tensor(x))

    @abstractmethod
    def forward(self, x):
        """
        Abstract method which performs forward pass.
        """
        pass

    def save(self, file_path: str):
        """
        Saves a model to a file.

        ## Parameters
        file_path: `str`
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: str):
        """
        Loads model from a file.

        ## Parameters
        file_path: `str`

        ## Raises
        `RuntimeError` if specified file stores model of different class
        """
        with open(file_path, 'rb') as file:
            loaded_module = pickle.load(file)
        if type(loaded_module) != cls:
            raise RuntimeError("Tried to load a model of different type!")
        return loaded_module