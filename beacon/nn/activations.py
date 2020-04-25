from beacon.nn.module import Module
from beacon.functional import functions as F
from abc import abstractmethod

class Activation(Module):
    @abstractmethod
    def __init__(self):
        """
        Represents abstract activation function layer.
        """
        self.activation = None

    def forward(self, x):
        return self.activation(x)

class Sigmoid(Activation):
    def __init__(self):
        """
        Sigmoid activation function.
        """
        self.activation = F.sigmoid

class Softplus(Activation):
    def __init__(self):
        """
        Softplus activation function.
        """
        self.activation = F.softplus

class Softsign(Activation):
    def __init__(self):
        """
        Softsign activation function.
        """
        self.activation = F.softsign

class Softmax(Activation):
    def __init__(self):
        """
        Softmax activation function.
        """
        self.activation = F.softmax

class HardSigmoid(Activation):
    def __init__(self):
        """
        Hard sigmoid activation function
        """
        self.activation = F.hard_sigmoid

class ReLU(Activation):
    def __init__(self, alpha=0.0):
        """
        ReLU activation function.

        ## Parameters
        alpha: `float` - scale factor, defaults to 0.0
        """
        self.activation = lambda x: F.relu(x, alpha)

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        """
        Leaky ReLU activation function.

        ## Parameters
        alpha: `float` - scale factor, defaults to 0.01
        """
        self.activation = lambda x: F.leaky_relu(x, alpha)

class ELU(Activation):
    def __init__(self, alpha=1.0):
        """
        ELU activation function.

        ## Parameters
        alpha: `float` - scale factor, defaults to 1.0
        """
        self.activation = lambda x: F.elu(x, alpha)

class Tanh(Activation):
    def __init__(self):
        """
        Tanh activation function.
        """
        self.activation = F.tanh