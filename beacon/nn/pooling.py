from beacon.nn.module import Module
from beacon.functional import functions as F
from abc import abstractmethod

class PoolingLayer(Module):
    @abstractmethod
    def __init__(self, kernel, stride=None, padding=(0,0)):
        """
        Represents abstract pooling layer.
        """
        super().__init__()
        self.pool_func = None
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return self.pool_func(x, self.kernel, self.stride, self.padding)

class MaxPool(PoolingLayer):

    def __init__(self, kernel, stride=None, padding=(0,0)):
        """
        Max pooling layer.
        """
        super().__init__(kernel, stride, padding)
        self.pool_func = F.max_pool

class AveragePool(PoolingLayer):

    def __init__(self, kernel, stride=None, padding=(0,0)):
        """
        Average pooling layer.
        """
        super().__init__(kernel, stride=stride, padding=padding)
        self.pool_func = F.average_pool
