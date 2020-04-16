from beacon.nn.module import Module, Parameter
from beacon.nn.init import normal, zeros
from beacon.tensor import functions as fn

class Linear(Module):
    
    def __init__(self, inputs, outputs, weight_initializer=normal, bias_initializer=zeros):
        super().__init__()
        self.weights = Parameter(shape=(inputs, outputs), initializer=normal)
        self.bias = Parameter(shape=(1, outputs), initializer=zeros)

    def forward(self, x):
        return fn.matmul(fn.to_tensor(x), self.weights) + self.bias