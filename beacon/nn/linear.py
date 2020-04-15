from beacon.nn.module import Module, Parameter
from beacon.tensor import functions as func

class Linear(Module):
    
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weights = Parameter(shape=(inputs, outputs))
        self.bias = Parameter(shape=(1, outputs))

    def forward(self, x):
        return func.matmul(func.to_tensor(x), self.weights) + self.bias