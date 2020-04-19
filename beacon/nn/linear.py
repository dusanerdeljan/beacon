from beacon.nn.module import Module, Parameter
from beacon.nn.init import normal, zeros
from beacon.tensor import functions as fn

class Linear(Module):
    
    def __init__(self, inputs, outputs, weight_initializer=normal, bias_initializer=zeros):
        """
        Linear module (fully-connected layer).

        ## Parameters
        inputs: `int` - number of incomming inputs

        output: `int` - number of neurons

        weight_initializer: `callable` - defaults to normal
        
        bias_initializer: `callable` - defaults to zeros
        """
        super().__init__()
        self.weights = Parameter(shape=(inputs, outputs), initializer=normal)
        self.bias = Parameter(shape=(1, outputs), initializer=zeros)

    def forward(self, x):
        return fn.matmul(x, self.weights) + self.bias