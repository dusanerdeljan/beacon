from beacon.nn.module import Module, Parameter
from beacon.nn.init import normal, zeros
from beacon.tensor import Tensor
from beacon.tensor import functions as fn
import numpy as np

class Conv(Module):

    def __init__(self, input_channels, output_channels, filter_size, mode='valid', filter_initializer=normal, bias_initializer=zeros):
        """
        Convolution module. Inputs to the conv module are of shape (input_channels, X, Y)

        ## Parameters
        input_channels: `int` - Number of input channels

        output_channels: `int` - Number of filters

        filter_size: `int` - Size of a single filter. Each filter is filter_size x filter_size

        mode: `str` - Convolution mode, deaults to valid

        filter_initializer: `callable` - Defaults to normal

        bias_initializer: `callable` - Defaults to zeros
        """
        super().__init__()
        self.mode = mode
        self.weights = [Parameter(shape=(input_channels, filter_size, filter_size), initializer=lambda shape: np.ones(shape=shape)) \
            for _ in range(output_channels)]
        self.biases = [Parameter(shape=(1,1,1), initializer=bias_initializer) for _ in range(output_channels)]

    def forward(self, x):
        filters = [fn.convolve(x, W, self.mode) + bias for W, bias in zip(self.weights, self.biases)]
        return fn.concatenate(filters, axis=0)
        
