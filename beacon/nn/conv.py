from beacon.nn.module import Module, Parameter
from beacon.nn.init import normal, zeros
from beacon.tensor import Tensor
from beacon.tensor import functions as fn
import numpy as np
from math import floor

class Convolution(Module):

    def __init__(self, input_channels, output_channels, filter_size, stride=1, padding=0, filter_initializer=normal, bias_initializer=zeros):
        """
        Convolution module.

        ## Parameters
        input_channels: `int` - Number of input channels

        output_channels: `int` - Number of filters

        filter_size: `int` - Size of a single filter. Each filter is filter_size x filter_size

        stride: `int` - Stride, defaults to 1

        padding: `int` - Padding from each side, deaults to 0

        filter_initializer: `callable` - Defaults to normal

        bias_initializer: `callable` - Defaults to zeros
        """
        super().__init__()
        self.weight = Parameter(shape=(output_channels, filter_size, filter_size, input_channels), initializer=filter_initializer)
        self.biases = Parameter(shape=(output_channels, 1, 1, 1), initializer=bias_initializer)
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        new_size = floor((x.shape[0] + 2*self.padding - self.filter_size) / self.stride) + 1
        conv = Tensor(data=np.zeros(shape=(new_size, new_size, self.out_channels)), requires_grad=True)
        for dim in range(self.weight.shape[0]):                 # iterate over all the filters
            for i in range(0, new_size, self.stride):           # iterate over height of the input
                for j in range(0, new_size, self.stride):       # iterate over width of the input
                    conv[...,dim] = fn.sum(self.weight[dim,...] * x[i:i+self.filter_size, j:j+self.filter_size, :]) + self.biases[dim,...]
        return conv
