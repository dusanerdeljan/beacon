from beacon.nn.module import Module, Parameter
from beacon.nn.init import normal, zeros
from beacon.tensor import Tensor

class Conv(Module):

    def __init__(self, input_channels, output_channels, filter_size, mode='valid', filter_initializer=normal, bias_initializer=zeros):
        """
        Convolution module. Inputs to the conv module are of shape (batch_size, input_channels, X, Y)

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
        self.weights = Parameter(shape=(input_channels, output_channels, filter_size, filter_size), initializer=filter_initializer)
        self.biases = Parameter(shape=(1, output_channels, 1, 1), initializer=bias_initializer)

    def forward(self, x):
        raise NotImplementedError("Not yet implemented.")
        
        
