from beacon.nn.module import Module, Parameter
from beacon.nn.init import normal, zeros
from beacon.tensor import Tensor
from beacon.tensor import functions as fn

class Conv(Module):

    def __init__(self, input_channels, output_channels, filter_size, stride=(1,1), padding=(0,0), filter_initializer=normal, bias_initializer=zeros):
        """
        Convolution module. Inputs to the conv module are of shape (batch_size, input_channels, height, width)

        ## Parameters
        input_channels: `int` - Number of input channels

        output_channels: `int` - Number of filters

        filter_size: `int` - Size of a single filter. Each filter is filter_size x filter_size

        stride: `tuple` - Stride, defaults to (1, 1)

        padding: `tuple` - Padding, defaults to (0, 0)

        filter_initializer: `callable` - Defaults to normal

        bias_initializer: `callable` - Defaults to zeros
        """
        super().__init__()
        self._stride = stride
        self._padding = padding
        self.weights = Parameter(shape=(output_channels, input_channels, filter_size, filter_size), initializer=filter_initializer)
        self.biases = Parameter(shape=(1, output_channels, 1, 1), initializer=bias_initializer)

    def forward(self, x):
        return fn.conv(x, self.weights, self._stride, self._padding) + self.biases
        
        
