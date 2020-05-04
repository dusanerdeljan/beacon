from beacon.nn.module import Module
from beacon.nn import Linear
from beacon.functional import functions as F

class RNN(Module):

    def __init__(self, inputs, outputs, activation=F.tanh):
        """
        RNN Module.

        ## Parameters
        inputs: `int` - number of input features

        outputs: `int` - number of output features

        activation: `callable` - activation function, defaults to Tanh
        """
        super().__init__()
        self._input_hidden = Linear(inputs, outputs)
        self._hidden_hidden = Linear(inputs, outputs, use_bias=False)
        self._activation = activation
        self._hidden = None

    def forward(self, x):
        if self._hidden is None:
            self._hidden = self._activation(self._input_hidden(x))
        else:
            self._hidden = self._activation(self._input_hidden(x) + self._hidden_hidden(self._hidden))
        return self._hidden

