from beacon.nn.module import Module
from beacon.nn import Linear
from beacon.functional import functions as F

class PeepholeLSTM(Module):

    def __init__(self, inputs, outputs):
        """
        Peephole LSTM Module.

        ## Parameters
        inputs: `int` - number of input features

        outputs: `int` - number of output features
        """
        super().__init__()
        self._x_f = Linear(inputs, outputs)
        self._x_i = Linear(inputs, outputs)
        self._x_o = Linear(inputs, outputs)
        self._h_f = Linear(outputs, outputs, use_bias=False)
        self._h_i = Linear(outputs, outputs, use_bias=False)
        self._h_o = Linear(outputs, outputs, use_bias=False)
        self._c_x = Linear(inputs, outputs)
        self._h = None
        self._c = None

    def forward(self, x):
        if self._c is None:
            f = F.sigmoid(self._x_f(x))
            i = F.sigmoid(self._x_i(x))
            o = F.sigmoid(self._x_o(x))
        else:
            f = F.sigmoid(self._x_f(x) + self._h_f(self._c))
            i = F.sigmoid(self._x_i(x) + self._h_i(self._c))
            o = F.sigmoid(self._x_o(x) + self._h_o(self._c))
        self._c = f * self._c + i * self._c_x(x)
        self._h = o * self._c
    
