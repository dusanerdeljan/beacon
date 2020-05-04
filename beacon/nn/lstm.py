from beacon.nn.module import Module
from beacon.nn import Linear
from beacon.functional import functions as F

class LSTM(Module):

    def __init__(self, inputs, outputs):
        """
        LSTM Module.

        ## Parameters
        inputs: `int` - number of input features

        outputs: `int` - number of output features
        """
        super().__init__()
        self._x_f = Linear(inputs, outputs)
        self._x_i = Linear(inputs, outputs)
        self._x_o = Linear(inputs, outputs)
        self._x_u = Linear(inputs, outputs)
        self._h_f = Linear(outputs, outputs, use_bias=False)
        self._h_i = Linear(outputs, outputs, use_bias=False)
        self._h_o = Linear(outputs, outputs, use_bias=False)
        self._h_u = Linear(outputs, outputs, use_bias=False)
        self._h = None
        self._c = None

    def forward(self, x):
        if self._h is None:
            f = F.sigmoid(self._x_f(x))
            i = F.sigmoid(self._x_i(x))
            o = F.sigmoid(self._x_o(x))
            u = F.tanh(self._x_u(x))
        else:
            f = F.sigmoid(self._x_f(x) + self._h_f(self._h))
            i = F.sigmoid(self._x_i(x) + self._h_i(self._h))
            o = F.sigmoid(self._x_o(x) + self._h_o(self._h))
            u = F.tanh(self._x_u(x) + self._h_u(self._h))
        self._c = i * u if self._c is None else (f * self._c) + (i * u)
        self._h = o * F.tanh(self._c)
        return self._h
    
