from beacon.nn.module import Module
from beacon.nn import Linear
from beacon.functional import functions as F

class GRU(Module):

    def __init__(self, inputs, outputs):
        """
        GRU Module - Fully gated unit.

        ## Parameters
        inputs: `int` - number of input features

        outputs: `int` - number of output features

        activation: `callable` - activation function, defaults to Tanh
        """
        super().__init__()
        self._z_x = Linear(inputs, outputs)
        self._r_x = Linear(inputs, outputs)
        self._h_x = Linear(inputs, outputs)
        self._z_h = Linear(outputs, outputs, use_bias=False)
        self._r_h = Linear(outputs, outputs, use_bias=False)
        self._h_h = Linear(outputs, outputs, use_bias=False)
        self._h = None

    def forward(self, x):
        if self._h is None:
            z = F.sigmoid(self._z_x(x))
            r = F.sigmoid(self._r_x(x))
        else:
            z = F.sigmoid(self._z_x(x) + self._z_h(self._h))
            r = F.sigmoid(self._r_x(x) + self._r_h(self._h))
        self._h = z * self._h + (1 - z) * F.tanh(self._h_x(x) + self._h_h(r * self._h))
        return self._h

