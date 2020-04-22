from beacon.nn.module import Module
from beacon.tensor import Tensor
from beacon.tensor import functions as fn
import numpy as np

class Dropout(Module):

    def __init__(self, dropout_rate):
        """
        Dropout module.

        ## Parameters
        droput_rate: `float` - probability that a neuron will be set to zero
        """
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if self.train_mode:
            activation_mask = Tensor(data=np.random.rand(*(x.shape)) / (1-self.dropout_rate), requires_grad=True)
            x = fn.mul(x, activation_mask > self.dropout_rate)
        return x