from beacon.tensor import Tensor
import numpy as np

class Parameter(Tensor):
    def __init__(self, shape=(1, 1)):
        super().__init__(data=np.random.randn(shape[0], shape[1]), requires_grad=True, nodes=None)