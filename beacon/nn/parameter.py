from beacon.tensor import Tensor
from beacon.nn.init import normal

class Parameter(Tensor):
    def __init__(self, shape=(1, 1), initializer=normal):
        super().__init__(data=initializer(shape), requires_grad=True, nodes=None)