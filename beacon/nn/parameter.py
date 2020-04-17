from beacon.tensor import Tensor
from beacon.nn.init import normal

class Parameter(Tensor):
    def __init__(self, shape=(1, 1), initializer=normal):
        """
        Represents one paraemeter in a neural network

        ## Prameters:
        shape: `tuple` - shape of the parameter
        
        initializer: `callable` - parameter initializer, defaults to normal
        """
        super().__init__(data=initializer(shape), requires_grad=True, nodes=None)