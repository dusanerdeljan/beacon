from beacon.tensor import Tensor

class beacon(object):
    """
    Provides no gradient block
    """

    class BeaconNoGrad(object):
        def __init__(self):
            super().__init__()

        def __enter__(self):
            Tensor.NO_GRAD = True

        def __exit__(self, type, value, traceback):
            Tensor.NO_GRAD = False

    @staticmethod
    def no_grad():
       return beacon.BeaconNoGrad()