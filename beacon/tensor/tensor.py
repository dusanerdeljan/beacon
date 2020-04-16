import numpy as np 

class Tensor(object):
    """
    Wrapper around numpy's ndarray with automatic computational graph constuction
    and automatic differentiation.
    """

    class ComputationalGraphNode(object):
        """
        Helper class which models a node in the computational graph.
        Stores tensor and derivative function of the primitive operation.
        """
        def __init__(self, tensor, df):
            super().__init__()
            self.tensor = tensor
            self.df = df

    NO_GRAD = False

    def __init__(self, data, requires_grad = False, nodes = None):
        super().__init__()
        self.data = self._to_numpy_ndarray(data)
        self.requires_grad = requires_grad
        self.nodes = nodes or []
        self.grad = None
        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self):
        """
        Sets gradient to zero. Will be used by optimizer in the optimization process.
        """
        self.grad = Tensor(data=np.zeros_like(self.data, dtype=np.float))

    def backward(self, grad=None):
        """
        Performs a backward pass.
        """
        if not self.requires_grad:
            raise RuntimeError("This tensor did not require grad!")
        grad = self._to_tensor(grad) if grad else Tensor(np.ones_like(self.data, dtype=np.float))
        self.grad.data += grad.data
        for node in self.nodes:
            node.tensor.backward(Tensor(data=node.df(grad.data)))

    def __repr__(self):
        """
        String representation.
        """
        return f"<Tensor data={self.data}, requires_grad={self.requires_grad}>"

    ### Redefined operators ###

    def __add__(self, t):
        from beacon.tensor.functions import add
        return add(self, self._to_tensor(t))

    def __radd__(self, t):
        from beacon.tensor.functions import add
        return add(self._to_tensor(t), self)

    def __sub__(self, t):
        from beacon.tensor.functions import sub
        return sub(self, self._to_tensor(t))

    def __rsub__(self, t):
        from beacon.tensor.functions import sub
        return sub(self._to_tensor(t), self)

    def __mul__(self, t):
        from beacon.tensor.functions import mul
        return mul(self, self._to_tensor(t))

    def __rmul__(self, t):
        from beacon.tensor.functions import mul
        return mul(self._to_tensor(t), self)

    def __neg__(self):
        from beacon.tensor.functions import neg
        return neg(self)

    def __truediv__(self, t):
        from beacon.tensor.functions import divide
        return divide(self, self._to_tensor(t))

    def __rtruediv__(self, t):
        from beacon.tensor.functions import divide
        return divide(self._to_tensor(t), self)

    def __pow__(self, t):
        from beacon.tensor.functions import power
        return power(self, self._to_tensor(t))

    def __rpow__(self, t):
        from beacon.tensor.functions import power
        return power(self._to_tensor(t), self)

    def __isub__(self, t):
        self.data = self.data - self._to_tensor(t).data

    def item(self):
        if self.data.shape != ():
            raise RuntimeError("Cannot call item on non-scalar type!")
        return self.data

    def __eq__(self, other):
        return Tensor(self.data == self._to_tensor(other).data)

    def __ne__(self, other):
        return Tensor(self.data != self._to_tensor(other).data)

    def __lt__(self, other):
        return Tensor(self.data < self._to_tensor(other).data)

    def __gt__(self, other):
        return Tensor(self.data > self._to_tensor(other).data)

    def __le__(self, other):
        return Tensor(self.data <= self._to_tensor(other).data)

    def __ge__(self, other):
        return Tensor(self.data >= self._to_tensor(other).data) 

    @classmethod
    def _to_numpy_ndarray(cls, data):
        """
        Convert passed data to numpy array if it isn't already and make sure its shape is (1, x) and not (x,)
        """
        if isinstance(data, np.ndarray):
            return data
        arr = np.array(data, dtype=np.float)
        if len(arr.shape) == 1:
            arr = np.reshape(arr, newshape=(1, arr.shape[0]))
        return arr

    @classmethod
    def _to_tensor(cls, tensor):
        """
        Convert passed tensor to Tensor if it isn't already.
        """
        if isinstance(tensor, Tensor):
            return tensor
        return Tensor(data=tensor)
