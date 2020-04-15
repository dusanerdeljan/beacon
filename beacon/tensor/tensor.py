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
        grad = grad or Tensor(np.ones_like(self.data, dtype=np.float))
        self.grad.data += grad.data
        for node in self.nodes:
            node.tensor.backward(Tensor(data=node.df(grad.data)))

    def __repr__(self):
        """
        String representation.
        """
        return f"Tensor data: {self.data}, requires_grad={self.requires_grad}"

    @classmethod
    def _to_numpy_ndarray(cls, data):
        """
        Convert passed data to numpy array if it isn't already.
        """
        if isinstance(data, np.ndarray):
            return data
        arr = np.array(data, dtype=np.float)
        if len(arr.shape) == 1:
            arr = np.reshape(arr, newshape=(arr.shape[0], 1))
        return arr

    @classmethod
    def _to_tensor(cls, tensor):
        """
        Convert passed tensor to Tensor if it isn's already.
        """
        if isinstance(tensor, Tensor):
            return tensor
        return Tensor(data=tensor)


def broadcast(target_grad, input_grad):
    while np.ndim(input_grad) > np.ndim(target_grad):
        input_grad = np.sum(input_grad, axis=0)
    for axis, dim in enumerate(np.shape(target_grad)):
        if dim == 1:
            input_grad = np.sum(input_grad, axis=axis, keepdims=True)
    return input_grad

def add(t1: Tensor, t2: Tensor):
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: broadcast(t1.grad, x)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: broadcast(t2.grad, x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)