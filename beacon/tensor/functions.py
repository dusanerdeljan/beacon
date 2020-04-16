from beacon.tensor import Tensor
import numpy as np

###########################################
### Definitions of Jacobian vector products
###########################################

def to_tensor(x):
    return Tensor._to_tensor(x)

def broadcast(target_grad, input_grad):
    while np.ndim(input_grad) > np.ndim(target_grad):
        input_grad = np.sum(input_grad, axis=0)
    for axis, dim in enumerate(np.shape(target_grad)):
        if dim == 1:
            input_grad = np.sum(input_grad, axis=axis, keepdims=True)
    return input_grad

def match_shape(x, shape, axis, keepdims):
    if shape == ():
        return x, 1
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = np.array(shape)
    new_shape[axis] = 1
    num_reps = np.prod(np.array(shape)[axis])
    return np.reshape(x, new_shape) + np.zeros(shape, dtype=np.float), num_reps

def add(t1: Tensor, t2: Tensor):
    data = t1.data + t2.data
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: broadcast(t1.grad.data, x)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: broadcast(t2.grad.data, x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def mul(t1: Tensor, t2: Tensor):
    data = t1.data * t2.data
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: broadcast(t1.grad.data, t2.data*x)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: broadcast(t2.grad.data, t1.data*x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sub(t1: Tensor, t2: Tensor):
    data = t1.data - t2.data
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: broadcast(t1.grad.data, x)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: broadcast(t2.grad.data, -x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def divide(t1: Tensor, t2: Tensor):
    data = t1.data / t2.data
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: broadcast(t1.grad.data, x /t2.data)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: broadcast(t2.grad.data, (-x * t1.data)/(t2.data**2))))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sum(t: Tensor, axis=None, keepdims=False):
    data = np.sum(t.data, axis=axis, keepdims=keepdims)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: match_shape(x, t.data.shape, axis, keepdims)[0]))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def matmul(t1: Tensor, t2: Tensor):
    data = np.matmul(t1.data, t2.data)
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: np.matmul(x, t2.data.T)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: np.matmul(t1.data.T, x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def neg(t: Tensor):
    data = -t.data
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: -x))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def exp(t: Tensor):
    data = np.exp(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: data*x))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def tanh(t: Tensor):
    data = np.tanh(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x / np.cosh(t.data)**2))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sinh(t: Tensor):
    data = np.sinh(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x*np.cosh(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def cosh(t: Tensor):
    data = np.tanh(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x*np.sinh(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def maximum(t1: Tensor, t2: Tensor):
    data = np.maximum(t1.data, t2.data)
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    def max_grad(x, z, y):
        return (x == z) / (1.0 + (x == y))
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: broadcast(t1.data, x * max_grad(t1.data, data, t2.data))))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: broadcast(t2.data, x * max_grad(t2.data, data, t1.data))))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def minimum(t1: Tensor, t2: Tensor):
    data = np.minimum(t1.data, t2.data)
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    def min_grad(x, z, y):
        return (x == z) / (1.0 + (x == y))
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: broadcast(t1.data, x * min_grad(t1.data, data, t2.data))))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: broadcast(t2.data, x * min_grad(t2.data, data, t1.data))))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def clip(t: Tensor, min_val, max_val):
    data = np.clip(t.data, min_val, max_val)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x * np.logical_and(data != min_val, data != max_val)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def power(t1: Tensor, t2: Tensor):
    data = np.power(t1.data, t2.data)
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, 
        df=lambda x: broadcast(t1.data, x*t2.data*(t1.data**np.where(t2.data, t2.data-1, 1.)))))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, 
        df=lambda x: broadcast(t2.data, x * np.log((np.where(t1.data, t1.data, 1.)) * data))))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def abs(t: Tensor):
    data = np.abs(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: np.sign(t.data)*x))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def log(t: Tensor):
    data = np.log(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x / t.data))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sqrt(t: Tensor):
    data = np.sqrt(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: 0.5*x*(t.data**-0.5)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def square(t: Tensor):
    data = np.square(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: 2*x*t.data))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def reshape(t: Tensor, shape):
    data = np.reshape(t.data, newshape=shape)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: np.reshape(x, np.shape(t.data))))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sin(t: Tensor):
    data = np.sin(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x * np.cos(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def cos(t: Tensor):
    data = np.cos(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: -x * np.sin(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def tan(t: Tensor):
    data = np.tan(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x / np.cos(t.data)**2))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def mean(t: Tensor, axis=None, keepdims=False):
    data = np.mean(t.data, axis=axis, dtype=np.float, keepdims=keepdims)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        def mean_grad(x):
            g, n = match_shape(x, np.shape(t.data), axis, keepdims)
            return g / n
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: mean_grad(x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def where(condition: Tensor, t1: Tensor=None, t2: Tensor=None):
    t1 = to_tensor(t1) if t1 is not None else None
    t2 = to_tensor(t2) if t2 is not None else None
    data = np.where(condition.data, t1.data if t1 else None, t2.data if t2 else None)
    t1g = to_tensor(t1).requires_grad if t1 else False
    t2g = to_tensor(t2).requires_grad if t2 else False
    requires_grad = (t1g or t2g) and not Tensor.NO_GRAD
    nodes = []
    if t1g:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: np.where(condition.data, x, np.zeros_like(x))))
    if t2g:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: np.where(condition.data, np.zeros_like(x), x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def zeros_like(t: Tensor):
    return to_tensor(np.zeros_like(t.data))