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

def add(t1: Tensor, t2: Tensor):
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: broadcast(t1.grad.data, x)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: broadcast(t2.grad.data, x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def mul(t1: Tensor, t2: Tensor):
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: broadcast(t1.grad.data, t2.data*x)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: broadcast(t2.grad.data, t1.data*x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sub(t1: Tensor, t2: Tensor):
    data = t1.data - t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: broadcast(t1.grad.data, x)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: broadcast(t2.grad.data, -x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def divide(t1: Tensor, t2: Tensor):
    data = t1.data / t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: broadcast(t1.grad.data, x /t2.data)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: broadcast(t2.grad.data, (-x * t1.data)/(t2.data**2))))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sum(t: Tensor):
    data = t.data.sum()
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x * np.ones_like(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def matmul(t1: Tensor, t2: Tensor):
    data = np.matmul(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: np.matmul(x, t2.data.T)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: np.matmul(t1.data.T, x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def neg(t: Tensor):
    data = -t.data
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: -x))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def exp(t: Tensor):
    data = np.exp(t.data)
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: data*x))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def tanh(t: Tensor):
    data = np.tanh(t.data)
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x / np.cosh(t.data)**2))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sinh(t: Tensor):
    data = np.sinh(t.data)
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x*np.cosh(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def cosh(t: Tensor):
    data = np.tanh(t.data)
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x*np.sinh(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def maximum(t1: Tensor, t2: Tensor):
    data = np.maximum(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
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
    requires_grad = t1.requires_grad or t2.requires_grad
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
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x * np.logical_and(data != min_val, data != max_val)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def power(t1: Tensor, t2: Tensor):
    data = np.power(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
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
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: np.sign(t.data)*x))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def log(t: Tensor):
    data = np.log(t.data)
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x / t.data))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sqrt(t: Tensor):
    data = np.sqrt(t.data)
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: 0.5*x*(t.data**-0.5)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def square(t: Tensor):
    data = np.square(t.data)
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: 2*x*t.data))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def reshape(t: Tensor, shape):
    data = np.reshape(t.data, newshape=shape)
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: np.reshape(x, np.shape(t.data))))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sin(t: Tensor):
    data = np.sin(t.data)
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x * np.cos(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def cos(t: Tensor):
    data = np.cos(t.data)
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: -x * np.sin(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def tan(t: Tensor):
    data = np.tan(t.data)
    requires_grad = t.requires_grad
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x / np.cos(t.data)**2))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)