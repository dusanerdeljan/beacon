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