from beacon.tensor import Tensor
import numpy as np

############################
### Differentiable operators
############################

def to_tensor(x):
    """
    Convert input parameter to tensor if it isn't already.

    ## Parameters
    x: `Tensor-like` - input parameter

    ## Example usage
    ```python
    from beacon.tensor import functinos as fn
    t = fn.to_tensor(10.0)
    ```
    """
    return Tensor._to_tensor(x)

def zeros_like(t: Tensor):
    """
    Return zero-tensor with the same shape as the input tensor.

    ## Parameters
    t: `Tensor` - input parameter

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functinos as fn
    t = Tensor([1, 2, 3])
    x = fn.zeros_like(t)
    ```
    """
    return to_tensor(np.zeros_like(t.data))

def add(t1: Tensor, t2: Tensor):
    """
    Adds two tensors.

    ## Parameters:
    t1: `Tensor` - first tensor

    t2: `Tensor` - second tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    x = fn.add(t1, t2)
    ```
    """
    data = t1.data + t2.data
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: _broadcast(t1.grad.data, x)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: _broadcast(t2.grad.data, x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def mul(t1: Tensor, t2: Tensor):
    """
    Multiplites two tensors (dot-product).

    ## Parameters:
    t1: `Tensor` - first tensor

    t2: `Tensor` - second tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    x = fn.mul(t1, t2)
    ```
    """
    data = t1.data * t2.data
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: _broadcast(t1.grad.data, t2.data*x)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: _broadcast(t2.grad.data, t1.data*x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sub(t1: Tensor, t2: Tensor):
    """
    Substracts two tensors.

    ## Parameters:
    t1: `Tensor` - first tensor

    t2: `Tensor` - second tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    x = fn.sub(t1, t2)
    ```
    """
    data = t1.data - t2.data
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: _broadcast(t1.grad.data, x)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: _broadcast(t2.grad.data, -x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def divide(t1: Tensor, t2: Tensor):
    """
    Divides two tensors.

    ## Parameters:
    t1: `Tensor` - first tensor

    t2: `Tensor` - second tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    x = fn.divide(t1, t2)
    ```
    """
    data = t1.data / t2.data
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: _broadcast(t1.grad.data, x /t2.data)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: _broadcast(t2.grad.data, -x * t1.data/ t2.data**2 )))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sum(t: Tensor, axis=None, keepdims=False):
    """
    Sums all the elements in tensor along given axis

    ## Parameters:
    t: `Tensor`

    axis: `int` - defaults to None

    keepdims: `bool` - defaults to False

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = fn.sum(t)
    ```
    """
    data = np.sum(t.data, axis=axis, keepdims=keepdims)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: _match_shape(x, t.data.shape, axis, keepdims)[0]))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def matmul(t1: Tensor, t2: Tensor):
    """
    Matrix multiplications of two tensors.

    ## Parameters:
    t1: `Tensor` - first tensor

    t2: `Tensor` - second tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    x = fn.matmul(t1, t2)
    ```
    """
    data = np.matmul(t1.data, t2.data)
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: np.matmul(x, t2.data.T)))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: np.matmul(t1.data.T, x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def neg(t: Tensor):
    """
    Unary negation of tensor elements.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = -t
    ```
    """
    data = -t.data
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: -x))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def exp(t: Tensor):
    """
    Applies exp function to all the elements of the input tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = fn.exp(t)
    ```
    """
    data = np.exp(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: data*x))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def tanh(t: Tensor):
    """
    Applies tanh function to all the elements of the input tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = fn.tanh(t)
    ```
    """
    data = np.tanh(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x / np.cosh(t.data)**2))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sinh(t: Tensor):
    """
    Applies sinh function to all the elements of the input tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = fn.sinh(t)
    ```
    """
    data = np.sinh(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x*np.cosh(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def cosh(t: Tensor):
    """
    Applies cosh function to all the elements of the input tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = fn.cosh(t)
    ```
    """
    data = np.tanh(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x*np.sinh(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def maximum(t1: Tensor, t2: Tensor):
    """
    Element-wise maximum of two tensor.

    ## Parameters:
    t1: `Tensor` - first tensor

    t2: `Tensor` - second tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t1 = Tensor([1, 4, 3])
    t2 = Tensor([2, 5, 6])
    x = fn.maximum(t1, t2)
    ```
    """
    data = np.maximum(t1.data, t2.data)
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    def max_grad(x, z, y):
        return (x == z) / (1.0 + (x == y))
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: _broadcast(t1.data, x * max_grad(t1.data, data, t2.data))))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: _broadcast(t2.data, x * max_grad(t2.data, data, t1.data))))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def minimum(t1: Tensor, t2: Tensor):
    """
    Element-wise minimum of two tensor.

    ## Parameters:
    t1: `Tensor` - first tensor

    t2: `Tensor` - second tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t1 = Tensor([1, 4, 3])
    t2 = Tensor([2, 5, 6])
    x = fn.minimum(t1, t2)
    ```
    """
    data = np.minimum(t1.data, t2.data)
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    def min_grad(x, z, y):
        return (x == z) / (1.0 + (x == y))
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, df=lambda x: _broadcast(t1.data, x * min_grad(t1.data, data, t2.data))))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, df=lambda x: _broadcast(t2.data, x * min_grad(t2.data, data, t1.data))))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def clip(t: Tensor, min_val, max_val):
    """
    Clips input tensor to minimum and maximum value.

    ## Parameters:
    t: `Tensor` - input tensor

    min_val: `float` - minimum value

    max_val: `float` - maximum value

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([2, 4, 6])
    x = fn.clip(t, 3, 8)
    ```
    """
    data = np.clip(t.data, min_val, max_val)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x * np.logical_and(data != min_val, data != max_val)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def power(t1: Tensor, t2: Tensor):
    """
    Raises first tensor to the power of second tensor.

    ## Parameters:
    t1: `Tensor` - first tensor

    t2: `Tensor` - second tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t1 = Tensor([1, 4, 3])
    t2 = Tensor([2, 5, 6])
    x = fn.power(t1, t2)
    ```
    """
    data = np.power(t1.data, t2.data)
    requires_grad = (t1.requires_grad or t2.requires_grad) and not Tensor.NO_GRAD
    nodes = []
    if t1.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t1, 
        df=lambda x: _broadcast(t1.data, x*t2.data*(t1.data**np.where(t2.data, t2.data-1, 1.)))))
    if t2.requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t2, 
        df=lambda x: _broadcast(t2.data, x * np.log((np.where(t1.data, t1.data, 1.)) * data))))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def abs(t: Tensor):
    """
    Applies abs function to all the elements of the input tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, -2, 3])
    x = fn.abs(t)
    ```
    """
    data = np.abs(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: np.sign(t.data)*x))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def log(t: Tensor):
    """
    Applies log (ln) function to all the elements of the input tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = fn.log(t)
    ```
    """
    data = np.log(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x / t.data))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sqrt(t: Tensor):
    """
    Applies sqrt function to all the elements of the input tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = fn.sqrt(t)
    ```
    """
    data = np.sqrt(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: 0.5*x*(t.data**-0.5)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def square(t: Tensor):
    """
    Applies square function to all the elements of the input tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = fn.square(t)
    ```
    """
    data = np.square(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: 2*x*t.data))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def reshape(t: Tensor, shape):
    """
    Reshapes tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    shape: `tuple` - new shape

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3], [4, 5, 6])
    x = fn.reshape(t, shape=(1, 6))
    ```
    """
    data = np.reshape(t.data, newshape=shape)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: np.reshape(x, np.shape(t.data))))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def sin(t: Tensor):
    """
    Applies sin function to all the elements of the input tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = fn.sin(t)
    ```
    """
    data = np.sin(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x * np.cos(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def cos(t: Tensor):
    """
    Applies cos function to all the elements of the input tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = fn.cos(t)
    ```
    """
    data = np.cos(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: -x * np.sin(t.data)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def tan(t: Tensor):
    """
    Applies tan function to all the elements of the input tensor.

    ## Parameters:
    t: `Tensor` - input tensor

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([1, 2, 3])
    x = fn.tan(t)
    ```
    """
    data = np.tan(t.data)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: x / np.cos(t.data)**2))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def mean(t: Tensor, axis=None, keepdims=False):
    """
    Numpy mean function equivalent.

    ## Parameters:
    t: `Tensor` - input tensor

    axis: `int` - defaults to None

    keepdims: `bool` - defaults to False

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    x = fn.mean(t, axis=1)
    ```
    """
    data = np.mean(t.data, axis=axis, dtype=np.float, keepdims=keepdims)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        def mean_grad(x):
            g, n = _match_shape(x, np.shape(t.data), axis, keepdims)
            return g / n
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: mean_grad(x)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def where(condition: Tensor, t1: Tensor=None, t2: Tensor=None):
    """
    Numpy where function equivalent.

    ## Parameters:
    condition: `Tensor` - condition tensor

    t1: `Tensor` - tensor from which to take elements if condition is met, defaults to None
    
    t2: `Tensor` - tensor from which to take elements if condition is not met, defaults to None

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    x = fn.where(t1 < 2, t1, t2)
    ```
    """
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

def min(t: Tensor, axis=None, keepdims=False):
    """
    Returns min elements of the input tensor alongside given axis.

    ## Parameters
    t: `Tensor` - input tensor

    axis: `int` - defaults to None

    keepdims: `bool` - defaults to None

    ## Example usage

    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    x = fn.min(t, axis=1, keepdims=True)
    ```
    """
    data = np.min(t.data, axis=axis, keepdims=keepdims)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: _min_max_grad(x, data, t.data, axis, keepdims)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def max(t: Tensor, axis=None, keepdims=False):
    """
    Returns max elements of the input tensor alongside given axis.

    ## Parameters
    t: `Tensor` - input tensor

    axis: `int` - defaults to None

    keepdims: `bool` - defaults to None

    ## Example usage

    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    x = fn.max(t, axis=1, keepdims=True)
    ```
    """
    data = np.max(t.data, axis=axis, keepdims=keepdims)
    requires_grad = t.requires_grad and not Tensor.NO_GRAD
    nodes = []
    if requires_grad:
        nodes.append(Tensor.ComputationalGraphNode(tensor=t, df=lambda x: _min_max_grad(x, data, t.data, axis, keepdims)))
    return Tensor(data=data, requires_grad=requires_grad, nodes=nodes)

def _broadcast(target_grad, input_grad):
    """
    Helper function. Unbroadcasts gradient if input tensor didn't have the same dimensions.
    """
    while np.ndim(input_grad) > np.ndim(target_grad):
        input_grad = np.sum(input_grad, axis=0)
    for axis, dim in enumerate(np.shape(target_grad)):
        if dim == 1:
            input_grad = np.sum(input_grad, axis=axis, keepdims=True)
    return input_grad

def _min_max_grad(x, result, inputs, axis, keepdims):
    """
    Helper function for min and max functions.
    """
    reps, _ = _match_shape(x, inputs.shape, axis, keepdims)
    argmax = x == _match_shape(result, inputs.shape, axis, keepdims)[0]
    return reps * argmax / np.sum(argmax, axis=axis, keepdims=True)

def _match_shape(x, shape, axis, keepdims):
    """
    Helper function. Matches shape of tensors and returns the number of repetitions when calculating gradient.
    """
    if shape == ():
        return x, 1
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = np.array(shape)
    new_shape[axis] = 1
    num_reps = np.prod(np.array(shape)[axis])
    return np.reshape(x, new_shape) + np.zeros(shape, dtype=np.float), num_reps