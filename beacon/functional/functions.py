from beacon.tensor import Tensor
from beacon.tensor import functions as fn
import numpy as np

########################
### Activation functions
########################

def sigmoid(t: Tensor):
    """
    Sigmoid activation function.

    ## Parameters
    t: `Tensor` - tensor on which to apply activation function

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    t = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    a = F.sigmoid(t)
    ```
    """
    return 1/ (1 + fn.exp(-t))

def softplus(t: Tensor):
    """
    Softplus activation function.

    ## Parameters
    t: `Tensor` - tensor on which to apply activation function

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    t = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    a = F.softplus(t)
    ```
    """
    return fn.log(1 + fn.exp(t))

def softsign(t: Tensor):
    """
    Softsign activation function.

    ## Parameters
    t: `Tensor` - tensor on which to apply activation function

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    t = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    a = F.softign(t)
    ```
    """
    return t / (1 + fn.abs(t))

def hard_sigmoid(t: Tensor):
    """
    Hard sigmoid activation function.

    ## Parameters
    t: `Tensor` - tensor on which to apply activation function

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    t = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    a = F.hard_sigmoid(t)
    ```
    """
    value = 0.2 * t + 0.5
    return fn.clip(value, 0.0, 1.0)

def relu(t: Tensor, alpha=0.0):
    """
    ReLU activation function.

    ## Parameters
    t: `Tensor` - tensor on which to apply activation function
    
    alpha: `float` - scale factor, defaults to 0.0

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    t = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    a = F.relu(t)
    ```
    """
    return t * (t >= 0.0) + alpha * t * (t < 0.0)

def leaky_relu(t: Tensor, alpha = 0.01):
    """
    Leaky ReLU activation function.

    ## Parameters
    t: `Tensor` - tensor on which to apply activation function

    alpha: `float` - scale factor, defaults to 0.01

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    t = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    a = F.leaky_relu(t)
    ```
    """
    return relu(t, alpha=alpha)

def softmax(t: Tensor):
    """
    Softmax activation function.

    ## Parameters
    t: `Tensor` - tensor on which to apply activation function

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    t = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    a = F.softmax(t)
    ```
    """
    y = fn.exp(t - fn.max(t, axis=-1, keepdims=True))
    return y / fn.sum(y, axis=-1, keepdims=True)

def elu(t: Tensor, alpha=1.0):
    """
    ELU activation function.

    ## Parameters
    t: `Tensor` - tensor on which to apply activation function

    alpha: `float` - scale factor, defaults to 1.0

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    t = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    a = F.elu(t)
    ```
    """
    return t * (t >= 0.0) + alpha * (fn.exp(t) - 1.0) * (t < 0.0)

def tanh(t: Tensor):
    """
    Tanh activation function.

    ## Parameters
    t: `Tensor` - tensor on which to apply activation function

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    t = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    a = F.tanh(t)
    ```
    """
    return fn.tanh(t)

##############
### Reductions
##############

def sum_over_batch_size(loss_function):
    """
    Decorator which reduces loss function over batch size.

    ## Parameters
    loss_function: `callable` - loss function

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.tensor import functions as fn
    from beacon.functional import functions as F

    @F.sum_over_batch_size
    def my_loss(output: Tensor, target: Tensor):
        return fn.sin(output - target)
    ```
    """
    def wrapper(*args, **kwargs):
        batch_loss = fn.sum(loss_function(*args, **kwargs))
        batch_loss.data /= np.size(args[0].data)
        return batch_loss
    return wrapper

########################
### Loss functions
########################

@sum_over_batch_size
def mean_squared_error(output: Tensor, target: Tensor):
    """
    Mean squared error loss function.

    ## Parameters
    output: `Tensor` - model's prediction

    target: `Target` - training sample targets

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    output = Tensor([[0.2, 0.7, 0.1], [0.4, 0.45, 0.15]], requires_grad=True)
    target = Tensor([[0, 1, 0], [1, 0, 0]], requires_grad=True)
    loss = F.mean_squared_error(output, target)
    ```
    """
    output, target = fn.to_tensor(output), fn.to_tensor(target)
    return fn.mean(fn.square(output - target), axis=-1)

@sum_over_batch_size
def mean_absolute_error(output: Tensor, target: Tensor):
    """
    Mean absolute error loss function.

    ## Parameters
    output: `Tensor` - model's prediction

    target: `Target` - training sample targets

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    output = Tensor([[0.2, 0.7, 0.1], [0.4, 0.45, 0.15]], requires_grad=True)
    target = Tensor([[0, 1, 0], [1, 0, 0]], requires_grad=True)
    loss = F.mean_absolute_error(output, target)
    ```
    """
    output, target = fn.to_tensor(output), fn.to_tensor(target)
    return fn.mean(fn.abs(output - target), axis=-1)

@sum_over_batch_size
def categorical_crossentropy(output: Tensor, target: Tensor):
    """
    Cross entropy loss function.

    ## Parameters
    output: `Tensor` - model's prediction

    target: `Target` - training sample targets

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    output = Tensor([[0.2, 0.7, 0.1], [0.4, 0.45, 0.15]], requires_grad=True)
    target = Tensor([[0, 1, 0], [1, 0, 0]], requires_grad=True)
    loss = F.categorical_crossentropy(output, target)
    ```
    """
    output, target = fn.to_tensor(output), fn.to_tensor(target)
    output = fn.clip(output, 1e-7, 1 - 1e-7)
    return target * -fn.log(output)

@sum_over_batch_size
def binary_crossentropy(output: Tensor, target: Tensor):
    """
    Binary cross entropy loss function.

    ## Parameters
    output: `Tensor` - model's prediction

    target: `Target` - training sample targets

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    output = Tensor([[0.89], [0.76], [0.1]], requires_grad=True)
    target = Tensor([[1], [1], [0]], requires_grad=True)
    loss = F.binary_crossentropy(output, target)
    ```
    """
    output, target = fn.to_tensor(output), fn.to_tensor(target)
    output = fn.clip(output, 1e-7, 1 - 1e-7)
    return (target * -fn.log(sigmoid(output)) + (1 - target) * -fn.log(1 - sigmoid(output)))

@sum_over_batch_size
def nll_loss(output: Tensor, target: Tensor):
    """
    Negative log likelihood loss function.

    ## Parameters
    output: `Tensor` - model's prediction

    target: `Target` - training sample targets

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    output = Tensor([[0.2, 0.7, 0.1], [0.4, 0.45, 0.15]], requires_grad=True)
    target = Tensor([[0, 1, 0], [1, 0, 0]], requires_grad=True)
    loss = F.nll_loss(output, target)
    ```
    """
    output, target = fn.to_tensor(output), fn.to_tensor(target)
    output = fn.clip(output, 1e-7, 1 - 1e-7)
    return -target * fn.log(output)

@sum_over_batch_size
def quadratic(output: Tensor, target: Tensor):
    """
    Quadratic loss function.

    ## Parameters
    output: `Tensor` - model's prediction

    target: `Target` - training sample targets

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    output = Tensor([[0.2, 0.7, 0.1], [0.4, 0.45, 0.15]], requires_grad=True)
    target = Tensor([[0, 1, 0], [1, 0, 0]], requires_grad=True)
    loss = F.quadratic(output, target)
    ```
    """
    output, target = fn.to_tensor(output), fn.to_tensor(target)
    return fn.square(output - target)

@sum_over_batch_size
def half_quadratic(output: Tensor, target: Tensor):
    """
    Half quadratic loss function.

    ## Parameters
    output: `Tensor` - model's prediction

    target: `Target` - training sample targets

    ## Example usage
    ```python
    from beacon.tensor import Tensor
    from beacon.functional import functions as F

    output = Tensor([[0.2, 0.7, 0.1], [0.4, 0.45, 0.15]], requires_grad=True)
    target = Tensor([[0, 1, 0], [1, 0, 0]], requires_grad=True)
    loss = F.half_quadratic(output, target)
    ```
    """
    output, target = fn.to_tensor(output), fn.to_tensor(target)
    return 0.5 * fn.square(output - target)

########################
### Pooling functions
########################

def max_pool(t: Tensor, kernel_size=2):
    """
    Applies max pool filter to input tensor.

    ## Parameters
    t: `Tensor` - input tensor

    kernel_size: `int` - size of max pool filter, defaults to 2

    ## Example usage
    from beacon.nn import Tensor
    from beacon.functions import functional as F
    import numpy as np
    t = Tensor(data=np.random.normal(size=(4, 4, 2)), requires_grad=True)
    x = F.max_pool(t, kernel_size=2)
    """
    return fn.max_pool(t, kernel_size)

def average_pool(t: Tensor, kernel_size=2):
    """
    Applies average pool filter to input tensor.

    ## Parameters
    t: `Tensor` - input tensor

    kernel_size: `int` - size of average pool filter, defaults to 2

    ## Example usage
    from beacon.nn import Tensor
    from beacon.functions import functional as F
    import numpy as np
    t = Tensor(data=np.random.normal(size=(4, 4, 2)), requires_grad=True)
    x = F.average_pool(t, kernel_size=2)
    """
    # return fn.average_pool(t, kernel_size)
    raise NotImplementedError("Average pooling derivative not yet implemented.")