from beacon.tensor import Tensor
from beacon.tensor import functions as fn

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

########################
### Loss functions
########################

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
    return fn.sum(fn.mean(fn.square(output - target), axis=-1))

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
    return fn.sum(fn.mean(fn.abs(output - target), axis=-1))

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
    return fn.sum(target * -fn.log(output))

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
    return -fn.sum(target * fn.log(output))

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
    return fn.sum(fn.square(output - target))

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
    return 0.5 * fn.sum(fn.square(output - target))