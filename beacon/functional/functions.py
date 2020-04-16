from beacon.tensor import Tensor
from beacon.tensor import functions as fn

########################
### Activation functions
########################

def sigmoid(t: Tensor):
    return 1/ (1 + fn.exp(-t))

def softplus(t: Tensor):
    return fn.log(1 + fn.exp(t))

def softsign(t: Tensor):
    return t / (1 + fn.abs(t))

def hard_sigmoid(t: Tensor):
    value = 0.2 * t + 0.5
    return fn.clip(value, 0.0, 1.0)

def relu(t: Tensor, alpha=0.0):
    return t * (t >= 0.0) + alpha * t * (t < 0.0)

def leaky_relu(t: Tensor, alpha = 0.01):
    return relu(t, alpha=alpha)

def softmax(t: Tensor):
    y = fn.exp(t)
    return y / fn.sum(y, axis=-1, keepdims=True)

def elu(t: Tensor, alpha=1.0):
    return t * (t >= 0.0) + alpha * (fn.exp(t) - 1.0) * (t < 0.0)

def tanh(t: Tensor):
    return fn.tanh(t)

########################
### Loss functions
########################

def mean_squared_error(output: Tensor, target: Tensor):
    return fn.sum(fn.mean(fn.square(output - target), axis=-1))

def mean_absolute_error(output: Tensor, target: Tensor):
    return fn.sum(fn.mean(fn.abs(output - target), axis=-1))

def categorical_crossentropy(output: Tensor, target: Tensor):
    output = fn.clip(output, 1e-7, 1 - 1e-7)
    return fn.sum(target * -fn.log(output), axis=-1, keepdims=False)

def binary_crossentropy(output: Tensor, target: Tensor):
    output = fn.clip(output, 1e-7, 1 - 1e-7)
    return (target * -fn.log(sigmoid(output)) + (1 - target) * -fn.log(1 - sigmoid(output)))

def nll_loss(output: Tensor, target: Tensor):
    output = fn.clip(output, 1e-7, 1 - 1e-7)
    return -fn.sum(target * fn.log(output))

def quadratic(output: Tensor, target: Tensor):
    return fn.sum(fn.square(output - target))

def half_quadratic(output: Tensor, target: Tensor):
    return 0.5 * fn.sum(fn.square(output - target))