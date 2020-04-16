from beacon.tensor import Tensor
from beacon.tensor import functions as fn

########################
### Activation functions
########################

def sigmoid(t: Tensor):
    return 1/ (1 + fn.exp(-t))

def relu(t: Tensor):
    pass

def leaky_relu(t: Tensor):
    pass

def softmax(t: Tensor):
    pass

def elu(t: Tensor):
    pass

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