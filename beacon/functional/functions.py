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
    return fn.sum(((output-target)**2) / output.data.shape[1])

def mean_absolute_error(output: Tensor, target: Tensor):
    return fn.sum(((output - target)) / output.data.shape[1])

def cross_entropy(output: Tensor, target: Tensor):
    pass

def nll_loss(output: Tensor, target: Tensor):
    pass

def quadratic(output: Tensor, target: Tensor):
    pass

def half_quadratic(output: Tensor, target: Tensor):
    pass