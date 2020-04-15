from beacon.tensor import Tensor
from beacon.tensor import functions as func

########################
### Activation functions
########################

def sigmoid(t: Tensor):
    return 1/ (1 + func.exp(-t))

def relu(t: Tensor):
    pass

def leaky_relu(t: Tensor):
    pass

def softmax(t: Tensor):
    pass

def elu(t: Tensor):
    pass

def tanh(t: Tensor):
    pass