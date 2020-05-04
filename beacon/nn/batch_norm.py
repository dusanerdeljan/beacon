from beacon.nn.module import Module, Parameter
from beacon.tensor import Tensor
from beacon.tensor import functions as fn
from beacon.nn.init import normal

class BatchNorm(Module):

    def __init__(self, input_shape, momentum=0.1, beta_initializer=normal, gamma_initializer=normal, epsilon=1e-8):
        """
        Batch normalization module.

        ## Parameters
        input_shape: `tuple` - shape of input features

        momentum: `float` - momentum when calculatin exponentially weighted average, defaults to 0.1

        beta_initializer: `callable` - defaults to normal

        gamma_initializer: `callable` - defaults to normal

        epsilong: `float` - numerical stability constant, defaults to 1e-8
        """
        super().__init__()
        self.momentum = fn.to_tensor(momentum)
        self.beta = Parameter(shape=input_shape, initializer=beta_initializer)
        self.gamma = Parameter(shape=input_shape, initializer=gamma_initializer)
        self.u_avg = fn.to_tensor(0)
        self.std_avg = fn.to_tensor(0)
        self.epsilon = fn.to_tensor(epsilon)

    def forward(self, x):
        if self.train_mode:
            mean = fn.mean(x)
            standard_deviation = fn.mean(fn.square(x - mean))
            self.u_avg.data = self.momentum.data * self.u_avg.data + (1 - self.momentum.data) * mean.data
            self.std_avg.data = self.momentum.data * self.std_avg.data + (1 - self.momentum.data) * standard_deviation.data
        else:
            mean = self.u_avg
            standard_deviation = self.std_avg
        x = (x - mean) / fn.sqrt(standard_deviation + self.epsilon)
        return fn.mul(x, self.gamma) + self.beta