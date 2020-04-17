# beacon
PyTorch-like deep learning library with dynamic computational graph construction and automatic differentiation. Automatic differentiation is based on `Autograd` written by Dougal Maclaurin, David Duvenaud and Matt Johnson.

## Example usage

### Defining a custom model

In order to create your own model you have to inherit from Module class and override `forward` method. Later on, you can train your model or use it as part of a bigger network.

```python
from beacon.nn import beacon
from beacon.nn import Module, Linear
from beacon.functional import functions as F

class MyModel(Module):
  def __init__(self):
        super().__init__()
        self.fc1 = Linear(inputs=2, outputs=4)
        self.fc2 = Linear(inputs=4, outputs=4)
        self.fc3 = Linear(inputs=4, outputs=1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

model = MyModel()
```

#### Customizing parameter initializer

When creating a model it is possible to customize the way that parameters are initialized. For example:

```python
from beacon.nn.init import xavier_normal, zeros
self.fc2 = Linear(inputs=4, outputs=4, weight_initializer=xavier_normal, bias_initializer=zeros)
```

### Training a model

#### Getting the data

```python
from beacon.data import data_sequence, data_generator

x_train = [[1, 0], [0, 0], [0, 1], [0, 0]]
y_train = [[1], [0], [1], [0]]
```

There are two possible ways of preparing data:
 * Using sequences
```python
X, Y = data_sequence(x_train, y_train, batch_size=4, shuffle=True)
```
 * Using generator
```python
for x, y in data_generator(x_train, y_train, batch_size=4, shuffle=True):
  do_some_task(x, y)
```

#### Selecting an optimizer

```python
from beacon.optim import Adam
optimizer = Adam(model.parameters(), lr=0.1)
```

#### Example training loop

```python
for epoch in range(1, 1001):
    full_loss = 0
    n_loss = 0

    for x, y in data_generator(x_train, y_train, batch_size=4, shuffle=True):
        optimizer.zero_grad()
        output = model(x)
        loss = F.mean_squared_error(output, y)
        loss.backward()
        optimizer.step(epoch)

        full_loss += loss.item()
        n_loss += 1

    print(f"Epoch: {epoch}, Loss={full_loss/n_loss}")
```

#### Evaluating a model

```python
with beacon.no_grad():
    for x in x_train:
        output = model(x)
        print(output)
```

## Extending beacon

Beacon can be easily extended by defining your own custom activation functions, loss functions or optimizers.

### Defining custom activation functions

When defining your activation function you have to make sure that all operations are differentiable. Your activation function has to take tensor as a parameter and return a tensor. When you define your activation function it can be used just as predefined activations.

```python
from beacon.tensor import functions as fn
from beacon.tensor import Tensor

def my_activation(t: Tensor) -> Tensor:
  return fn.sin(t)
```

### Defining custom loss functions

When defining your loss function you have to make sure that all operations are differentiable. Your loss function has to take two tensors as parameters and return 0-dimensional tensor. When you define your loss function it can be used just as predefined losses.

```python
from beacon.tensor import functions as fn
from beacon.tensor import Tensor

def my_loss(output: Tensor, target: Tensor) -> Tensor:
  return fn.sum(fn.sin(fn.square(output - target)))
```

### Defining custom optimizers

When defining custom optimizer you have to inherit from Optimizer base class and redefine `_step` method. When you define your optimizer it can be used just as predefined optimizers.

```python
from beacon.optim.optimizer import Optimizer
from beacon.tensor import functions as fn

class MyOptimizer(Optimizer):

    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr=lr)

    def _step(self, epoch):
        for parameter in self.parameters:
            parameter -= fn.square(self.lr*parameter.grad)
```
