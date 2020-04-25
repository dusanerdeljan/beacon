from beacon.nn.module import Module

class Sequential(Module):

    def __init__(self, *layers):
        """
        Sequential model.

        ## Parameters
        layers: `tuple(Module)` - List of modules

        ## Example usage
        ```python
        from beacon.nn.modules import Sequential
        from beacon.nn import Linear
        from beacon.nn.activations import Sigmoid
        model = Sequential(
            Linear(2,4),
            Sigmoid(),
            Linear(4,4),
            Sigmoid(),
            Linear(4,1),
            Sigmoid()
        )

        layers = [Linear(2,4),Sigmoid(),Linear(4,4),Sigmoid(),Linear(4,1),Sigmoid()]
        model2 = Sequential(*layers)
        ```
        """
        self.layers = layers
        for layer in self.layers:
            if not isinstance(layer, Module):
                raise RuntimeError("Invalid arguments to sequential model.")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x