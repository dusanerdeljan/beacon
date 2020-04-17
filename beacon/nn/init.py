import numpy as np

def normal(shape: tuple):
    """
    Normal initializer.

    ## Parameters
    shape: `tuple` - shape of inputs tensor

    ## Example usage
    ```python
    from beacon.nn.init import normal
    t = Tensor(data=normal(shape=(5, 3)))
    ```
    """
    return np.random.normal(size=shape)

def uniform(shape: tuple):
    """
    Uniform initializer.

    ## Parameters
    shape: `tuple` - shape of inputs tensor

    ## Example usage
    ```python
    from beacon.nn.init import uniform
    t = Tensor(data=uniform(shape=(5, 3)))
    ```
    """
    return np.random.uniform(size=shape)

def xavier_normal(shape: tuple):
    """
    Xavier normal initializer.

    ## Parameters
    shape: `tuple` - shape of inputs tensor

    ## Example usage
    ```python
    from beacon.nn.init import xavier_normal
    t = Tensor(data=xavier_normal(shape=(5, 3)))
    ```
    """
    factor = 2.0 * np.sqrt(6.0 / (shape[0]*shape[1]))
    return factor * normal(shape)

def xavier_uniform(shape: tuple):
    """
    Xavier uniform initializer.

    ## Parameters
    shape: `tuple` - shape of inputs tensor

    ## Example usage
    ```python
    from beacon.nn.init import xavier_uniform
    t = Tensor(data=xavier_uniform(shape=(5, 3)))
    ```
    """
    factor = 2.0 * np.sqrt(6.0 / (shape[0]*shape[1]))
    return factor * uniform(shape)

def lecun_normal(shape: tuple):
    """
    LeCun normal initializer.

    ## Parameters
    shape: `tuple` - shape of inputs tensor

    ## Example usage
    ```python
    from beacon.nn.init import lecun_normal
    t = Tensor(data=lecun_normal(shape=(5, 3)))
    ```
    """
    factor = 1.0 / np.sqrt(shape[1])
    return factor * normal(shape)

def lecun_uniform(shape: tuple):
    """
    LeCun uniform initializer.

    ## Parameters
    shape: `tuple` - shape of inputs tensor

    ## Example usage
    ```python
    from beacon.nn.init import lecun_uniform
    t = Tensor(data=lecun_uniform(shape=(5, 3)))
    ```
    """
    factor = 2.0 * np.sqrt(3.0 / shape[1])
    return factor * uniform(shape)

def he_normal(shape: tuple):
    """
    He normal initializer.

    ## Parameters
    shape: `tuple` - shape of inputs tensor

    ## Example usage
    ```python
    from beacon.nn.init import he_normal
    t = Tensor(data=he_normal(shape=(5, 3)))
    ```
    """
    factor = np.sqrt(2.0 / shape[1])
    return factor * normal(shape)

def he_uniform(shape: tuple):
    """
    He uniform initializer.

    ## Parameters
    shape: `tuple` - shape of inputs tensor

    ## Example usage
    ```python
    from beacon.nn.init import he_uniform
    t = Tensor(data=he_uniform(shape=(5, 3)))
    ```
    """
    factor = 2.0 * np.sqrt(6.0 / shape[1])
    return factor * uniform(shape)

def zeros(shape: tuple):
    """
    Zeros initializer.

    ## Parameters
    shape: `tuple` - shape of inputs tensor

    ## Example usage
    ```python
    from beacon.nn.init import zeros
    t = Tensor(data=zeros(shape=(5, 3)))
    ```
    """
    return np.zeros(shape)