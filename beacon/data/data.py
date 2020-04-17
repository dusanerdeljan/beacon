import numpy as np

def data_sequence(inputs, labels, batch_size=32, shuffle=True):
    """
    Prepares data for training/evaluation and returns it in the form of numpy array.

    ## Parameters:
    inputs: `np.array or python list` - inputs to the neural network

    labels: `np.array or python list` - targets
    
    batch_size: `int` - defaults to 32

    shuffle: `bool` - defaults to True

    Inputs and labels must have the same first dimenstion!

    Shape (M, N) means:

     * There are M Training sample
     * Each training sample has N inputs
    """
    X, Y = [], []
    indices = np.arange(0, np.asarray(inputs).shape[0], batch_size)
    if shuffle:
        np.random.shuffle(indices)
    for batch_begin in indices:
        X.append(inputs[batch_begin:batch_begin+batch_size])
        Y.append(labels[batch_begin:batch_begin+batch_size])
    return X, Y

def data_generator(inputs, labels, batch_size=32, shuffle=True):
    """
    Yields batch inputs and batch targets.

    ## Parameters:
    inputs: `np.array or python list` - inputs to the neural network

    labels: `np.array or python list` - targets

    batch_size: `int` - defaults to 32

    shuffle: `bool` - defaults to True

    Inputs and labels must have the same first dimenstion!

    Shape (M, N) means:

     * There are M Training sample
     * Each training sample has N inputs
    """
    indices = np.arange(0, np.asarray(inputs).shape[0], batch_size)
    if shuffle:
        np.random.shuffle(indices)
    for batch_begin in indices:
        yield np.asarray(inputs[batch_begin:batch_begin+batch_size], dtype=np.float), np.asarray(labels[batch_begin:batch_begin+batch_size], dtype=np.float)