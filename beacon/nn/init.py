import numpy as np

def normal(shape: tuple):
    return np.random.normal(size=shape)

def uniform(shape: tuple):
    return np.random.uniform(size=shape)

def xavier_normal(shape: tuple):
    factor = 2.0 * np.sqrt(6.0 / (shape[0]*shape[1]))
    return factor * normal(shape)

def xavier_uniform(shape: tuple):
    factor = 2.0 * np.sqrt(6.0 / (shape[0]*shape[1]))
    return factor * uniform(shape)

def lecun_normal(shape: tuple):
    factor = 1.0 / np.sqrt(shape[1])
    return factor * normal(shape)

def lecun_uniform(shape: tuple):
    factor = 2.0 * np.sqrt(3.0 / shape[1])
    return factor * uniform(shape)

def he_normal(shape: tuple):
    factor = np.sqrt(2.0 / shape[1])
    return factor * normal(shape)

def he_uniform(shape: tuple):
    factor = 2.0 * np.sqrt(6.0 / shape[1])
    return factor * uniform(shape)

def zeros(shape: tuple):
    return np.zeros(shape)