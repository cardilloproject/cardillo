import numpy as np


def prox_Rn0(x):
    return np.maximum(x, 0)


def prox_sphere(x, radius):
    nx = np.linalg.norm(x)
    if nx > 0:
        return x if nx <= radius else radius * x / nx
    else:
        return x if nx <= radius else radius * x
