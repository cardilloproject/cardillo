import numpy as np


def prox_R0_nm(x):
    return np.minimum(x, 0)


def prox_R0_np(x):
    return np.maximum(x, 0)


def prox_sphere(x, radius):
    nx = np.linalg.norm(x)
    if nx > 0:
        return x if nx <= radius else radius * x / nx
    else:
        return x if nx <= radius else radius * x
