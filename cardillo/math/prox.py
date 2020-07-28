import numpy as np
from cardillo.math.algebra import norm2

def prox_Rn0(x):
    return np.maximum(x, 0)

def prox_circle(x, radius):
    nx = norm2(x)
    if nx > 0:
        return x if nx <= radius else radius * x / nx
    else:
        return x if nx <= radius else radius * x 