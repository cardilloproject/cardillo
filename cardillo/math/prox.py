import numpy as np

def prox_Rn0(x):
    return np.maximum(x, 0)