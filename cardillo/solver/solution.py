import numpy as np
import pickle

class Solution():
    r"""Class to store and manage solver outputs.

    Parameters
    ----------
    t : numpy.ndarray
        time
    q : numpy.ndarray
        generalized coordinates
    u : numpy.ndarray 
        generalized velocities
    la_g : numpy.ndarray
         constraint forces of bilateral constraints on position level
    la_gamma : numpy.ndarray
         constraint forces of bilateral constraints on velocity level
    la_N : numpy.ndarray
         contact forces in normal direction
    la_T : numpy.ndarray
         contact forces in tangent direction
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.sol_keys = kwargs.keys()

    def unpack(self):
        """Return solution fields
        """
        return tuple([self.__dict__[key] for key in self.sol_keys])

def save_solution(sol, filename):
    """Store a `Solution` object into a given file.

    Parameters
    ----------
    sol: `Solution`
        Solution object
    filename: str
        Filename where the solution will be saved
    """
    with open(filename, mode='wb') as f:
        pickle.dump(sol, f)

def load_solution(filename):
    """Load a `Solution` object from a given file.

    Parameters
    ----------
    filename: str
        Filename where the solution was saved
    """
    with open(filename, mode='rb') as f:
        return pickle.load(f)