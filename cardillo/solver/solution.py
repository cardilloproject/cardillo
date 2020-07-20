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
    """
    def __init__(self, t=None, q=None, u=None, la_g=None, la_gamma=None, la_N=None):
        self.t = t
        self.q = q
        self.u = u
        self.la_g = la_g
        self.la_gamma = la_gamma
        self.la_N = la_N

    def unpack(self):
        """Return solution fields: t, q, u, la_g, la_gamma, la_N
        """
        return self.t, self.q, self.u, self.la_g, self.la_gamma, self.la_N

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