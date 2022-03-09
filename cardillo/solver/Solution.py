import pickle


def save_solution(sol, filename):
    """Store a `Solution` object into a given file."""
    with open(filename, mode="wb") as f:
        pickle.dump(sol, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_solution(filename):
    """Load a `Solution` object from a given file."""
    with open(filename, mode="rb") as f:
        return pickle.load(f)


class Solution:
    """Class to store solver outputs."""

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def save(self, filename):
        save_solution(self, filename)
