from collections import namedtuple
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
    
    def __iter__(self):
        return self.SolutionIterator(self)

    class SolutionIterator:
        def __init__(self, solution) -> None:
            self._solution = solution
            self._index = 0
            self._retVal = namedtuple('Result', [*self._solution.__dict__.keys()])
        
        def __next__(self):
            if self._index < len(self._solution.t):
                result = self._retVal(*( \
                    self._solution.__getattribute__(key)[self._index] \
                    if self._solution.__getattribute__(key) is not None \
                    else None  \
                    for key in self._solution.__dict__))
                # result = self._retVal(*(eval(f'self._solution.{key}[{self._index}]') \
                    # for key in self._solution.__dict__))  
                self._index += 1
                return result
            raise StopIteration
