from collections import namedtuple
import dill


def save_solution(sol, filename):
    """Store a `Solution` object into a given file."""
    with open(filename, mode="wb") as f:
        dill.dump(sol, f)


def load_solution(filename):
    """Load a `Solution` object from a given file."""
    with open(filename, mode="rb") as f:
        return dill.load(f)


class Solution:
    """Class to store solver outputs."""

    def __init__(
        self,
        system,
        t,
        q,
        u=None,
        u_dot=None,
        la_g=None,
        la_gamma=None,
        la_c=None,
        la_N=None,
        la_F=None,
        **kwargs,
    ):
        self.system = system
        self.t = t
        self.q = q
        self.u = u
        self.u_dot = u_dot
        self.la_g = la_g
        self.la_gamma = la_gamma
        self.la_c = la_c
        self.la_N = la_N
        self.la_F = la_F
        self.solver_summary = None

        self.__dict__.update(**kwargs)

    def save(self, filename):
        save_solution(self, filename)

    def __iter__(self):
        return self.SolutionIterator(self)

    class SolutionIterator:
        def __init__(self, solution) -> None:
            self._solution = solution
            self.keys = [*self._solution.__dict__.keys()]
            # remove non-iterable keys
            self.keys.remove("solver_summary")
            self.keys.remove("system")

            self._index = 0
            self._retVal = namedtuple("Result", self.keys)

        def __next__(self):
            if self._index < len(self._solution.t):
                try:
                    result = self._retVal(
                        *(
                            (
                                None
                                if self._solution.__getattribute__(key) is None
                                else (
                                    self._solution.__getattribute__(key)[:, self._index]
                                    if self._solution.__getattribute__(key).shape[0]
                                    == 0
                                    else self._solution.__getattribute__(key)[
                                        self._index
                                    ]
                                )
                            )
                            for key in self.keys
                        )
                    )
                except:
                    RuntimeWarning("Solution iterator failed.")
                self._index += 1
                return result
            raise StopIteration
