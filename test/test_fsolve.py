import numpy as np
from cardillo.math.fsolve import fsolve
from scipy.sparse import csc_array
from scipy.optimize import rosen_der, rosen_hess


def test_fsolve():
    fun = rosen_der
    jac = lambda x: csc_array(rosen_hess(x))
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    sol = fsolve(fun, x0, jac)
    print(f"sol:\n{sol}")

    assert np.allclose(sol.x, np.ones_like(x0))
    assert sol.success
    assert sol.error < 1
    assert np.allclose(sol.fun, np.zeros_like(x0))


if __name__ == "__main__":
    test_fsolve()
