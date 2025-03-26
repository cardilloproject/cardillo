import numpy as np
from scipy.sparse.linalg._isolve.iterative import _get_atol_rtol
from scipy.sparse.linalg._isolve.utils import make_system


# inspired by scipy's conjugate gradient method,
# see https://github.com/scipy/scipy/blob/v1.15.2/scipy/sparse/linalg/_isolve/iterative.py#L0-L1
def cr(A, b, x0=None, *, rtol=1e-5, atol=0.0, maxiter=None, M=None, callback=None):
    """Use Conjugate Residual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        `A` must represent a hermitian, positive definite matrix.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right-hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray, optional
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(r) <= max(rtol * norm(b), atol)`` should be satisfied.
    maxiter : integer, optional
        Maximum number of iterations. Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse array, ndarray, LinearOperator}, optional
        Preconditioner for `A`. Should approximate the inverse of `A`.
    callback : function, optional
        User-supplied function to call after each iteration. Called
        as ``callback(xk)``, where ``xk`` is the current solution vector.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations

    Notes
    -----
    The preconditioner `M` should be a matrix such that ``M @ A`` has a smaller
    condition number than `A`, see [2]_.

    References
    ----------
    .. [1] "Conjugate residual method, Wikipedia,
           https://en.wikipedia.org/wiki/Conjugate_residual_method
    .. [2] "Preconditioner",
           Wikipedia, https://en.wikipedia.org/wiki/Preconditioner

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = _get_atol_rtol("cr", bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0, np.inf

    n = len(b)
    if maxiter is None:
        maxiter = n * 10

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    matvec = A.matvec
    psolve = M.matvec
    r = b - matvec(x) if x.any() else b.copy()
    # r = psolve(r)
    # p = r.copy()
    p = np.zeros_like(r)
    beta = 0.0

    z = psolve(r)
    Az = matvec(z)
    zAz = dotprod(z, Az)

    # Ap = matvec(p)
    # Ar = matvec(r)
    # rAr = dotprod(r, Ar)
    # rAr_old = rAr.copy()

    for iteration in range(maxiter):
        # if np.linalg.norm(r) < atol:
        if np.linalg.norm(r) <= max(rtol * bnrm2, atol):
            return postprocess(x), iteration, r, True

        # see Algorithm 2 in https://www.sciencedirect.com/science/article/pii/S0377042708002264
        p = z + beta * p
        Ap = matvec(p)
        alpha = zAz / dotprod(psolve(Ap), Ap)
        x += alpha * p
        r -= alpha * Ap
        z = psolve(r)
        Az = matvec(z)
        beta = 1.0 / zAz
        zAz = dotprod(z, Az)
        beta *= zAz

        # z = psolve(Ap)
        # alpha = rAr_old / dotprod(Ap, z)
        # x += alpha * p
        # r -= alpha * z

        # Ar = matvec(r)
        # rAr = dotprod(r, Ar)
        # beta = rAr / rAr_old
        # rAr_old = rAr.copy()

        # p = r + beta * p
        # # Ap = Ar + beta * Ap
        # Ap = matvec(p)

        if callback:
            callback(x)

    return postprocess(x), maxiter, r, False


def nonlinear_cr(
    fun,
    x0,
    maxiter=int(1e2),
    atol=1e-3,
    rtol=1e-3,
    sigma1=1e-4,
    sigma2=1e-4,
    r=0.8,
    # r=0.95,
    t=0.5,
    tol=1e-3,
):
    """
    A derivative-free conjugate residual method using secant condition for
    general large-scale nonlinear equations.

    References
    ----------
    Zhang2019: https://link.springer.com/article/10.1007/s11075-019-00725-7
    """
    assert 0 < sigma1
    assert 0 < sigma2
    assert 0 < r < 1
    assert 0 < t

    def cost(x):
        F = fun(x)
        return 0.5 * np.dot(F, F)

    x = x0.copy()
    d = np.zeros_like(x)
    beta = 0.0
    F = fun(x)
    f = 0.5 * np.dot(F, F)
    scale = atol + np.abs(F) * rtol
    for iterations in range(maxiter):
        print(f" - iterations: {iterations}:")
        d = -F + beta * d

        # monotone line search
        k = 0
        alpha = r**k
        eta = 1 / (k + 1) ** 2
        d2 = np.dot(d, d)
        F2 = 2 * f
        x_new = x + alpha * d
        F_new = fun(x_new)
        f_new = 0.5 * np.dot(F_new, F_new)
        ls_converged = (
            f_new <= f - sigma1 * alpha**2 * d2 - sigma2 * alpha**2 * F2 + eta * f
        )
        # print(f"   line search:")
        # print(f"    * k: {k}; ls_converged: {ls_converged}; alpha: {alpha}; eta: {eta}")
        while not ls_converged:
            k += 1
            alpha = r**k
            eta = 1 / (k + 1) ** 2
            x_new = x + alpha * d
            F_new = fun(x_new)
            f_new = 0.5 * np.dot(F_new, F_new)
            ls_converged = (
                f_new <= f - sigma1 * alpha**2 * d2 - sigma2 * alpha**2 * F2 + eta * f
            )
            # print(
            #     f"    * k: {k}; ls_converged: {ls_converged}; alpha: {alpha}; eta: {eta}"
            # )

        s = x_new - x
        x = x_new.copy()
        F = F_new.copy()
        f = f_new
        # eps = 1e-8
        eps = 1e-6
        beta = np.abs(np.dot(F, d)) / np.abs(np.dot(d, (fun(x + eps * d) - F) / eps))
        beta = np.dot(
            F,
        )

        converged = f < tol
        error = np.linalg.norm(F / scale) / scale.size**0.5
        converged = error < 1
        # error = f
        # converged = error < tol
        print(f"   converged: {converged}; error: {error}; cost: {f}")
        if converged:
            break

    return x, iterations + 1, error

    ####################
    # old implementation
    ####################
    # x = x0.copy()
    # r = f(x)
    # scale = atol + np.abs(r) * rtol

    # # p = r.copy()
    # # q = f(x + p) - f(x)
    # p = np.zeros_like(r)
    # q = r.copy()
    # for iterations in range(maxiter):
    #     s = f(x + r) - r
    #     beta = np.dot(s, q) / np.dot(q, q)
    #     q = s - beta * p
    #     p = r - beta * p
    #     alpha = np.dot(r, q) / np.dot(q, q)
    #     x += alpha * p
    #     r = f(x)

    #     error = np.linalg.norm(r / scale) / scale.size**0.5
    #     if error < 1:
    #         return x, iterations, error

    # return x, iterations, error


# Test function
def test_cr():
    A = np.array([[4, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    x_true = np.linalg.solve(A, b)
    x, iteration, r = cr(A, b)
    print(f"iteration: {iteration}")
    print(f"x: {x}")
    print(f"r: {r}")
    assert np.allclose(
        x, x_true, atol=1e-5
    ), "CR solution does not match expected solution"


def test_cr_sparse():
    from scipy.sparse import random, triu, csr_array
    from scipy.sparse.linalg import spsolve, LinearOperator, spilu, cg

    def random_sparse_symmetric(n, density=0.01, format="csr", random_state=None):
        """Generate a large sparse symmetric matrix.

        Parameters
        ----------
        n : int
            Size of the matrix (n x n).
        density : float, optional
            Density of nonzero elements (default is 1%).
        format : str, optional
            Sparse format (default is 'csr').
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        scipy.sparse matrix
            Sparse symmetric matrix of size (n, n).
        """
        rng = np.random.default_rng(random_state)
        S = random(n, n, density=density, format="csr", random_state=rng)
        S = triu(S)  # Keep only upper triangular part
        S = csr_array(
            S + S.T - S.diagonal() * np.eye(n)
        )  # Symmetrize and keep diagonal
        return S.asformat(format)

    n = 1000  # Matrix size
    # A = random_sparse_symmetric(n, density=0.5)
    A = np.random.rand(n, n)
    A = A + A.T
    b = np.random.rand(n)
    print(f"spsolve")
    # x_true = spsolve(A, b)
    x_true = np.linalg.solve(A, b)
    print(f"done")
    M = spilu(A)
    preconditioner = LinearOperator(A.shape, lambda x: M.solve(x))
    # M = 1 / A.diagonal()
    # preconditioner = LinearOperator(A.shape, lambda x: M * x)
    # preconditioner = None
    print(f"cr")
    x, iteration, r = cr(A, b, M=preconditioner)
    # x, iteration = cg(A, b, M=preconditioner)
    r = None
    print(f"done")
    print(f"iteration: {iteration}")
    # print(f"x: {x}")
    # print(f"r: {r}")
    assert np.allclose(
        x, x_true, atol=1e-5
    ), "CR solution does not match expected solution"


def test_nonlinear_cr():
    # Problem 2

    def fun(x):
        F = np.zeros_like(x)
        F[0] = 2 * x[0] + np.sin(x[0]) - 1
        F[1:-1] = -2 * x[:-2] + 2 * x[1:-1] + np.sin(x[1:-1]) - 1
        F[-1] = 2 * x[-1] + np.sin(x[-1]) - 1
        return F

    n = 5
    x0 = np.zeros(n)

    x, iterations, error = nonlinear_cr(fun, x0)
    print(f"x: {x}")
    print(f"iterations: {iterations}")
    print(f"error: {error}")


if __name__ == "__main__":
    # test_cr()
    # test_cr_sparse()
    test_nonlinear_cr()
