import numpy as np
from scipy.sparse.linalg._isolve.iterative import _get_atol_rtol
from scipy.sparse.linalg._isolve.utils import make_system


# inspired by scipy's conjugate gradient method,
# see https://github.com/scipy/scipy/blob/v1.15.2/scipy/sparse/linalg/_isolve/iterative.py#L0-L1
def cr(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):
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

    atol, _ = _get_atol_rtol('cr', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    n = len(b)
    if maxiter is None:
        maxiter = n * 10

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    matvec = A.matvec
    psolve = M.matvec
    r = b - matvec(x) if x.any() else b.copy()
    r = psolve(r)
    p = r.copy()

    Ap = matvec(p)
    Ar = matvec(r)
    rAr = dotprod(r, Ar)
    rAr_old = rAr.copy()

    for iteration in range(maxiter):
        # if np.linalg.norm(r) < atol:
        if np.linalg.norm(r) <= max(rtol * bnrm2, atol):
            return postprocess(x), iteration, r

        z = psolve(Ap)
        alpha = rAr_old / dotprod(Ap, z)
        x += alpha * p
        r -= alpha * z

        Ar = matvec(r)
        rAr = dotprod(r, Ar)
        beta = rAr / rAr_old
        rAr_old = rAr.copy()

        p = r + beta * p
        # Ap = Ar + beta * Ap
        Ap = matvec(p)

        if callback:
            callback(x)

    return postprocess(x), maxiter, r

# Test function
def test_cr():
    A = np.array([[4, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    x_true = np.linalg.solve(A, b)
    x, iteration, r = cr(A, b)
    print(f"iteration: {iteration}")
    print(f"x: {x}")
    print(f"r: {r}")
    assert np.allclose(x, x_true, atol=1e-5), "CR solution does not match expected solution"

def test_cr_sparse():
    from scipy.sparse import random, triu, csr_array
    from scipy.sparse.linalg import spsolve, LinearOperator, spilu, cg

    def random_sparse_symmetric(n, density=0.01, format='csr', random_state=None):
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
        S = random(n, n, density=density, format='csr', random_state=rng)
        S = triu(S)  # Keep only upper triangular part
        S = csr_array(S + S.T - S.diagonal() * np.eye(n))  # Symmetrize and keep diagonal
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
    preconditioner = None
    print(f"cr")
    x, iteration, r = cr(A, b, M=preconditioner)
    # x, iteration = cg(A, b, M=preconditioner)
    r = None
    print(f"done")
    print(f"iteration: {iteration}")
    # print(f"x: {x}")
    # print(f"r: {r}")
    assert np.allclose(x, x_true, atol=1e-5), "CR solution does not match expected solution"

# test_cr()
test_cr_sparse()
