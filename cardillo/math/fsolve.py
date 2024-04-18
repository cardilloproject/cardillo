import numpy as np
from scipy.linalg import qr, solve_triangular, svd
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse.linalg._dsolve._superlu import SuperLU
from scipy.optimize import OptimizeResult
from warnings import warn

from cardillo.math.approx_fprime import approx_fprime
from cardillo.solver import SolverOptions


def lu_solve(A, b):
    """
    Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

    References:
    -----------
    Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html#scipy.sparse.linalg.spsolve
    """
    return spsolve(A, b)


def rank_revealing_qr_solve(A, b, singular_decimals=12, verbose=True):
    """
    Solves over-determined and underdetemined linar systems :math:`Ax=b`, see [1] and [2].

    Parameters
    ----------
    A : (m, n) array_like
        Any matrix.

    b : ndarray
        A column vector to solve for with `m` rows.

    singular_decimals : int, optional
        Number of decimal places to round to for identification of rank 
        deficiency (default: 12).

    Returns
    -------
    x : ndarray
        Solution to the linear system equation
    Notes
    -----
    This uses QR-Decomposition using the permutation matrix to find a soultion to the
    given linear system `Ax=b` through back substitution.

    References
    ----------
    [1] https://inst.eecs.berkeley.edu/~ee127/sp21/livebook/l_lineqs_solving.html \\
    [2] https://github.com/scipy/scipy/blob/f74bbcef8e382e0d5c4277babeeb558ca6f91714/scipy/linalg/_qr_solve.py
    """
    # convert to numpy arrays
    A = np.asarray(A.toarray())
    b = np.asarray(b)

    # shape of input matrix
    m, n = A.shape

    # dense QR-decomposition with column pivoting
    Q, R, p = qr(A, pivoting=True)

    # the number of linearly independent columns & rows in A (the rank)
    rank = sum(np.around(np.diag(R), singular_decimals) != 0)

    # detect rank deficiency
    if verbose and (n - rank) > 0:
        print(f"rank deficiency!")

    # new rhs
    c = Q.T @ b
    c1 = c[:rank]

    # extract r by r upper left block of R
    R11 = R[:rank, :rank]

    # compute basic solution
    z = np.zeros(n - rank)
    y = solve_triangular(R11, c1)

    # concatenate both solutions
    xp = np.concatenate((y, z))

    # apply permutation
    x = np.zeros_like(b)
    x[p] = xp

    return x


def svd_solve(A, b, verbose=True):
    """See https://stackoverflow.com/a/59292892/7280763 and Lee2012 section 6.

    References:
    -----------
    Lee2012: http://math.uchicago.edu/~may/REU2012/REUPapers/Lee.pdf
    """
    # compute svd of A with matlab default driver "gesvd",
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html
    U, s, Vh = svd(A.toarray(), lapack_driver="gesvd")

    # shape of input matrix
    m, n = A.shape

    # compute matrix rank of R with given precision
    positive = s > 1.0e-12
    nonzero_idx = np.where(positive)[0]
    rank = len(nonzero_idx)

    if verbose:
        print(f"n: {m}, m: {n}, rank: {rank}")

    # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
    c = np.dot(U.T, b)

    # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w
    w = np.zeros_like(b)
    w[nonzero_idx] = np.dot(np.diag(1 / s[nonzero_idx]), c[nonzero_idx])

    # Vh x = w <=> x = Vh.H w
    xk1 = np.dot(Vh.conj().T, w)

    return xk1


def pinv_solve(A, b, rcond=1e-14, *args):
    return np.linalg.pinv(A.toarray(), rcond=rcond) @ b


def fsolve(
    fun,
    x0,
    jac=None,
    fun_args=(),
    jac_args=(),
    inexact=False,
    options=SolverOptions(),
) -> tuple[np.ndarray, bool, float, int, np.ndarray]:
    """Solve a nonlinear system of equations using (inexact) Newton method.
    This function is inspired by scipy's `solve_collocation_system` found
    in `scipy.integrate._ivp.radau`. Absolute and relative errors are used 
    to terminate the iteration in accordance with Kelly1995 (1.12). See also 
    Hairer1996 below (8.21).

    Parameters
    ----------
    fun : callable
        Nonlinear function with signature `fun(x, *fun_args)`.
    x0 : ndarray, shape (n,)
        Initial guess.
    jac : callable, SuperLU, optional
        Function defining the sparse Jacobian of `fun`. Alternatvely, this
        can be an `SuperLU` object. Then, an inexact Newton method is
        performed, see `inexact`.
    fun_args: tuple
        Additional arguments passed to `fun`.
    jac_args: tuple
        Additional arguments passed to `jac`.
    inexact: Bool, optional
        Apply inexact Newton method (Newton chord) with constant `J = jac(x0)`.
    options: SolverOptions
        Defines all required solver options.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: `x` the solution array, `success` a
        Boolean flag indicating if the optimizer exited successfully, `error` 
        the relative error, `fun` the current function value and `nit`, 
        `nfev`, `njev` the counters for number of iterations, function and 
        Jacobian evaluations, respectively.

    References
    ----------
    Kelly1995: https://epubs.siam.org/doi/book/10.1137/1.9780898718898 \\
    Haireri1996: https://link.springer.com/book/10.1007/978-3-642-05221-7
    """
    nit = 0
    nfev = 0
    njev = 0

    if not isinstance(fun_args, tuple):
        fun_args = (fun_args,)
    if not jac_args:
        jac_args = fun_args
    elif not isinstance(jac_args, tuple):
        jac_args = (jac_args,)

    # wrap function
    def fun(x, *args, f=fun):
        nonlocal nfev
        nfev += 1
        return np.atleast_1d(f(x, *args))

    # compute Jacobian matrix using finite differences
    if jac is None or options.numerical_jacobian_method:

        def jacobian(x, *args):
            nonlocal njev
            njev += 1
            return csc_array(
                approx_fprime(
                    x,
                    lambda y: fun(y, *args),
                    method=options.numerical_jacobian_method,
                    eps=options.numerical_jacobian_eps,
                )
            )

    else:
        # do inexact newton with given LU-decomposition
        if isinstance(jac, SuperLU):
            inexact = True
            lu = jac
        else:
            assert callable(jac), "user-defined jacobian must be callable"

            # wrap jacobian
            def jacobian(x, *args):
                nonlocal njev
                njev += 1
                return jac(x, *args)

            if inexact:
                lu = splu(jacobian(x0, *jac_args))
                njev += 1

    if inexact:

        def solve(x, rhs):
            return lu.solve(rhs)

    else:

        def solve(x, rhs):
            return options.linear_solver(jacobian(x, *jac_args), rhs)

    # eliminate round-off errors
    Delta_x = np.zeros_like(x0)
    x = x0 + Delta_x

    # initial function value
    f = np.atleast_1d(fun(x, *fun_args))

    # scaling with relative and absolute tolerances
    scale = options.newton_atol + np.abs(f) * options.newton_rtol

    # error of initial guess
    error = np.linalg.norm(f / scale) / scale.size**0.5
    converged = error < 1

    # Newton loop
    if not converged:
        for i in range(options.newton_max_iter):
            # Newton update
            dx = solve(x, f)
            Delta_x -= dx
            x = x0 + Delta_x

            # new function value, error and convergence check
            f = np.atleast_1d(fun(x, *fun_args))
            error = np.linalg.norm(f / scale) / scale.size**0.5
            converged = error < 1
            if converged:
                break

        if not converged:
            warn(f"fsolve is not converged after {i} iterations with error {error:.2e}")

        nit = i + 1

    return OptimizeResult(
        x=x,
        success=converged,
        error=error,
        fun=f,
        nit=nit,
        nfev=nfev,
        njev=njev,
    )
