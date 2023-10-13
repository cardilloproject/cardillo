import numpy as np
from warnings import warn
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import qr, solve_triangular, svd
from cardillo.math import approx_fprime


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


def fsolve(
    fun,
    x0,
    jac="3-point",
    fun_args=(),
    jac_args=(),
    error_function=lambda x: np.max(np.absolute(x)),
    atol=1.0e-8,
    eps=1.0e-6,
    max_iter=20,
    linear_solver=lu_solve,
):
    if not isinstance(fun_args, tuple):
        fun_args = (fun_args,)
    if not jac_args:
        jac_args = fun_args
    elif not isinstance(jac_args, tuple):
        jac_args = (jac_args,)

    # compute Jacobian matrix using finite differences
    if jac in ["2-point", "3-point", "cs"]:
        jacobian = lambda x, *args: csc_matrix(
            approx_fprime(x, lambda y: fun(y, *args), method=jac, eps=eps)
        )
    else:
        jacobian = jac
    assert callable(jacobian)

    # prepare solution vector; make a copy since we modify the value
    x = np.atleast_1d(x0).copy()

    # initial guess, error and convergence
    f = np.atleast_1d(fun(x, *fun_args))
    error = error_function(f)
    converged = error <= atol

    # Newton loop
    i = 0
    while (not converged) and (i < max_iter):
        i += 1
        J = jacobian(x, *jac_args)
        x -= linear_solver(J, f)
        f = np.atleast_1d(fun(x, *fun_args))
        error = error_function(f)
        converged = error <= atol

    if not converged:
        warn(f"fsolve is not converged after {i} iterations with error {error:2.3f}")

    return x, converged, error, i, f
