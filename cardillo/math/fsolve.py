import numpy as np
from warnings import warn
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, lsqr, lsmr, LinearOperator
from cardillo.math import approx_fprime


def lu_solve(A, b):
    """
    Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

    References:
    -----------
    Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html#scipy.sparse.linalg.spsolve
    """
    return spsolve(A, b)


def lsqr_solve(A, b):
    """
    Find the least-squares solution to a large, sparse, linear system of
    equations Ax=b.

    References:
    -----------
    Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html#scipy.sparse.linalg.lsqr
    """
    return lsqr(A, b, atol=0, btol=0, conlim=0)[0]


def lsqr_operator_solve(A, b):
    """
    Find the least-squares solution to a large, sparse, linear system of
    equations Ax=b using scipy's LinearOperator's. This can give us a huge
    performance boost since we never have to assemble the sparse matrix if
    this is provided by the system.

    References:
    -----------
    Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html
    Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html#scipy.sparse.linalg.lsqr
    """

    def Ax(x):
        return A @ x

    def ATx(x):
        return A.T @ x

    operator = LinearOperator(A.shape, matvec=Ax, rmatvec=ATx)
    return lsqr(operator, b, atol=0, btol=0, conlim=0)[0]


def lsmr_solve(A, b):
    """
    Find the least-squares solution to a large, sparse, linear system of
    equations Ax=b.

    References:
    -----------
    Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html#scipy.sparse.linalg.lsmr
    """
    return lsmr(A, b, atol=0, btol=0, conlim=0)[0]


def fsolve(
    fun,
    x0,
    jac="2-point",
    fun_args=(),
    jac_args=(),
    error_function=lambda x: np.max(np.absolute(x)),
    atol=1.0e-8,
    eps=1.0e-6,
    max_iter=20,
    # linear_solver=lu_solve,
    linear_solver=lsqr_solve,
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
            approx_fprime(x, lambda y: fun(y, *args), eps=eps, method=jac)
        )
    else:
        jacobian = jac
    assert callable(jacobian)

    # prepare solution vector
    x = np.asarray(x0)

    # initial guess, error and convergence
    f = fun(x, *fun_args)
    error = error_function(f)
    converged = error <= atol

    # Newton loop
    i = 0
    while (not converged) and (i < max_iter):
        i += 1
        J = jacobian(x, *jac_args)
        x -= linear_solver(J, f)
        f = fun(x, *fun_args)
        error = error_function(f)
        converged = error <= atol

    if not converged:
        warn(f"fsolve is not converged after {i} iterations with error {error:2.3f}")

    return x, converged, error, i, f
