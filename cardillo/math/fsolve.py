import numpy as np
from warnings import warn
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, lsqr
from cardillo.math import approx_fprime

# def linear__solver(A, b):
#     return spsolve(A, b)

#     # return lsqr(A, b, atol=1.0e-12, btol=1.0e-12)[0]

#     # # no underflow errors
#     # return np.linalg.lstsq(A.toarray(), b, rcond=None)[0]

#     # # TODO: Can we get this sparse?
#     # # using QR decomposition, see https://de.wikipedia.org/wiki/QR-Zerlegung#L%C3%B6sung_regul%C3%A4rer_oder_%C3%BCberbestimmter_Gleichungssysteme
#     # Q, R = np.linalg.qr(A.toarray())
#     # z = Q.T @ b
#     # return np.linalg.solve(R, z)  # solving R*x = Q^T*b

#     # # solve normal equation (should be independent of the conditioning
#     # # number!)
#     # return spsolve(A.T @ A, A.T @ b)


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
):
    if not isinstance(fun_args, tuple):
        fun_args = (fun_args,)
    if not isinstance(jac_args, tuple):
        jac_args = (jac_args,)

    # compute Jacobian matrix using finite differences
    if jac in ["2-point", "3-point", "cs"]:
        jac = lambda x, jac_args: csc_matrix(
            approx_fprime(lambda x: f(x, *jac_args), x, eps=eps, method=jac)
        )
    else:
        assert callable(jac)

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
        J = jac(x, *jac_args)
        x -= spsolve(J, f)
        # x -= linear__solver(J, f)
        f = fun(x, *fun_args)
        error = error_function(f)
        converged = error <= atol

    if not converged:
        warn(f"fsolve is not converged after {i} iterations with error {error:2.3f}")

    return x, converged, error, i, f
