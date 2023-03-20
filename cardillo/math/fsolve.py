import numpy as np
from warnings import warn
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, lsqr, lsmr, LinearOperator
from cardillo.math import approx_fprime

try:

    def sparse_qr_solve(A, b):
        """
        Solve the sparse linear system Ax=b, using PySPQR wrapper to SuitSparse's sparse QR-solve
        function.

        References:
        -----------
        PySPQR: https://github.com/yig/PySPQR \\
        SuiteSparseQR: http://faculty.cse.tamu.edu/davis/suitesparse.html
        """
        import sparseqr

        return sparseqr.solve(A, b, tolerance=0)

except:
    pass


def qr_solve(A, b):
    """
    Solve the sparse linear system Ax=b, using numpy's wrapper to dense Lapack
    functions lstsq_m/ lstsq_n.

    References:
    -----------
    numpy: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    """
    return np.linalg.lstsq(A.toarray(), b, rcond=None)[0]


def qr_overdetermined_solve(A, b):
    """Solve the sparse (overdetermined) linear system Ax=b using dense 
    QR-decomposition.
    
    References:
    Wiki1: https://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems \\
    Wiki2: https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution
    """
    # QR decomposition of A
    Q, R = np.linalg.qr(A.toarray())

    # solve triangular system
    from scipy.linalg import solve_triangular

    return solve_triangular(R, Q.T @ b)


def qr_underdetermined_solve(A, b):
    """Solve the sparse (underdetermined) linear system Ax=b using dense 
    QR-decomposition.
    
    References:
    Wiki1: https://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems \\
    Wiki2: https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution
    """
    # QR decomposition of A
    Q, R = np.linalg.qr(A.toarray().T)

    # solve triangular system
    from scipy.linalg import solve_triangular

    return Q @ solve_triangular(R.T, b, lower=True)


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


def qr_solve(A, b):
    """
    Solve the sparse linear system Ax=b, using numpy's wrapper to dense Lapack
    functions lstsq_m/ lstsq_n.
    References:
    -----------
    numpy: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    """
    return np.linalg.lstsq(A.toarray(), b, rcond=None)[0]


def qr_overdetermined_solve(A, b):
    """Solve the sparse (overdetermined) linear system Ax=b using dense 
    QR-decomposition.
    
    References:
    Wiki1: https://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems \\
    Wiki2: https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution
    """

    # solve triangular system
    from scipy.linalg import qr, solve_triangular

    # get shape of A
    m, n = A.shape

    # QR decomposition of A
    Q, R, p = qr(A.toarray(), pivoting=True, mode="full")

    x2 = np.zeros(n, dtype=float)
    x2[p] = solve_triangular(R, Q.T @ b)
    return x2


def qr_underdetermined_solve(A, b):
    """Solve the sparse (underdetermined) linear system Ax=b using dense 
    QR-decomposition.
    
    References:
    Wiki1: https://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems \\
    Wiki2: https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution
    """
    # QR decomposition of A
    Q, R = np.linalg.qr(A.toarray().T)

    # solve triangular system
    from scipy.linalg import solve_triangular

    return Q @ solve_triangular(R.T, b, lower=True)


def svd_solve(A, b):
    """See https://stackoverflow.com/a/59292892/7280763 and Lee2012 section 6.

    References:
    -----------
    Lee2012: http://math.uchicago.edu/~may/REU2012/REUPapers/Lee.pdf
    """
    # compute svd of A
    U, s, Vh = np.linalg.svd(A.toarray(), full_matrices=False)

    # from scipy.linalg import svd
    # U, s, Vh = svd(A.toarray())

    # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
    c = np.dot(U.T, b)
    # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w (trivial inversion of a diagonal matrix)
    w = np.dot(np.diag(1 / s), c)
    # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
    x = np.dot(Vh.conj().T, w)
    return x


def normal_equation_solve(A, b):
    # from scipy.linalg import solve
    # A = A.toarray()
    # return solve(A.T @ A, A.T @ b, assume_a="sym")

    A = A.toarray()
    AA = A.T @ A
    Ab = A.T @ b

    # # compute svd of A
    # U, s, Vh = np.linalg.svd(AA)

    from scipy.linalg import svd

    U, s, Vh = svd(AA)

    c = np.dot(U.T, Ab)
    w = np.dot(np.diag(1 / s), c)
    x = np.dot(Vh.conj().T, w)
    return x


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
    # linear_solver=lsqr_solve,
    # linear_solver=qr_solve,
    # linear_solver=sparse_qr_solve,
    # linear_solver=qr_overdetermined_solve,
    # linear_solver=qr_underdetermined_solve,
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
