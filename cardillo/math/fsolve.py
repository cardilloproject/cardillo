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


def lu_solve(A, b, *args):
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
    # return lsqr(A, b, damp=1e-4, atol=0, btol=0, conlim=0)[0]


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
    # return lsmr(A, b, damp=1e-4, atol=0, btol=0, conlim=0)[0]


def qr_overdetermined_singular_solve(A, b, xk=None, rank_precision=12):
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

    # # QR decomposition of A
    # Q, R, p = qr(A.toarray(), pivoting=True, mode="full")
    # # Q, R, p = qr(A.toarray(), pivoting=True, mode="economic")

    import sparseqr

    Q, R, p, rank = sparseqr.qr(A)
    # Q = Q.toarray()
    # R = R.toarray()

    # x2 = np.zeros_like(b)
    # x2[p] = solve_triangular(R, Q.T @ b)
    # error = np.linalg.norm(A @ x2 - b)
    # print(f"error qr_solve: {error}")
    # return x2

    # # compute matrix rank of R with given precision
    # rank = sum(np.around(np.diag(R), rank_precision) != 0)
    # print(f"n: {n}; rank: {rank}")

    # extract left columns of Q
    from scipy.sparse import csc_matrix, csr_matrix

    Q1 = csc_matrix(Q)[:, :rank]

    # extract upper left block of R
    R11 = csr_matrix(R)[:rank, :rank]

    # padd solution with zeros
    xp = np.zeros_like(b)
    # xp[:rank] = solve_triangular(R11, Q1.T @ b)
    from scipy.sparse.linalg import spsolve_triangular

    xp[:rank] = spsolve_triangular(R11, Q1.T @ b, lower=False)

    # apply permutation
    P = np.eye(len(p))[:, p]
    x = P @ xp

    # # compute error
    # error = np.linalg.norm(A @ x - b)
    # print(f"error qr_solve: {error}")
    return x


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


def svd_solve(A, b, xk):
    """See https://stackoverflow.com/a/59292892/7280763 and Lee2012 section 6.

    References:
    -----------
    Lee2012: http://math.uchicago.edu/~may/REU2012/REUPapers/Lee.pdf
    """
    # compute svd of A with matlab default driver "gesvd",
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html
    from scipy.linalg import svd

    U, s, Vh = svd(A.toarray(), lapack_driver="gesvd")

    # shape of input matrix
    m, n = A.shape

    # compute matrix rank of R with given precision
    positive = s > 1.0e-12
    nonzero_idx = np.where(positive)[0]
    zero_idx = np.where(~positive)[0]
    rank = len(nonzero_idx)

    # rank_precision = 12
    # rank = sum(np.around(s, rank_precision) != 0)

    # print(f"s: {s}")
    print(f"n: {m}, m: {n}, rank: {rank}")

    # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
    c = np.dot(U.T, b)

    # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w (trivial inversion of a diagonal matrix)
    # w = np.dot(np.diag(1 / s), c)
    w = np.zeros_like(b)
    w[nonzero_idx] = np.dot(np.diag(1 / s[nonzero_idx]), c[nonzero_idx])

    # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
    xk1 = np.dot(Vh.conj().T, w)

    return xk1


def MNGN_svd(J, rx, xk, x0):
    """Minimal norm Gauss-Newton (MNGN) solved with svd decomposition,
    see Pes2020, section 3."""
    # compute svd of A with matlab default driver "gesvd",
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html
    from scipy.linalg import svd

    U, s, Vh = svd(J.toarray(), lapack_driver="gesvd")
    # U, s, Vh = svd(J.toarray(), lapack_driver="gesdd")

    # shape of input matrix
    m, n = J.shape

    # compute matrix rank of R with given precision
    # s[3:5] = 0
    positive = s > 1.0e-12
    nonzero_idx = np.where(positive)[0]
    zero_idx = np.where(~positive)[0]
    rank = len(nonzero_idx)
    # print(f"nonzero_idx: {nonzero_idx}")
    # print(f"zero_idx: {zero_idx}")
    # print(f"rank: {rank}")
    if rank < n:
        print(f"rank deficency!")

    b = U.T @ rx

    y = np.zeros_like(rx)
    y[nonzero_idx] = b[nonzero_idx] / s[nonzero_idx]
    # y[zero_idx] = Vh[zero_idx] @ xk
    # y[zero_idx] = Vh[zero_idx] @ (xk - x0)
    # return Vh.conj().T @ y
    return Vh.T @ y

    # Delta_x = -(
    #     Vh.T[:, nonzero_idx] @ (b[nonzero_idx] / s[nonzero_idx])
    #     # + Vh.T[:, zero_idx] @ (Vh.T[:, zero_idx].T @ xk) # TODO: Is this correct?
    #     + (np.eye(m) - Vh.T[:, nonzero_idx] @ Vh.T[:, nonzero_idx].T) @ xk
    # )

    return -Delta_x


def MNGN_qr(A, b, xk=None, x0=None, rank_precision=12):
    """Solve the sparse (overdetermined) linear system Ax=b with rank
    deficiency using column pivoted QR decomposition.
    """
    from scipy.linalg import qr, solve_triangular

    # get shape of A
    m, n = A.shape

    # QR decomposition of A
    Q, R, p = qr(A.toarray(), pivoting=True, mode="full")
    # Q, R, p = qr(A.toarray(), pivoting=True, mode="economic")
    # print(f"Q.shape: {Q.shape}, R.shape: {R.shape}")

    # build permutation matrix
    Pi = np.eye(len(p))[:, p]

    # compute matrix rank of R with given precision
    rank = sum(np.around(np.diag(R), rank_precision) != 0)
    # print(f"n: {n}; rank: {rank}")

    # extract left columns of Q
    Q1 = Q[:, :rank]
    Q2 = Q[:, rank:]
    # TODO: Use Q @ b and split with rank will be more efficient
    c = Q1.T @ b
    d = Q2.T @ b

    # extract upper left block of R
    R11 = R[:rank, :rank]
    R12 = R[:rank, rank:]

    if n - rank > 0:
        print(f"rank deficiency!")

    # basic solution
    z = np.zeros(n - rank)

    # # use guess form last iteration
    # z = xk[p][rank :]
    # print(f"z: {z}")

    # # compute z part that minimizes ||(R11 y , z)||^2
    # z = np.linalg.solve(R12.T @ R12 + np.eye(n - rank), -c.T @ R12)
    # print(f"z: {z}")

    # # compute z part that minimizes ||(y , z)||^2
    # R11_inv = np.linalg.inv(R11)
    # M = R11_inv.T @ R11_inv
    # z = np.linalg.solve(R12.T @ M @ R12 + np.eye(n - rank), - c.T @ M @ R12)
    # print(f"z: {z}")

    # padd solution with zeros
    # xp = np.zeros_like(b)
    # xp[:rank] = solve_triangular(R11, c)
    y = solve_triangular(R11, c - R12 @ z)
    xp = np.concatenate((y, z))

    # apply permutation
    x = Pi @ xp

    # # compute error
    # error = np.linalg.norm(A @ x - b)
    # print(f"error qr_solve: {error}")
    return x


def normal_equation_solve(A, b, use_svd=True):
    A = A.toarray()
    AA = A.T @ A
    Ab = A.T @ b

    if use_svd:
        from scipy.linalg import svd

        U, s, Vh = svd(AA)

        c = np.dot(U.T, Ab)
        w = np.dot(np.diag(1 / s), c)
        x = np.dot(Vh.conj().T, w)
        return x
    else:
        from scipy.linalg import solve

        return solve(A.T @ A, A.T @ b, assume_a="sym")


def normal_equation_tikhonov_solve(A, b):
    A = A.toarray()
    AA = A.T @ A
    Ab = A.T @ b

    alpha = 1e-7
    G = alpha * np.eye(len(b))
    GG = G.T @ G

    from scipy.linalg import solve

    return solve(AA + GG, Ab, assume_a="sym")

    from scipy.linalg import svd

    U, s, Vh = svd(AA + GG)

    c = np.dot(U.T, Ab)
    w = np.dot(np.diag(1 / s), c)
    x = np.dot(Vh.conj().T, w)
    return x


# see https://github.com/scipy/scipy/blob/f74bbcef8e382e0d5c4277babeeb558ca6f91714/scipy/linalg/_qr_solve.py
def qr_solve2(A, y, silent=True):
    """
    Solves over-determined and underdetemined linar systems :math:`y=Ax`.
    Parameters
    ----------
    A : (M, N) array_like
        Any matrix.

    y : ndarray
        A column vector to solve for with `M` rows.

    silent : {'True', 'False'}, optional
        To log if the solution is a true solution.
    Returns
    -------
    x : ndarray
        Solution to the linear system equation
    Notes
    -----
    This uses QR-Decomposition using the permutation matrix to find a soultion to the
    given linear system `y=Ax` through back substitution.
    References
    ----------
    .. [1] https://inst.eecs.berkeley.edu/~ee127/sp21/livebook/l_lineqs_solving.html
    Examples
    --------
    Given `A` is `m\times n`, `x \in \mathbb{R}^{n}` `y\in\mathbb{R}^m` in the equation `y=Ax`, solve for `x`:
    >>> import numpy as np
    >>> from scipy import linalg
    >>> A = np.random.rand( m, n )
    >>> x = np.random.rand( n )
    >>> y = A@x
    >>> x_0 = _qr_solve.qr_solve(A, y)
    >>> np.allclose(y, A@x_0)
    True
    """

    A = np.asarray(A.toarray())
    y = np.asarray(y)

    # Solving y=Ax for an x_0
    # A has m rows and n columns -> y has m rows and x has n rows
    m, n = A.shape

    # QR decomposition
    from scipy.linalg import qr

    Q, R, P = qr(A, pivoting=True)
    # Q is an m by m orthogonal matrix
    # R is an m by n upper right triangle matrix

    # P is a permuntation matrix with n rows and n columns
    # P can order A by rank for *rank revealing*
    P = np.eye(len(P))[:, P]

    # Let r be the number of linearly independent columns & rows in A (the rank)
    rank = r = sum(np.around(np.diag(R), 12) != 0)

    # Q is a m by m square orthogonal matrix
    # Q_1 has m rows and r columns
    Q_1 = Q[:, :rank]

    # R_1 is an r by r upper right triangular matrix
    R_1 = R[:rank, :rank]

    # R_2 is an r by m-r matrix
    R_2 = R[:, : -(rank - m)]

    # z_1 is a column vector with r elements
    # z_1 = scipy.linalg.solve(R_1,Q_1.T@y)
    z_1 = np.linalg.solve(R_1, Q_1.T @ y)

    # We pad z_1 with r-m zeros at the bottom for a solution vector
    padding = np.zeros(n - r)  # zero padding
    x_0 = P @ np.hstack([z_1, padding])  # Add padding

    # if not silent:
    #     #Log if there was a solution
    #     is_solution = np.allclose(y,A@x_0)
    #     logging.info("Solution Found!") if is_solution else loggin.info("No Solution!")

    return x_0


def lm_solve(A, b):
    from scipy.optimize import least_squares

    # fun = lambda x: A @ x - b
    # jac = lambda x: A.toarray()

    AA = A.T @ A
    Ab = A.T @ b
    fun = lambda x: AA @ x - Ab
    jac = lambda x: AA.toarray()

    sol = least_squares(fun, np.zeros_like(b), jac=jac, method="lm")
    return sol.x


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
    # linear_solver=lu_solve,
    # linear_solver=MNGN_svd,
    linear_solver=MNGN_qr,
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

        # x -= linear_solver(J, f, x)
        # f = np.atleast_1d(fun(x, *fun_args))
        # error = error_function(f)

        dx = linear_solver(J, f, x, x0)
        x -= dx
        f = np.atleast_1d(fun(x, *fun_args))
        # TODO: Does ||f|| work in all cases?
        error = error_function(dx)
        # error = error_function(f)

        converged = error <= atol

    if not converged:
        warn(f"fsolve is not converged after {i} iterations with error {error:2.3f}")

    return x, converged, error, i, f


def mngn(
    fun,
    x0,
    jac="3-point",
    fun_args=(),
    jac_args=(),
    error_function=lambda x: np.max(np.absolute(x)),
    atol=1.0e-8,
    eps=1.0e-6,
    max_iter=20,
    linear_solver=MNGN_svd,
):
    """Minimal norm Gauss-Newton, see Pes2020."""
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
        dx = linear_solver(J, f, x)
        x -= dx
        # x -= linear_solver(J, f, x)
        f = np.atleast_1d(fun(x, *fun_args))
        # error = error_function(f)
        # error = error_function(J.T @ f)
        error = error_function(dx)
        converged = error <= atol

    if not converged:
        warn(f"fsolve is not converged after {i} iterations with error {error:2.3f}")

    return x, converged, error, i, f
