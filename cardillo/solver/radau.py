import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix, csr_matrix, issparse, eye, bmat
from scipy.sparse.linalg import spsolve, splu
from scipy.optimize._numdiff import group_columns
from scipy.integrate._ivp.common import (
    validate_max_step,
    validate_tol,
    select_initial_step,
    norm,
    num_jac,
    EPS,
    warn_extraneous,
    validate_first_step,
)
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
from scipy.optimize._numdiff import approx_derivative
from scipy.integrate import solve_ivp
from tqdm import tqdm

from cardillo.solver import Solution

S6 = 6**0.5

# Butcher tableau. A is not used directly, see below.
C = np.array([(4 - S6) / 10, (4 + S6) / 10, 1])
E = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3

# Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
# and a complex conjugate pair. They are written below.
MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
MU_COMPLEX = (
    3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3)) - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6))
)

# These are transformation matrices.
T = np.array(
    [
        [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
        [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
        [1, 1, 0],
    ]
)
TI = np.array(
    [
        [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
        [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
        [0.50287263494578682, -2.57192694985560522, 0.59603920482822492],
    ]
)
# These linear combinations are used in the algorithm.
TI_REAL = TI[0]
TI_COMPLEX = TI[1] + 1j * TI[2]

# Interpolator coefficients.
P = np.array(
    [
        [13 / 3 + 7 * S6 / 3, -23 / 3 - 22 * S6 / 3, 10 / 3 + 5 * S6],
        [13 / 3 - 7 * S6 / 3, -23 / 3 + 22 * S6 / 3, 10 / 3 - 5 * S6],
        [1 / 3, -8 / 3, 10 / 3],
    ]
)


NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


def solve_collocation_system(
    fun, t, y, h, Z0, scale, tol, LU_real, LU_complex, solve_lu, mass_matrix
):
    """Solve the collocation system.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    h : float
        Step to try.
    Z0 : ndarray, shape (3, n)
        Initial guess for the solution. It determines new values of `y` at
        ``t + h * C`` as ``y + Z0``, where ``C`` is the Radau method constants.
    scale : ndarray, shape (n)
        Problem tolerance scale, i.e. ``rtol * abs(y) + atol``.
    tol : float
        Tolerance to which solve the system. This value is compared with
        the normalized by `scale` error.
    LU_real, LU_complex
        LU decompositions of the system Jacobians.
    solve_lu : callable
        Callable which solves a linear system given a LU decomposition. The
        signature is ``solve_lu(LU, b)``.
    mass_matrix : {None, array_like, sparse_matrix},
           Defines the constant mass matrix of the system, with shape (n,n).
           It may be singular, thus defining a problem of the differential-
           algebraic type (DAE).

    Returns
    -------
    converged : bool
        Whether iterations converged.
    n_iter : int
        Number of completed iterations.
    Z : ndarray, shape (3, n)
        Found solution.
    rate : float
        The rate of convergence.
    """
    n = y.shape[0]
    M_real = MU_REAL / h
    M_complex = MU_COMPLEX / h

    W = TI.dot(Z0)
    Z = Z0

    F = np.empty((3, n))
    ch = h * C

    dW_norm_old = None
    dW = np.empty_like(W)
    converged = False
    rate = None
    for k in range(NEWTON_MAXITER):
        for i in range(3):
            F[i] = fun(t + ch[i], y + Z[i])

        if not np.all(np.isfinite(F)):
            break

        f_real = F.T.dot(TI_REAL) - M_real * mass_matrix.dot(W[0])
        f_complex = F.T.dot(TI_COMPLEX) - M_complex * mass_matrix.dot(W[1] + 1j * W[2])

        dW_real = solve_lu(LU_real, f_real)
        dW_complex = solve_lu(LU_complex, f_complex)

        dW[0] = dW_real
        dW[1] = dW_complex.real
        dW[2] = dW_complex.imag

        dW_norm = norm(dW / scale)
        if dW_norm_old is not None:
            rate = dW_norm / dW_norm_old

        if rate is not None and (
            rate >= 1 or rate ** (NEWTON_MAXITER - k) / (1 - rate) * dW_norm > tol
        ):
            break

        W += dW
        Z = T.dot(W)

        if dW_norm == 0 or rate is not None and rate / (1 - rate) * dW_norm < tol:
            converged = True
            break

        dW_norm_old = dW_norm

    return converged, k + 1, Z, rate


def predict_factor(h_abs, h_abs_old, error_norm, error_norm_old):
    """Predict by which factor to increase/decrease the step size.

    The algorithm is described in [1]_.

    Parameters
    ----------
    h_abs, h_abs_old : float
        Current and previous values of the step size, `h_abs_old` can be None
        (see Notes).
    error_norm, error_norm_old : float
        Current and previous values of the error norm, `error_norm_old` can
        be None (see Notes).

    Returns
    -------
    factor : float
        Predicted factor.

    Notes
    -----
    If `h_abs_old` and `error_norm_old` are both not None then a two-step
    algorithm is used, otherwise a one-step algorithm is used.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems", Sec. IV.8.
    """
    if error_norm_old is None or h_abs_old is None or error_norm == 0:
        multiplier = 1
    else:
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25

    with np.errstate(divide="ignore"):
        factor = min(1, multiplier) * error_norm**-0.25

    return factor


class Radau(OdeSolver):
    """Implicit Runge-Kutta method of Radau IIA family of order 5.

    The implementation follows [1]_. The error is controlled with a
    third-order accurate embedded formula. A cubic polynomial which satisfies
    the collocation conditions is used for the dense output.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below). The
        vectorized implementation allows a faster approximation of the Jacobian
        by finite differences (required for this solver).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. HHere `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : {None, array_like, sparse_matrix, callable}, optional
        Jacobian matrix of the right-hand side of the system with respect to
        y, required by this method. The Jacobian matrix has shape (n, n) and
        its element (i, j) is equal to ``d f_i / d y_j``.
        There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)`` as necessary.
              For the 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
        elements in *each* row, providing the sparsity structure will greatly
        speed up the computations [2]_. A zero entry means that a corresponding
        element in the Jacobian is always zero. If None (default), the Jacobian
        is assumed to be dense.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.
    mass : {None, array_like, sparse_matrix}, optional
        Defines the constant mass matrix of the system, with shape (n,n).
        It may be singular, thus defining a problem of the differential-
        algebraic type (DAE), see [1]. The default value is None.
    index_algebraic_vars : {None, array_like}, optional
        This is important for differential-algebraic systems of differentiation
        index > 1. In estimating the error the index 2 variables are multiplied
        by h, the index 3 variables by h^2, etc. If not provided all equations
        are assumed to define ode's.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.

    References
    ----------
    .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13, pp. 117-120, 1974.
    """

    def __init__(
        self,
        fun,
        t0,
        y0,
        t_bound,
        max_step=np.inf,
        rtol=1e-3,
        atol=1e-6,
        jac=None,
        jac_sparsity=None,
        vectorized=False,
        first_step=None,
        mass_matrix=None,
        index_array=None,
        **extraneous,
    ):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized)
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)
        # Select initial step assuming the same order which is used to control
        # the error.
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun,
                self.t,
                self.y,
                self.f,
                self.direction,
                3,
                self.rtol,
                self.atol,
            )
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.h_abs_old = None
        self.error_norm_old = None

        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol**0.5))
        self.sol = None

        self.jac_factor = None
        self.jac, self.J = self._validate_jac(jac, jac_sparsity)
        self._nlusove = 0
        if issparse(self.J):

            def lu(A):
                self.nlu += 1
                return splu(A)

            def solve_lu(LU, b):
                self._nlusove += 1
                return LU.solve(b)

            I = eye(self.n, format="csc")
        else:

            def lu(A):
                self.nlu += 1
                return lu_factor(A, overwrite_a=True)

            def solve_lu(LU, b):
                self._nlusove += 1
                return lu_solve(LU, b, overwrite_b=True)

            I = np.identity(self.n)

        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I

        (
            self.mass_matrix,
            self.index_algebraic_vars,
            self.nvars_algebraic,
        ) = self._validate_mass_matrix(mass_matrix)

        self.current_jac = True
        self.LU_real = None
        self.LU_complex = None
        self.Z = None

        self.index_array = self._validate_index_array(index_array)

    def _validate_index_array(self, index_array):
        if index_array is None:
            index_array = np.zeros(len(self.y), dtype=int)
            assert self.nvars_algebraic == 0
        else:
            index_array = np.asarray(index_array, dtype=int)
            index_array = np.maximum(0, index_array - 1)
            assert len(index_array) == len(self.y)
        return index_array

    def _validate_mass_matrix(self, mass_matrix):
        if mass_matrix is None:
            M = self.I
            index_algebraic_vars = None
            nvars_algebraic = 0
        elif callable(mass_matrix):
            raise ValueError(
                "`mass_matrix` should be a constant matrix, but is" " callable"
            )
        else:
            if issparse(mass_matrix):
                M = csc_matrix(mass_matrix)
                index_algebraic_vars = np.where(np.all(M.toarray() == 0, axis=1))[0]
            else:
                M = np.asarray(mass_matrix, dtype=float)
                index_algebraic_vars = np.where(np.all(M == 0, axis=1))[0]
            if M.shape != (self.n, self.n):
                raise ValueError(
                    "`mass_matrix` is expected to have shape {}, "
                    "but actually has {}.".format((self.n, self.n), M.shape)
                )
            nvars_algebraic = index_algebraic_vars.size

        return M, index_algebraic_vars, nvars_algebraic

    def _validate_jac(self, jac, sparsity):
        t0 = self.t
        y0 = self.y

        if jac is None:
            if sparsity is not None:
                if issparse(sparsity):
                    sparsity = csc_matrix(sparsity)
                groups = group_columns(sparsity)
                sparsity = (sparsity, groups)

            def jac_wrapped(t, y, f):
                self.njev += 1
                J, self.jac_factor = num_jac(
                    self.fun_vectorized, t, y, f, self.atol, self.jac_factor, sparsity
                )
                return J

            J = jac_wrapped(t0, y0, self.f)
        elif callable(jac):
            J = jac(t0, y0)
            self.njev = 1
            if issparse(J):
                J = csc_matrix(J)

                def jac_wrapped(t, y, _=None):
                    self.njev += 1
                    return csc_matrix(jac(t, y), dtype=float)

            else:
                J = np.asarray(J, dtype=float)

                def jac_wrapped(t, y, _=None):
                    self.njev += 1
                    return np.asarray(jac(t, y), dtype=float)

            if J.shape != (self.n, self.n):
                raise ValueError(
                    "`jac` is expected to have shape {}, but "
                    "actually has {}.".format((self.n, self.n), J.shape)
                )
        else:
            if issparse(jac):
                J = csc_matrix(jac)
            else:
                J = np.asarray(jac, dtype=float)

            if J.shape != (self.n, self.n):
                raise ValueError(
                    "`jac` is expected to have shape {}, but "
                    "actually has {}.".format((self.n, self.n), J.shape)
                )
            jac_wrapped = None

        return jac_wrapped, J

    def _step_impl(self):
        t = self.t
        y = self.y
        f = self.f
        n = y.size

        max_step = self.max_step
        atol = self.atol
        rtol = self.rtol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            h_abs_old = None
            error_norm_old = None
        elif self.h_abs < min_step:
            h_abs = min_step
            h_abs_old = None
            error_norm_old = None
        else:
            h_abs = self.h_abs
            h_abs_old = self.h_abs_old
            error_norm_old = self.error_norm_old

        J = self.J
        LU_real = self.LU_real
        LU_complex = self.LU_complex

        current_jac = self.current_jac
        jac = self.jac

        rejected = False
        step_accepted = False
        message = None
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            if self.sol is None:
                Z0 = np.zeros((3, y.shape[0]))
            else:
                Z0 = self.sol(t + h * C).T - y

            scale = atol + np.abs(y) * rtol

            converged = False
            while not converged:
                if LU_real is None or LU_complex is None:
                    LU_real = self.lu(MU_REAL / h * self.mass_matrix - J)
                    LU_complex = self.lu(MU_COMPLEX / h * self.mass_matrix - J)

                converged, n_iter, Z, rate = solve_collocation_system(
                    self.fun,
                    t,
                    y,
                    h,
                    Z0,
                    scale,
                    self.newton_tol,
                    LU_real,
                    LU_complex,
                    self.solve_lu,
                    self.mass_matrix,
                )

                if not converged:
                    if current_jac:
                        break

                    J = self.jac(t, y, f)
                    current_jac = True
                    LU_real = None
                    LU_complex = None

            if not converged:
                h_abs *= 0.5
                LU_real = None
                LU_complex = None
                continue

            y_new = y + Z[-1]
            ZE = Z.T.dot(E) / h
            error = self.solve_lu(LU_real, f + ZE)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            # see [1], chapter IV.8, page 127
            error = self.solve_lu(LU_real, f + self.mass_matrix.dot(ZE))
            if self.index_algebraic_vars is not None:
                # correct for the overestimation of the error on
                # algebraic variables, ideally multiply their errors by
                # (h ** (index - 1))
                error *= h**self.index_array

                # we exclude the number of algebraic components, otherwise
                # they artificially lower the error norm
                error_norm = (
                    np.linalg.norm(error / scale) / (n - self.nvars_algebraic) ** 0.5
                )
            else:
                error_norm = norm(error / scale)

            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

            if rejected and error_norm > 1:  # try with stabilised error estimate
                error = self.solve_lu(
                    LU_real, self.fun(t, y + error) + self.mass_matrix.dot(ZE)
                )
                if self.index_algebraic_vars is not None:
                    # correct for the overestimation of the error on
                    # algebraic variables, ideally multiply their errors by
                    # (h ** (index - 1))
                    error *= h**self.index_array

                    # we exclude the number of algebraic components, otherwise
                    # they artificially lower the error norm
                    error_norm = (
                        np.linalg.norm(error / scale)
                        / (n - self.nvars_algebraic) ** 0.5
                    )
                else:
                    error_norm = norm(error / scale)
            if error_norm > 1:
                factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
                h_abs *= max(MIN_FACTOR, safety * factor)

                LU_real = None
                LU_complex = None
                rejected = True
            else:
                step_accepted = True

        # Step is converged and accepted
        recompute_jac = jac is not None and n_iter > 2 and rate > 1e-3

        factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
        factor = min(MAX_FACTOR, safety * factor)

        if not recompute_jac and factor < 1.2:
            factor = 1
        else:
            LU_real = None
            LU_complex = None

        f_new = self.fun(t_new, y_new)
        if recompute_jac:
            J = jac(t_new, y_new, f_new)
            current_jac = True
        elif jac is not None:
            current_jac = False

        self.h_abs_old = self.h_abs
        self.error_norm_old = error_norm

        self.h_abs = h_abs * factor

        self.y_old = y

        self.t = t_new
        self.y = y_new
        self.f = f_new

        self.Z = Z

        self.LU_real = LU_real
        self.LU_complex = LU_complex
        self.current_jac = current_jac
        self.J = J

        self.t_old = t
        self.sol = self._compute_dense_output()

        return step_accepted, message

    def _compute_dense_output(self):
        Q = np.dot(self.Z.T, P)
        return RadauDenseOutput(self.t_old, self.t, self.y_old, Q)

    def _dense_output_impl(self):
        return self.sol


class RadauDenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, Q):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q
        self.order = Q.shape[1] - 1
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        if t.ndim == 0:
            p = np.tile(x, self.order + 1)
            p = np.cumprod(p)
        else:
            p = np.tile(x, (self.order + 1, 1))
            p = np.cumprod(p, axis=0)
        # Here we don't multiply by h, not a mistake.
        y = np.dot(self.Q, p)
        if y.ndim == 2:
            y += self.y_old[:, None]
        else:
            y += self.y_old

        return y


class RadauIIa:
    def __init__(
        self,
        system,
        t1,
        dt,
        rtol=1.0e-3,
        atol=1.0e-3,
        dae_index="GGL",
    ):
        self.system = system
        self.rtol = rtol
        self.atol = atol
        assert dae_index in [2, 3, "GGL"]
        self.dae_index = dae_index

        #######################################################################
        # integration time
        #######################################################################
        t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t_eval = np.arange(t0, self.t1 + self.dt, self.dt)

        self.frac = (t1 - t0) / 101
        self.pbar = tqdm(total=100, leave=True)
        self.i = 0

        #######################################################################
        # dimensions
        #######################################################################
        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = system.nla_g
        self.nla_gamma = system.nla_gamma
        self.ny = self.nq + self.nu + self.nla_g + self.nla_gamma
        if dae_index == "GGL":
            self.ny += self.nla_g

        ######################
        # singular mass matrix
        ######################
        self.mass_matrix = eye(self.ny, format="lil")
        self.mass_matrix[self.nq + self.nu :, self.nq + self.nu :] = 0

        ###########################
        # construct DAE index array
        ###########################
        if dae_index == 2:
            self.index_array = np.concatenate(
                (
                    np.zeros(self.nq + self.nu, dtype=int),
                    2 * np.ones(self.nla_g, dtype=int),
                    2 * np.ones(self.nla_gamma, dtype=int),
                )
            )
        elif dae_index == 3:
            self.index_array = np.concatenate(
                (
                    np.zeros(self.nq + self.nu, dtype=int),
                    3 * np.ones(self.nla_g, dtype=int),
                    2 * np.ones(self.nla_gamma, dtype=int),
                )
            )
        else:
            self.index_array = np.concatenate(
                (
                    np.zeros(self.nq + self.nu, dtype=int),
                    2 * np.ones(self.nla_g, dtype=int),
                    2 * np.ones(self.nla_gamma, dtype=int),
                    3 * np.ones(self.nla_g, dtype=int),
                )
            )

        #######################################################################
        # consistent initial conditions
        #######################################################################
        t0 = system.t0
        q0 = system.q0
        u0 = system.u0

        # solve for consistent initial accelerations and Lagrange mutlipliers
        M0 = self.system.M(t0, q0, scipy_matrix=csr_matrix)
        h0 = self.system.h(t0, q0, u0)
        W_g0 = self.system.W_g(t0, q0, scipy_matrix=csr_matrix)
        W_gamma0 = self.system.W_gamma(t0, q0, scipy_matrix=csr_matrix)
        zeta_g0 = self.system.zeta_g(t0, q0, u0)
        zeta_gamma0 = self.system.zeta_gamma(t0, q0, u0)
        # fmt: off
        A = bmat(
            [
                [        M0, -W_g0, -W_gamma0],
                [    W_g0.T,  None,      None],
                [W_gamma0.T,  None,      None],
            ],
            format="csc",
        )
        b = np.concatenate([
            h0, 
            -zeta_g0, 
            -zeta_gamma0
        ])
        # fmt: on

        u_dot_la_g_la_gamma = spsolve(A, b)
        u_dot0 = u_dot_la_g_la_gamma[: self.nu]
        la_g0 = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
        la_gamma0 = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        g0 = system.g(t0, q0)
        g_dot0 = system.g_dot(t0, q0, u0)
        g_ddot0 = system.g_ddot(t0, q0, u0, u_dot0)
        gamma0 = system.gamma(t0, q0, u0)
        gamma_dot0 = system.gamma_dot(t0, q0, u0, u_dot0)

        assert np.allclose(
            g0, np.zeros(self.nla_g)
        ), "Initial conditions do not fulfill g0!"
        assert np.allclose(
            g_dot0, np.zeros(self.nla_g)
        ), "Initial conditions do not fulfill g_dot0!"
        assert np.allclose(
            g_ddot0, np.zeros(self.nla_g)
        ), "Initial conditions do not fulfill g_ddot0!"
        assert np.allclose(
            gamma0, np.zeros(self.nla_gamma)
        ), "Initial conditions do not fulfill gamma0!"
        assert np.allclose(
            gamma_dot0, np.zeros(self.nla_gamma)
        ), "Initial conditions do not fulfill gamma_dot0!"

        # consistent initial conditions
        self.y0 = np.zeros(self.ny, dtype=float)
        self.y0[: self.nq] = q0
        self.y0[self.nq : self.nq + self.nu] = u0
        self.y0[self.nq + self.nu : self.nq + self.nu + self.nla_g] = la_g0
        self.y0[
            self.nq
            + self.nu
            + self.nla_g : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
        ] = la_gamma0

    def unpack(self, y):
        q = y[: self.nq]
        u = y[self.nq : self.nq + self.nu]
        la_g = y[self.nq + self.nu : self.nq + self.nu + self.nla_g]
        la_gamma = y[
            self.nq
            + self.nu
            + self.nla_g : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
        ]
        mu_g = y[self.nq + self.nu + self.nla_g + self.nla_gamma :]
        return q, u, la_g, la_gamma, mu_g

    def fun(self, t, y):
        # update progress bar
        i1 = int(t // self.frac)
        self.pbar.update(i1 - self.i)
        self.pbar.set_description(f"t: {t:0.2e}s < {self.t1:0.2e}s")
        self.i = i1

        q, u, la_g, la_gamma, mu_g = self.unpack(y)
        q, u = self.system.step_callback(t, q, u)

        M = self.system.M(t, q, scipy_matrix=csc_matrix)
        h = self.system.h(t, q, u)
        self.W_g = self.system.W_g(t, q)
        self.W_gamma = self.system.W_gamma(t, q)

        dy = np.zeros(self.ny, dtype=y.dtype)

        # kinematic equation
        dy[: self.nq] = self.system.q_dot(t, q, u)
        if self.dae_index == "GGL":
            dy[: self.nq] += self.system.g_q(t, q, scipy_matrix=csc_matrix).T @ mu_g

        # equations of motion
        self.rhs = h + self.W_g @ la_g + self.W_gamma @ la_gamma

        # Note: This uses a non constant mass matrix which is not supported by
        # the Radau implementation. However, since we use only simplified Newton
        # iterations, numerical experiments have shown that good convergence
        # is obtained in all tested cases.
        self.mass_matrix[self.nq : self.nq + self.nu, self.nq : self.nq + self.nu] = M
        dy[self.nq : self.nq + self.nu] = self.rhs

        # bilateral constraints on velocity level
        dy[
            self.nq
            + self.nu
            + self.nla_g : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
        ] = self.system.gamma(t, q, u)

        # bilateral constraints on position level
        if self.dae_index == 2:
            dy[self.nq + self.nu : self.nq + self.nu + self.nla_g] = self.system.g_dot(
                t, q, u
            )
        elif self.dae_index == 3:
            dy[self.nq + self.nu : self.nq + self.nu + self.nla_g] = self.system.g(t, q)
        else:
            dy[self.nq + self.nu : self.nq + self.nu + self.nla_g] = self.system.g_dot(
                t, q, u
            )
            dy[self.nq + self.nu + self.nla_g + self.nla_gamma :] = self.system.g(t, q)
        return dy

    def jac(self, t, y):
        q, u, la_g, la_gamma, mu_g = self.unpack(y)

        q_dot_q = self.system.q_dot_q(t, q, u)
        B = self.system.B(t, q)

        rhs_q = (
            self.system.h_q(t, q, u, scipy_matrix=csc_matrix)
            + self.system.Wla_g_q(t, q, la_g, scipy_matrix=csc_matrix)
            + self.system.Wla_gamma_q(t, q, la_gamma, scipy_matrix=csc_matrix)
        )
        rhs_u = self.system.h_u(t, q, u, scipy_matrix=csc_matrix)
        W_g = self.W_g
        W_gamma = self.W_gamma

        gamma_q = self.system.gamma_q(t, q, u)

        # fmt: off
        if self.dae_index == 2:
            g_dot_q = self.system.g_dot_q(t, q, u)
            jac = bmat(
                [
                    [q_dot_q,         B, None,    None],
                    [  rhs_q,     rhs_u,  W_g, W_gamma],
                    [g_dot_q,     W_g.T, None,    None],
                    [gamma_q, W_gamma.T, None,    None],
                ],
                format="csr",
            )
        elif self.dae_index == 3:
            g_q = self.system.g_q(t, q)
            jac = bmat(
                [
                    [q_dot_q,         B, None,    None],
                    [  rhs_q,     rhs_u,  W_g, W_gamma],
                    [    g_q,      None, None,    None],
                    [gamma_q, W_gamma.T, None,    None],
                ],
                format="csr",
            )
        else:
            g_dot_q = self.system.g_dot_q(t, q, u)
            g_q = self.system.g_q(t, q)
            g_q_T_mu_g = self.system.g_q_T_mu_g(t, q, mu_g)
            jac = bmat(
                [
                    [q_dot_q + g_q_T_mu_g,         B, None,    None, g_q.T],
                    [               rhs_q,     rhs_u,  W_g, W_gamma,  None],
                    [             g_dot_q,     W_g.T, None,    None,  None],
                    [             gamma_q, W_gamma.T, None,    None,  None],
                    [                 g_q,      None, None,    None,  None],
                ],
                format="csr",
            )
        # fmt: on

        return jac

        # Note: Uncomment to check analytical Jacobian against numerical one
        # jac_num = approx_derivative(
        #     fun=lambda x: self.fun(t, x), x0=y, method="3-point", rel_step=1e-6
        # )

        # diff = jac - jac_num
        # error = np.linalg.norm(diff)
        # # error = np.linalg.norm(diff[:self.nq])
        # # error = np.linalg.norm(diff[self.nq : self.nq + self.nu])
        # # error = np.linalg.norm(diff[self.nq : self.nq + self.nu, :self.nq])
        # # error = np.linalg.norm(diff[self.nq : self.nq + self.nu, self.nq : self.nq + self.nu])
        # # error = np.linalg.norm(diff[self.nq : self.nq + self.nu, self.nq + self.nu : self.nq + self.nu + self.nla_g])
        # # error = np.linalg.norm(diff[self.nq + self.nu : ])
        # if error > 1.0e-5:
        #     print(f"error jac: {error}")
        # return jac_num

    def solve(self):
        sol = solve_ivp(
            self.fun,
            self.t_eval[[0, -1]],
            self.y0,
            t_eval=self.t_eval,
            method=Radau,
            rtol=self.rtol,
            atol=self.atol,
            jac=self.jac,
            dense_output=True,
            mass_matrix=self.mass_matrix,
            index_array=self.index_array,
        )

        assert sol.success, "solve_ivp failed"

        t = sol.t
        nt = len(t)
        q = np.zeros((nt, self.nq), dtype=float)
        u = np.zeros((nt, self.nu), dtype=float)
        la_g = np.zeros((nt, self.nla_g), dtype=float)
        la_gamma = np.zeros((nt, self.nla_gamma), dtype=float)
        for i in range(nt):
            q[i], u[i], la_g[i], la_gamma[i], _ = self.unpack(sol.y[:, i])

        return Solution(t=t, q=q, u=u, la_g=la_g, la_gamma=la_gamma)
