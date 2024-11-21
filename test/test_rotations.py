from collections import namedtuple
import numpy as np

from cardillo.math.approx_fprime import approx_fprime
from cardillo.math import ax2skew
from cardillo.math.rotations import (
    Exp_SO3,
    Exp_SO3_psi,
    Log_SO3,
    Log_SO3_A,
    T_SO3,
    T_SO3_psi,
    T_SO3_inv,
    T_SO3_inv_psi,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    Log_SO3_quat,
    T_SO3_quat,
    T_SO3_quat_P,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
)


def test_orthogonality(A_fct, q):
    ex, ey, ez = A_fct(q).T
    assert np.isclose(ex @ ey, 0), ex @ ey
    assert np.isclose(ey @ ez, 0), ey @ ez
    assert np.isclose(ez @ ex, 0), ez @ ex


def test_normality(A_fct, q):
    ex, ey, ez = A_fct(q).T
    assert np.isclose(ex @ ex, 1), ex @ ex
    assert np.isclose(ey @ ey, 1), ey @ ey
    assert np.isclose(ez @ ez, 1), ez @ ez


def test_derivative(A_fct, A_q_fct, q):
    A_q = A_q_fct(q)
    A_q_num = approx_fprime(q, A_fct, method="3-point")
    assert np.isclose(np.linalg.norm(A_q - A_q_num), 0), f"{q}, \n{A_q}, \n{A_q_num}"


def test_T_SO3(A_fct, A_q_fct, T_fct, q):
    # v = B_delta_phi_IB or B_omega_IB or B_kappa_IB
    # dq = delta_q or q_dot or q_xi
    # relation to check:
    # v = T_SO3(q) @ dq
    # ax2skew(v) = A.T @ dA
    dq = np.random.rand(len(q))
    v = T_fct(q) @ dq

    A = A_fct(q)
    A_q = A_q_fct(q)
    dA = np.einsum("ijk, k -> ij", A_q, dq)
    assert np.isclose(np.linalg.norm(ax2skew(v) - A.T @ dA), 0), f"{v}, \n{A.T @ dA}"


def test_inverse(A_fct, A_inv_fct, q):
    A = A_fct(q)
    A_inv = A_inv_fct(q)

    A_shape = A.shape
    A_inv_shape = A_inv.shape

    assert A_shape[0] == 3
    assert A_shape[0] == A_inv_shape[1]
    assert A_shape[1] == A_inv_shape[0]

    forward = A @ A_inv
    backward = A_inv @ A
    # A = T_SO3:
    #   omega = T_SO3(q) @ q_dot = T_SO3(q) @ T_SO3_inv(q) @ omega
    #                              \________eye(3)_______/
    # as omega is a minimal velocity, this must always be true
    # the other directrion only True if omega.shape == q.shape == 3 (AxisAngle, Euler-Angles, ...), else there are different q_dot leading to the same omega...
    # TODO: but than it has to be symmetrical smh
    assert np.isclose(np.linalg.norm(forward - np.eye(A_shape[0])), 0)
    if A_shape[1] == 3:
        assert np.isclose(np.linalg.norm(backward - np.eye(A_shape[1])), 0)
    else:
        # TODO: what's the motivation behind this?
        assert np.isclose(np.linalg.norm(backward - backward.T), 0)


def test_Log_SO3(Exp_fct, Log_fct, q):
    # generate transformation
    A = Exp_fct(q)
    # extract coordinates (they can be different from q)
    q_ = Log_fct(A)
    # generate matrix with extracted coordinates
    A_ = Exp_fct(q_)

    assert np.isclose(np.linalg.norm(A_ - A), 0)


if __name__ == "__main__":
    Parametrization = namedtuple(
        "Parametrization",
        [
            "name",
            # regular functions
            "Exp_SO3",
            "Log_SO3",
            "T_SO3",
            "T_SO3_inv",
            # derivatives
            "Exp_SO3_q",
            "Log_SO3_A",
            "T_SO3_q",
            "T_SO3_inv_q",
            # dimensions of the parametrization
            "dimension",
        ],
    )
    AxisAngle = Parametrization(
        "AxisAngle",
        Exp_SO3,
        Log_SO3,
        T_SO3,
        T_SO3_inv,
        Exp_SO3_psi,
        Log_SO3_A,
        T_SO3_psi,
        T_SO3_inv_psi,
        3,
    )
    parametrizations = [AxisAngle]

    def make_Quaternion_parametrization(normalize):
        if normalize == None:
            myargs = lambda q: [q]
            name = f"Quaternion w/o arguments"
        else:
            myargs = lambda q: [q, normalize]
            name = f"Quaternion normalize={str(normalize)}"

        p = Parametrization(
            name,
            lambda q: Exp_SO3_quat(*myargs(q)),
            Log_SO3_quat,  # Log returns always a normalized quaternion
            lambda q: T_SO3_quat(*myargs(q)),
            lambda q: T_SO3_inv_quat(*myargs(q)),
            lambda q: Exp_SO3_quat_p(*myargs(q)),
            None,  # no implementation of Log_SO3_A_quat
            lambda q: T_SO3_quat_P(*myargs(q)),
            lambda q: T_SO3_inv_quat_P(*myargs(q)),
            4,
        )
        return p

    parametrizations.extend(
        [make_Quaternion_parametrization(n) for n in [None, True, False]]
    )

    for p in parametrizations:
        q = np.random.rand(p.dimension)
        q = np.arange(1, p.dimension + 1, dtype=float)
        q = q / np.linalg.norm(q)

        # # check if A is orthonormal
        # if p.name != "Quaternion normalize=False" or True:
        #     test_orthogonality(p.Exp_SO3, q)
        #     test_normality(p.Exp_SO3, q)

        # check if A_q is correct
        test_derivative(p.Exp_SO3, p.Exp_SO3_q, q)

        # # check if T_SO3 is correct
        # test_T_SO3(p.Exp_SO3, p.Exp_SO3_q, p.T_SO3, q)

        # # check if T_SO3_inv is correct
        # test_inverse(p.T_SO3, p.T_SO3_inv, q)

        # check derivative of T_SO3
        test_derivative(p.T_SO3, p.T_SO3_q, q)

        # check derivative of T_SO3_inv
        test_derivative(p.T_SO3_inv, p.T_SO3_inv_q, q)

        # check matrix logarithm
        test_Log_SO3(p.Exp_SO3, p.Log_SO3, q)
        if p.Log_SO3_A is not None:
            test_derivative(p.Log_SO3, p.Log_SO3_A, np.eye(3))

        print(f"Success of {p.name}!")
