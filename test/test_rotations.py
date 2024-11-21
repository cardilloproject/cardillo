from collections import namedtuple
import numpy as np
import pytest

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

# cases:
#   0: no argument
#   1: True
#   2: False
#   3: no argument, but q gets normalized bevore passed to the actual function
#   4: True, but q gets normalized bevore passed to the actual function
#   5: False, but q gets normalized bevore passed to the actual function

q3 = np.random.rand(3)
q4 = np.random.rand(4)

# should work for all
# TODO: implement version that works for case=2
test_parameters_orthogonality = [
    [Exp_SO3, q3, 0],
    *[[Exp_SO3_quat, q4, i] for i in [0, 1, 3, 4, 5]],
]

# is not supposed to work with no normalize and non-unit quaternions (case=2)
test_parameters_normality = [
    [Exp_SO3, q3, 0],
    *[[Exp_SO3_quat, q4, i] for i in [0, 1, 3, 4, 5]],
]

# all derivatives should work
test_parameters_Exp_SO3_derivative = [
    [Exp_SO3, Exp_SO3_psi, q3, 0],
    *[[Exp_SO3_quat, Exp_SO3_quat_p, q4, i] for i in [0, 1, 2, 3, 4, 5]],
]

# all T_SO3 should work
# TODO: implementation seems wrong for no normalize (case = [2, 5])
test_parameters_T_SO3 = [
    [Exp_SO3, Exp_SO3_psi, T_SO3, q3, 0],
    *[[Exp_SO3_quat, Exp_SO3_quat_p, T_SO3_quat, q4, i] for i in [0, 1, 3, 4]],
]

# all T_SO3_inv should work
# TODO: the implementation is wrong for all kind of non-unit quaternions (cas = [0, 1, 2])
test_parameters_T_SO3_inv = [
    [T_SO3, T_SO3_inv, q3, 0],
    *[[T_SO3_quat, T_SO3_inv_quat, q4, i] for i in [3, 4, 5]],
]


class Helper:
    def __init__(self, fct, case, isDerivative=False):
        assert case >= 0 and case <= 5
        self.fct = fct
        self.case = case
        self.isDerivative = isDerivative

    def P(self, q):
        q2 = q @ q
        return 1 / np.sqrt(q2) * (np.eye(len(q), dtype=q.dtype) - np.outer(q, q) / q2)

    def __call__(self, q):
        if self.isDerivative and self.case >= 3:
            if self.case == 3:
                fct = self.fct(q / np.linalg.norm(q))
                P = self.P(q)
                return np.einsum("ijk,kl->ijl", fct, P)
            elif self.case == 4:
                fct = self.fct(q / np.linalg.norm(q), True)
                P = self.P(q)
                return np.einsum("ijk,kl->ijl", fct, P)
            elif self.case == 5:
                fct = self.fct(q / np.linalg.norm(q), False)
                P = self.P(q)
                return np.einsum("ijk,kl->ijl", fct, P)

        if self.case == 0:
            return self.fct(q)
        elif self.case == 1:
            return self.fct(q, True)
        elif self.case == 2:
            return self.fct(q, False)
        elif self.case == 3:
            return self.fct(q / np.linalg.norm(q))
        elif self.case == 4:
            return self.fct(q / np.linalg.norm(q), True)
        elif self.case == 5:
            return self.fct(q / np.linalg.norm(q), False)


@pytest.mark.parametrize("A_fct, q, case", test_parameters_orthogonality)
def test_orthogonality(A_fct, q, case):
    ex, ey, ez = Helper(A_fct, case)(q).T
    assert np.isclose(ex @ ey, 0), f"case: {case}, e: {ex @ ey}, q: {q}"
    assert np.isclose(ey @ ez, 0), f"case: {case}, e: {ey @ ez}, q: {q}"
    assert np.isclose(ez @ ex, 0), f"case: {case}, e: {ez @ ex}, q: {q}"


@pytest.mark.parametrize("A_fct, q, case", test_parameters_normality)
def test_normality(A_fct, q, case):
    ex, ey, ez = Helper(A_fct, case)(q).T
    assert np.isclose(ex @ ex, 1), f"case: {case}, e: {ex @ ex}, q: {q}"
    assert np.isclose(ey @ ey, 1), f"case: {case}, e: {ey @ ey}, q: {q}"
    assert np.isclose(ez @ ez, 1), f"case: {case}, e: {ez @ ez}, q: {q}"


@pytest.mark.skip(reason="Helper function for derivative.")
def test_derivative(A_fct, A_q_fct, q, case):
    A_q = Helper(A_q_fct, case, True)(q)
    A_q_num = approx_fprime(q, Helper(A_fct, case), method="3-point")
    e = np.linalg.norm(A_q - A_q_num)
    assert np.isclose(e, 0), f"case: {case}, e: {e}, q: {q}"


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("A_fct, A_q_fct, q, case", test_parameters_Exp_SO3_derivative)
def test_Exp_derivative(A_fct, A_q_fct, q, case):
    test_derivative(A_fct, A_q_fct, q, case)


@pytest.mark.parametrize("A_fct, A_q_fct, T_fct, q, case", test_parameters_T_SO3)
# @pytest.mark.skip(reason="no way of currently testing this")
def test_T_SO3(A_fct, A_q_fct, T_fct, q, case):
    # v = B_delta_phi_IB or B_omega_IB or B_kappa_IB
    # dq = delta_q or q_dot or q_xi
    # relation to check:
    # v = T_SO3(q) @ dq
    # ax2skew(v) = A.T @ dA
    dq = np.random.rand(len(q))
    v = Helper(T_fct, case)(q) @ dq

    A = Helper(A_fct, case)(q)
    A_q = Helper(A_q_fct, case)(q)
    dA = np.einsum("ijk, k -> ij", A_q, dq)
    e = np.linalg.norm(ax2skew(v) - A.T @ dA)
    assert np.isclose(e, 0), f"case: {case}, e: {e}, q: {q}"


@pytest.mark.parametrize("T_fct, T_inv_fct, q, case", test_parameters_T_SO3_inv)
def test_T_SO3_inv(T_fct, T_inv_fct, q, case):
    A = Helper(T_fct, case)(q)
    A_inv = Helper(T_inv_fct, case)(q)

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
    e0 = np.linalg.norm(forward - np.eye(A_shape[0]))
    assert np.isclose(e0, 0), f"case: {case}, e: {e0}, q: {q}"
    if A_shape[1] == 3:
        e1 = np.linalg.norm(backward - np.eye(A_shape[1]))
        assert np.isclose(e1, 0), f"case: {case}, e: {e1}, q: {q}"
    else:
        # TODO: what's the motivation behind this?
        e1 = np.linalg.norm(backward - backward.T)
        assert np.isclose(e1, 0), f"case: {case}, e: {e1}, q: {q}"


# TODO: is there a clever way to reuse the test_derivative function from above?
@pytest.mark.parametrize("A_fct, A_q_fct, q, case", test_parameters_Exp_SO3_derivative)
@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
def test_T_derivative(A_fct, A_q_fct, q, case):
    A_q = Helper(A_q_fct, case, True)(q)
    A_q_num = approx_fprime(q, Helper(A_fct, case), method="3-point")
    e = np.linalg.norm(A_q - A_q_num)
    assert np.isclose(e, 0), f"case: {case}, e: {e}, q: {q}"


@pytest.mark.skip(reason="no way of currently testing this")
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

        # # check if A_q is correct
        # test_derivative(p.Exp_SO3, p.Exp_SO3_q, q)

        # # check if T_SO3 is correct
        # test_T_SO3(p.Exp_SO3, p.Exp_SO3_q, p.T_SO3, q)

        # # check if T_SO3_inv is correct
        # test_inverse(p.T_SO3, p.T_SO3_inv, q)

        # # check derivative of T_SO3
        # test_derivative(p.T_SO3, p.T_SO3_q, q)

        # # check derivative of T_SO3_inv
        # test_derivative(p.T_SO3_inv, p.T_SO3_inv_q, q)

        # # check matrix logarithm
        # test_Log_SO3(p.Exp_SO3, p.Log_SO3, q)
        # if p.Log_SO3_A is not None:
        #     test_derivative(p.Log_SO3, p.Log_SO3_A, np.eye(3))

        print(f"Success of {p.name}!")
