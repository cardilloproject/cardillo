import numpy as np
from itertools import product
import pytest

from cardillo.math.approx_fprime import approx_fprime
from cardillo.math import skew2ax
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
#   3: no argument, but q gets normalized before passed to the actual function
#   4: True, but q gets normalized before passed to the actual function
#   5: False, but q gets normalized before passed to the actual function

arguments = [
    {},
    {"normalize": True},
    {"normalize": False},
]

functions = [
    lambda x: x,
    lambda x: x / np.linalg.norm(x),
]

# TODO: Make these tests more verbose using something like this:
cases = product(arguments, functions)

# test rotation vectors with magnitude <= pi
q3 = 2 / np.sqrt(3) * np.pi * (np.random.rand(3) - 0.5)
# test quaternions with magnitude <= 2
q4 = 4 * (np.random.rand(4) - 0.5)

# A is used to test Log_SO3
A = Exp_SO3(q3)

# all matrices should be orthogonal
test_parameters_orthogonality = [
    [Exp_SO3, q3, 0],
    *[[Exp_SO3_quat, q4, i] for i in [0, 1, 2, 3, 4, 5]],
]

# is not supposed to work with non-normalized and non-unit quaternions (case=2)
test_parameters_normality = [
    [Exp_SO3, q3, 0],
    *[[Exp_SO3_quat, q4, i] for i in [0, 1, 3, 4, 5]],
]

# all derivatives should work
test_parameters_Exp_SO3_q = [
    [Exp_SO3, Exp_SO3_psi, q3, 0],
    *[[Exp_SO3_quat, Exp_SO3_quat_p, q4, i] for i in [0, 1, 2, 3, 4, 5]],
]

# all T_SO3 should work
test_parameters_T_SO3 = [
    [Exp_SO3, Exp_SO3_psi, T_SO3, q3, 0],
    *[[Exp_SO3_quat, Exp_SO3_quat_p, T_SO3_quat, q4, i] for i in [0, 1, 2, 3, 4, 5]],
]

# all T_SO3_inv should work
test_parameters_T_SO3_inv = [
    [T_SO3, T_SO3_inv, q3, 0],
    *[[T_SO3_quat, T_SO3_inv_quat, q4, i] for i in [0, 1, 2, 3, 4, 5]],
]

# all T_SO3_q should work
test_parameters_T_SO3_q = [
    [T_SO3, T_SO3_psi, q3, 0],
    *[[T_SO3_quat, T_SO3_quat_P, q4, i] for i in [0, 1, 2, 3, 4, 5]],
]

# all T_SO3_inv_q should work
test_parameters_T_SO3_inv_q = [
    [T_SO3_inv, T_SO3_inv_psi, q3, 0],
    *[[T_SO3_inv_quat, T_SO3_inv_quat_P, q4, i] for i in [0, 1, 2, 3, 4, 5]],
]

# all Log_SO3 should work
test_parameters_Log_SO3 = [
    [Exp_SO3, Log_SO3, q3, 0],
    *[[Exp_SO3_quat, Log_SO3_quat, q3, i] for i in [0, 1, 2, 3, 4, 5]],
]

# no implementation of Log_SO3_A
test_parameters_Log_SO3_A = [
    [Log_SO3, Log_SO3_A, q3, 0],
]


class Helper:
    def __init__(self, f, case, isDerivative=False):
        assert 0 <= case <= 5
        self.f = f
        self.case = case
        self.isDerivative = isDerivative

    def P(self, q):
        q2 = q @ q
        return 1 / np.sqrt(q2) * (np.eye(len(q), dtype=q.dtype) - np.outer(q, q) / q2)

    def __call__(self, q):
        # direct function call
        if self.case == 0:
            return self.f(q)
        elif self.case == 1:
            return self.f(q, True)
        elif self.case == 2:
            return self.f(q, False)

        # function call, with normalized q
        q_normalized = q / np.linalg.norm(q)
        if self.case == 3:
            fct = self.f(q_normalized)
        elif self.case == 4:
            fct = self.f(q_normalized, True)
        elif self.case == 5:
            fct = self.f(q_normalized, False)

        # if it is the derivate, we need also the inner derivative,
        # i.e., d/dq (q / norm(q)) = P(q)
        if self.isDerivative:
            P = self.P(q)
            return fct @ P
        else:
            return fct


def derivative_test(A_fct, A_q_fct, q, case):
    A_q = Helper(A_q_fct, case, True)(q)
    A_q_num = approx_fprime(q, Helper(A_fct, case), method="3-point")
    e = np.linalg.norm(A_q - A_q_num)
    assert np.isclose(e, 0, atol=1e-7), f"case: {case}, e: {e:.5e}, q: {q}"


@pytest.mark.parametrize("A_fct, q, case", test_parameters_orthogonality)
def test_orthogonality(A_fct, q, case):
    ex, ey, ez = Helper(A_fct, case)(q).T
    assert np.isclose(ex @ ey, 0), f"case: {case}, e: {ex @ ey:.5e}, q: {q}"
    assert np.isclose(ey @ ez, 0), f"case: {case}, e: {ey @ ez:.5e}, q: {q}"
    assert np.isclose(ez @ ex, 0), f"case: {case}, e: {ez @ ex:.5e}, q: {q}"


@pytest.mark.parametrize("A_fct, q, case", test_parameters_normality)
def test_normality(A_fct, q, case):
    ex, ey, ez = Helper(A_fct, case)(q).T
    assert np.isclose(ex @ ex, 1), f"case: {case}, e: {ex @ ex - 1:.5e}, q: {q}"
    assert np.isclose(ey @ ey, 1), f"case: {case}, e: {ey @ ey - 1:.5e}, q: {q}"
    assert np.isclose(ez @ ez, 1), f"case: {case}, e: {ez @ ez - 1:.5e}, q: {q}"


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("Exp_fct, Exp_q_fct, q, case", test_parameters_Exp_SO3_q)
def test_Exp_SO3_q(Exp_fct, Exp_q_fct, q, case):
    derivative_test(Exp_fct, Exp_q_fct, q, case)


@pytest.mark.parametrize("A_fct, A_q_fct, T_fct, q, case", test_parameters_T_SO3)
def test_T_SO3(A_fct, A_q_fct, T_fct, q, case):
    # v = B_delta_phi_IB or B_omega_IB or B_kappa_IB
    # dq = delta_q or q_dot or q_xi
    # relation to check:
    # v = T_SO3(q) @ dq
    # ax2skew(v) = A.T @ dA
    dq = np.random.rand(len(q))
    T = Helper(T_fct, case)(q)
    v = T @ dq

    A = Helper(A_fct, case)(q)
    A_q = Helper(A_q_fct, case)(q)
    dA = A_q @ dq
    e = np.linalg.norm(v - skew2ax(A.T @ dA))
    assert np.isclose(e, 0), f"case: {case}, e: {e:.5e}, q: {q}"


@pytest.mark.parametrize("T_fct, T_inv_fct, q, case", test_parameters_T_SO3_inv)
def test_T_SO3_inv(T_fct, T_inv_fct, q, case):
    T = Helper(T_fct, case)(q)
    T_inv = Helper(T_inv_fct, case)(q)

    T_shape = T.shape
    T_inv_shape = T_inv.shape

    assert T_shape[0] == 3
    assert T_shape[0] == T_inv_shape[1]
    assert T_shape[1] == T_inv_shape[0]

    # take the (pseudo-inverse)
    T_pinv = np.linalg.pinv(T)
    e = np.linalg.norm(T_pinv - T_inv)
    assert np.isclose(e, 0), f"case: {case}, e: {e:.5e}, q: {q}"


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("T_fct, T_q_fct, q, case", test_parameters_T_SO3_q)
def test_T_SO3_q(T_fct, T_q_fct, q, case):
    derivative_test(T_fct, T_q_fct, q, case)


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("T_inv_fct, T_inv_q_fct, q, case", test_parameters_T_SO3_inv_q)
def test_T_SO3_inv_q(T_inv_fct, T_inv_q_fct, q, case):
    derivative_test(T_inv_fct, T_inv_q_fct, q, case)


@pytest.mark.parametrize("Exp_fct, Log_fct, psi, case", test_parameters_Log_SO3)
def test_Log_SO3(Exp_fct, Log_fct, psi, case):
    # generate transformation always from rotation vector
    A = Exp_SO3(psi)

    # extract coordinates
    # Log function always takes just a matrix -> no need for Helper and case
    q = Log_fct(A)
    if len(q) == len(psi):
        # rotation vector should be equal
        # no complement needed, as 0 <= norm(psi) < 3 < pi (each random component is in [0, 1))
        e0 = np.linalg.norm(q - psi)
    else:
        e0 = np.linalg.norm(q) - 1

    # assert same rotation vector or a unit quaternion
    assert np.isclose(e0, 0), f"case: {case}, e: {e0:.5e}, psi: {psi}"

    # generate matrix with extracted coordinates
    A_ = Helper(Exp_fct, case)(q)

    # assert the same matrix
    e1 = np.linalg.norm(A_ - A)
    assert np.isclose(e1, 0), f"case: {case}, e: {e1:.5e}, q: {q}"


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("Log_fct, Log_fct_q, psi, case", test_parameters_Log_SO3_A)
def test_Log_SO3_A(Log_fct, Log_fct_q, psi, case):
    A = Exp_SO3(psi)
    derivative_test(Log_fct, Log_fct_q, A, case)
