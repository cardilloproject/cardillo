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

# arguments for Quaternions
arguments_quat = [
    {},
    {"normalize": True},
    {"normalize": False},
]

# fmt: off
functions = [
    (
        lambda x: x, 
        lambda x: np.eye(len(x), dtype=x.dtype),
        "x"
    ),
    (
        lambda x: x / np.linalg.norm(x),
        lambda x: 1 / np.sqrt(x @ x) * (np.eye(len(x), dtype=x.dtype) - np.outer(x, x) / (x @ x)),
        "x/||x||"
    ),
]
# fmt: on

# make list to use it multiple times and as np.array for slicing
cases_quat = np.array(list(product(arguments_quat, functions)), dtype=object)
# cases_quat:
#   0: no argument
#   1: no argument, x is normalized
#   2: True
#   3: True, x is normalized
#   4: False
#   5: False, x is normalized

cases_so3 = [
    ({}, functions[0]),
]

# test rotation vectors with magnitude <= pi
q3 = 2 / np.sqrt(3) * np.pi * (np.random.rand(3) - 0.5)
# test quaternions with magnitude <= 2
q4 = 4 * (np.random.rand(4) - 0.5)

# A is used to test Log_SO3
A_test = Exp_SO3(q3)

# all matrices should be orthogonal
test_parameters_orthogonality = [
    [Exp_SO3, q3, cases_so3],
    [Exp_SO3_quat, q4, cases_quat],
]

# is not supposed to work with non-normalized and non-unit quaternions
idx_normality = np.array([0, 1, 2, 3, 5])
test_parameters_normality = [
    [Exp_SO3, q3, cases_so3],
    [Exp_SO3_quat, q4, cases_quat[idx_normality]],
]

# all derivatives should work
test_parameters_Exp_SO3_q = [
    [Exp_SO3, Exp_SO3_psi, q3, cases_so3],
    [Exp_SO3_quat, Exp_SO3_quat_p, q4, cases_quat],
]

# all T_SO3 should work
test_parameters_T_SO3 = [
    [Exp_SO3, Exp_SO3_psi, T_SO3, q3, cases_so3],
    [Exp_SO3_quat, Exp_SO3_quat_p, T_SO3_quat, q4, cases_quat],
]

# all T_SO3_inv should work
test_parameters_T_SO3_inv = [
    [T_SO3, T_SO3_inv, q3, cases_so3],
    [T_SO3_quat, T_SO3_inv_quat, q4, cases_quat],
]

# all T_SO3_q should work
test_parameters_T_SO3_q = [
    [T_SO3, T_SO3_psi, q3, cases_so3],
    [T_SO3_quat, T_SO3_quat_P, q4, cases_quat],
]

# all T_SO3_inv_q should work
test_parameters_T_SO3_inv_q = [
    [T_SO3_inv, T_SO3_inv_psi, q3, cases_so3],
    [T_SO3_inv_quat, T_SO3_inv_quat_P, q4, cases_quat],
]

# all Log_SO3 should work
test_parameters_Log_SO3 = [
    [Exp_SO3, Log_SO3, q3, cases_so3],
    [Exp_SO3_quat, Log_SO3_quat, q3, cases_quat],
]

# no implementation of Log_SO3_A
test_parameters_Log_SO3_A = [
    [Log_SO3, Log_SO3_A, A_test, cases_so3],
]


# wrapper for the function to be called only with the parametrizing coordinates
def wrapper(f, case):
    kwargs = case[0]
    inner_function = case[1][0]
    return lambda x: f(inner_function(x), **kwargs)


# wrapper to compute the total derivative, when the the derivative of the outer function is given
def wrapper_chain_rule(f_deriv, case):
    kwargs = case[0]
    inner_function = case[1][0]
    inner_function_deriv = case[1][1]
    return lambda x: f_deriv(inner_function(x), **kwargs) @ inner_function_deriv(x)


# function to test derivatives
def derivative_test(f, f_x, x, case):
    A_q = wrapper_chain_rule(f_x, case)(x)
    A_q_num = approx_fprime(x, wrapper(f, case), method="3-point")
    e = np.linalg.norm(A_q - A_q_num)
    assert np.isclose(
        e, 0, atol=1e-7
    ), f"Called f({case[1][2]}, {case[0]}), e: {e:.5e}, q: {x}"


@pytest.mark.parametrize("A_fct, q, cases", test_parameters_orthogonality)
def test_orthogonality(A_fct, q, cases):
    for case in cases:
        ex, ey, ez = wrapper(A_fct, case)(q).T
        assert np.isclose(
            ex @ ey, 0
        ), f"Called f({case[1][2]}, {case[0]}), e: {ex @ ey:.5e}, q: {q}"
        assert np.isclose(
            ey @ ez, 0
        ), f"Called f({case[1][2]}, {case[0]}), e: {ey @ ez:.5e}, q: {q}"
        assert np.isclose(
            ez @ ex, 0
        ), f"Called f({case[1][2]}, {case[0]}), e: {ez @ ex:.5e}, q: {q}"


@pytest.mark.parametrize("A_fct, q, cases", test_parameters_normality)
def test_normality(A_fct, q, cases):
    for case in cases:
        ex, ey, ez = wrapper(A_fct, case)(q).T
        assert np.isclose(
            ex @ ex, 1
        ), f"Called f({case[1][2]}, {case[0]}), e: {ex @ ex - 1:.5e}, q: {q}"
        assert np.isclose(
            ey @ ey, 1
        ), f"Called f({case[1][2]}, {case[0]}), e: {ey @ ey - 1:.5e}, q: {q}"
        assert np.isclose(
            ez @ ez, 1
        ), f"Called f({case[1][2]}, {case[0]}), e: {ez @ ez - 1:.5e}, q: {q}"


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("Exp_fct, Exp_q_fct, q, cases", test_parameters_Exp_SO3_q)
def test_Exp_SO3_q(Exp_fct, Exp_q_fct, q, cases):
    for case in cases:
        derivative_test(Exp_fct, Exp_q_fct, q, case)


@pytest.mark.parametrize("A_fct, A_q_fct, T_fct, q, cases", test_parameters_T_SO3)
def test_T_SO3(A_fct, A_q_fct, T_fct, q, cases):
    # v = B_delta_phi_IB or B_omega_IB or B_kappa_IB
    # dq = delta_q or q_dot or q_xi
    # relation to check:
    # v = T_SO3(q) @ dq
    # ax2skew(v) = A.T @ dA
    dq = np.random.rand(len(q))
    for case in cases:
        T = wrapper(T_fct, case)(q)
        v = T @ dq

        A = wrapper(A_fct, case)(q)
        A_q = wrapper(A_q_fct, case)(q)
        dA = A_q @ dq
        e = np.linalg.norm(v - skew2ax(A.T @ dA))
        assert np.isclose(
            e, 0
        ), f"Called f({case[1][2]}, {case[0]}), e: {e:.5e}, q: {q}"


@pytest.mark.parametrize("T_fct, T_inv_fct, q, cases", test_parameters_T_SO3_inv)
def test_T_SO3_inv(T_fct, T_inv_fct, q, cases):
    for case in cases:
        T = wrapper(T_fct, case)(q)
        T_inv = wrapper(T_inv_fct, case)(q)

        T_shape = T.shape
        T_inv_shape = T_inv.shape

        assert T_shape[0] == 3
        assert T_shape[0] == T_inv_shape[1]
        assert T_shape[1] == T_inv_shape[0]

        # take the (pseudo-inverse)
        T_pinv = np.linalg.pinv(T)
        e = np.linalg.norm(T_pinv - T_inv)
        assert np.isclose(
            e, 0
        ), f"Called f({case[1][2]}, {case[0]}), e: {e:.5e}, q: {q}"


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("T_fct, T_q_fct, q, cases", test_parameters_T_SO3_q)
def test_T_SO3_q(T_fct, T_q_fct, q, cases):
    for case in cases:
        derivative_test(T_fct, T_q_fct, q, case)


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize(
    "T_inv_fct, T_inv_q_fct, q, cases", test_parameters_T_SO3_inv_q
)
def test_T_SO3_inv_q(T_inv_fct, T_inv_q_fct, q, cases):
    for case in cases:
        derivative_test(T_inv_fct, T_inv_q_fct, q, case)


@pytest.mark.parametrize("Exp_fct, Log_fct, psi, cases", test_parameters_Log_SO3)
def test_Log_SO3(Exp_fct, Log_fct, psi, cases):
    for case in cases:
        # extract coordinates
        # Log function always takes just a matrix -> no need for wrapper and case
        q = Log_fct(A_test)
        if len(q) == len(psi):
            # rotation vector should be equal
            # no complement needed, as 0 <= norm(psi) < 3 < pi (each random component is in [0, 1))
            e0 = np.linalg.norm(q - psi)
        else:
            e0 = np.linalg.norm(q) - 1

        # assert same rotation vector or a unit quaternion
        assert np.isclose(
            e0, 0
        ), f"Called f({case[1][2]}, {case[0]}), e: {e0:.5e}, psi: {psi}"

        # generate matrix with extracted coordinates
        A_ = wrapper(Exp_fct, case)(q)

        # assert the same matrix
        e1 = np.linalg.norm(A_ - A_test)
        assert np.isclose(
            e1, 0
        ), f"Called f({case[1][2]}, {case[0]}), e: {e1:.5e}, q: {q}"


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("Log_fct, Log_fct_q, A, cases", test_parameters_Log_SO3_A)
def test_Log_SO3_A(Log_fct, Log_fct_q, A, cases):
    for case in cases:
        derivative_test(Log_fct, Log_fct_q, A, case)


if __name__ == "__main__":
    test_Exp_SO3_q(Exp_SO3, Exp_SO3_psi, q3, cases_so3)
    test_Exp_SO3_q(Exp_SO3_quat, Exp_SO3_quat_p, q4, cases_quat)

    test_Exp_SO3_q(T_SO3, T_SO3_psi, q3, cases_so3)
    test_Exp_SO3_q(T_SO3_quat, T_SO3_quat_P, q4, cases_quat)
