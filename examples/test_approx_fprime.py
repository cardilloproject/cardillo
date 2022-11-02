import numpy as np
from cardillo.math import approx_fprime
from scipy.optimize._numdiff import approx_derivative
import matplotlib.pyplot as plt


def mathworks():
    # Complex Step Differentiation example from
    # https://blogs.mathworks.com/cleve/2013/10/14/complex-step-differentiation/
    def f(x):
        return np.exp(x) / (np.sin(x) ** 3 + np.cos(x) ** 3)[0]

    def f_x(x):
        den = np.sin(x) ** 3 + np.cos(x) ** 3
        return np.exp(x) * (
            1.0 / den
            - (3 * np.cos(x) * np.sin(x) ** 2 - 3 * np.sin(x) * np.cos(x) ** 2)
            / den**2
        )

    x0 = np.pi / 4

    num = 500
    eps = np.power(10.0, -np.linspace(1, 16, num=num))
    err = np.nan * np.ones((num, 6))
    for i in range(num):
        err[i, 0] = np.abs(approx_fprime(x0, f, eps=eps[i], method="2-point") - f_x(x0))
        err[i, 1] = np.abs(approx_fprime(x0, f, eps=eps[i], method="3-point") - f_x(x0))
        err[i, 2] = np.abs(approx_fprime(x0, f, eps=eps[i], method="cs") - f_x(x0))
        err[i, 3] = np.abs(
            approx_derivative(f, x0, rel_step=eps[i], abs_step=eps[i], method="2-point")
            - f_x(x0)
        )
        err[i, 4] = np.abs(
            approx_derivative(f, x0, rel_step=eps[i], abs_step=eps[i], method="3-point")
            - f_x(x0)
        )
        err[i, 5] = np.abs(
            approx_derivative(f, x0, rel_step=eps[i], abs_step=eps[i], method="cs")
            - f_x(x0)
        )

    fig, ax = plt.subplots()
    ax.loglog(eps, err[:, 0], label="2-point")
    ax.loglog(eps, err[:, 1], label="3-point")
    ax.loglog(eps, err[:, 2], label="cs")
    ax.loglog(eps, err[:, 3], "--", label="scipy 2-point")
    ax.loglog(eps, err[:, 4], "--", label="scipy 3-point")
    ax.loglog(eps, err[:, 5], "--", label="scipy cs")
    ax.invert_xaxis()
    ax.set_xlabel("eps")
    ax.set_ylabel("error")
    ax.legend()
    ax.grid()
    plt.show()


def quadratic_form():
    A = np.random.rand(2, 2)

    def f(x):
        return x.T @ A @ x

    def f_x(x):
        return x.T @ (A + A.T)

    x0 = np.array([1, 2])

    # f_x_num = approx_fprime(x0, f, method="2-point")
    # f_x_num = approx_fprime(x0, f, method="3-point")
    f_x_num = approx_fprime(x0, f, method="cs", eps=1.0e-12)
    f_x_ = f_x(x0)
    diff = f_x_ - f_x_num
    error = np.linalg.norm(diff)
    print(f"f_x_num: {f_x_num}")
    print(f"f_x_: {f_x_}")
    print(f"error: {error}")


def matrix_valued():
    def f(X):
        return np.trace(X) * X

    def f_x(X):
        n, m = X.shape
        assert n == m

        return np.einsum("ij,kl->ijkl", X, np.eye(n),) + np.einsum(
            "mm,ik,jl->ijkl",
            X,
            np.eye(n),
            np.eye(n),
        )

    x0 = np.random.rand(2, 2)

    # f_x_num = approx_fprime(x0, f, method="2-point")
    # f_x_num = approx_fprime(x0, f, method="3-point")
    f_x_num = approx_fprime(x0, f, method="cs", eps=1.0e-12)
    f_x_ = f_x(x0)
    diff = f_x_ - f_x_num
    error = np.linalg.norm(diff)
    print(f"f_x_num:\n{f_x_num}")
    print(f"f_x_:\n{f_x_}")
    print(f"error: {error}")


if __name__ == "__main__":
    # mathworks()
    # quadratic_form()
    matrix_valued()
