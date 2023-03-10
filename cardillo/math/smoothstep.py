import numpy as np


# special cases of the smooth step function,
# see https://en.wikipedia.org/wiki/Smoothstep#Generalization_to_higher-order_equations
def smoothstep0(x, x_min=0, x_max=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return x


def smoothstep1(x, x_min=0, x_max=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return -2 * x**3 + 3 * x**2


def smoothstep2(x, x_min=0, x_max=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return 6 * x**5 - 15 * x**4 + 10 * x**3


def smoothstep3(x, x_min=0, x_max=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return -20 * x**7 + 70 * x**6 - 84 * x**5 + 35 * x**4


def smoothstep4(x, x_min=0, x_max=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return 70 * x**9 - 315 * x**8 + 540 * x**7 - 420 * x**6 + 126 * x**5


def smoothstep5(x, x_min=0, x_max=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return (
        -252 * x**11
        + 1386 * x**10
        - 3080 * x**9
        + 3465 * x**8
        - 1980 * x**7
        + 462 * x**6
    )


def smoothstep6(x, x_min=0, x_max=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return (
        924 * x**13
        - 6006 * x**12
        + 16380 * x**11
        - 24024 * x**10
        + 20020 * x**9
        - 9009 * x**8
        + 1716 * x**7
    )


# python implementation of the smooth step function for general N,
# see https://en.wikipedia.org/wiki/Smoothstep and https://stackoverflow.com/questions/45165452/how-to-implement-a-smooth-clamp-function-in-python
from scipy.special import comb


def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


if __name__ == "__main__":
    a = 0
    b = 1
    x = np.linspace(-1, 2, 1000)
    import matplotlib.pyplot as plt

    plt.plot(x, smoothstep(x, N=0), label=str(0))
    plt.plot(x, smoothstep0(x), "--k", label="S_0(x)")
    plt.plot(x, smoothstep(x, N=1), label=str(1))
    plt.plot(x, smoothstep1(x), "--k", label="S_1(x)")
    plt.plot(x, smoothstep(x, N=2), label=str(2))
    plt.plot(x, smoothstep2(x), "--k", label="S_2(x)")
    plt.plot(x, smoothstep(x, N=3), label=str(3))
    plt.plot(x, smoothstep3(x), "--k", label="S_3(x)")
    plt.plot(x, smoothstep(x, N=4), label=str(4))
    plt.plot(x, smoothstep4(x), "--k", label="S_4(x)")
    plt.plot(x, smoothstep(x, N=5), label=str(5))
    plt.plot(x, smoothstep5(x), "--k", label="S_5(x)")
    plt.plot(x, smoothstep(x, N=6), label=str(6))
    plt.plot(x, smoothstep6(x), "--k", label="S_6(x)")
    plt.legend()
    plt.show()
