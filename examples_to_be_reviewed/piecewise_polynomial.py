import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

if __name__ == "__main__":
    domain = [0, 1]
    window = [0, 1]

    n = 3
    Xi = np.linspace(0, 1, num=n + 1)
    print(f"Xi: {Xi}")

    # TODO: How can we construct higher order shape functions?
    p = 2
    p0 = Polynomial([0], domain=domain, window=window)
    p1 = Polynomial([1], domain=domain, window=window)

    N = []
    # N = np.empty((p, n, n), dtype=Polynomial)
    for k in range(p):
        Nk = []
        for i in range(n):
            Xii = Xi[i]
            Xi1 = Xi[i + 1]
            diff = Xi1 - Xii
            if k == 0:
                Ni = [p0 for _ in range(n)]
                # Ni = np.array([p0 for _ in range(n)])
                Ni[i] = p1
            else:
                omega_ik = Polynomial([-Xii / diff, 1 / diff])
                omega_i1k = Polynomial([-Xii / diff, 1 / diff])
                Ni = []
                for j in range(n - k):
                    # Ni.append(omega_ik * N[k - 1, j] + (1.0 - omega_ik) * N[k - 1, j + 1])
                    Ni.append(
                        omega_ik * N[k - 1][j] + (1.0 - omega_ik) * N[k - 1][j + 1]
                    )
                Ni = np.array(Ni)

            Nk.append(Ni)
            # N[k, i] = Ni
        N.append(Nk)

    def Ni(i):
        def Ni_impl(xi):
            return np.piecewise(
                xi,
                [Xi[j] <= xi <= Xi[j + 1] for j in range(len(N[-1][i]))],
                # N[-1][i]
                [Nij(xi) for Nij in N[-1][i]],
            )

        return Ni_impl

    # NN = [
    #     lambda xi: np.piecewise(
    #         xi,
    #         [Xi[j] <= xi <= Xi[j+1] for j in range(n)],
    #         N[-1][i]
    #         # [*N[-1][i], 0]
    #     )
    #     for i in range(n)
    # ]

    # def Ni(i):
    #     def Ni_impl(xi):
    #         return np.piecewise(
    #             xi,
    #             [Xi[i] <= xi <= Xi[i+1]],
    #             [Polynomial([1], domain=domain, window=window)(xi), 0]
    #         )

    #     # def Ni_impl():
    #     #     return np.where(
    #     #         [Xi[i] <= xi <= Xi[i+1]]
    #     #     )

    #     return Ni_impl

    NN = [Ni(i) for i in range(n)]

    # exit()

    # N0 = lambda xi: np.piecewise(
    #     xi,
    #     [Xi[0] <= xi <= Xi[1]],
    #     # [1, 0]
    #     [Polynomial([1], domain=domain, window=window)(xi), 0]
    # )
    # N1 = lambda xi: np.piecewise(
    #     xi,
    #     [Xi[1] <= xi <= Xi[2]],
    #     [Polynomial([1], domain=domain, window=window)(xi), 0]
    # )

    xis = np.linspace(0, 1, num=1000)
    Ns = [[NN[i](xi) for xi in xis] for i in range(len(NN))]
    # N0s = [N0(xi) for xi in xis]
    # N1s = [N1(xi) for xi in xis]
    fig, ax = plt.subplots()
    for i in range(len(NN)):
        ax.plot(xis, Ns[i], label=f"N{i}")
    # ax.plot(xis, N0s, "-r", label="N0")
    # ax.plot(xis, N1s, "--b", label="N1")
    ax.grid()
    ax.legend()
    plt.show()
