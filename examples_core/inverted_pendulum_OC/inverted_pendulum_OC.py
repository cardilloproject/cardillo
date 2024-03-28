from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import minimize

from cardillo import System


class ActuatedPendulum:
    """Model of an acutuated pendulum described with angle (phi) as minimal
    coordinate. The actuation (tau) is a moment acting on the pendulum."""

    def __init__(self, l, m, tau, g=9.81, phi0=0, phi_dot0=0):
        self.l = l
        self.m = m
        self.g = g
        if not callable(tau):
            self.tau = lambda t: tau
        else:
            self.tau = tau
        self.q0 = np.array([phi0])
        self.u0 = np.array([phi_dot0])
        self.nq = 1
        self.nu = 1
        self.nla_tau = 1
        self.ntau = 1

        self._M = (m * (l**2) * (1 / 4 + 1 / 12)) * np.eye(1)

    def q_dot(self, t, q, u):
        return u

    def M(self, t, q):
        return self._M

    def h(self, t, q, u):
        return -self.m * self.g * self.l / 2 * np.sin(q)

    def W_tau(self, t, q):
        return np.ones(1)

    def la_tau(self, t, q, u):
        return self.tau(t)


def optimal_control_pendulum(l, m, tN, phiN, phi_dotN, dt, g=9.81, phi0=0, phi_dot0=0):
    r"""Solves the optimal control problem using the direct approach.

    Parameters
    ----------
    l : float
        Length of penulum
    m : float
        Mass of pendulum
    tN : float
        Final time
    phiN : float
        Final angle
    phi_dotN : float
        Final angular velocity
    dt : float
        Time step
    g : float
        Gravitational acceleration
    phi0 : float
        Initial angle
    phi_dot0 : float
        Initial angular velocity

    Returns
    -------
    system : cardillo.System
        System with optimal actuation set as feed-forward control
    t : np.ndarray
        Time instances
    q : np.ndarray
        Optimal trajectory
    tau : np.ndarray
        Optimal control input (moment)

    """
    system = System()
    tau0 = 0
    system.add(ActuatedPendulum(l, m, tau0, g=g, phi0=phi0, phi_dot0=phi_dot0))
    system.assemble()

    t = np.arange(system.t0, tN, dt)
    nt = len(t)
    phi_initial_guess = np.zeros_like(t)
    tau_inital_guess = np.zeros(nt - 2)
    x0 = np.concatenate((phi_initial_guess[2:-2], tau_inital_guess))
    phi1 = phi0 + dt * phi_dot0
    phiN1 = phiN - dt * phi_dotN

    def constraints(x):
        q = np.concatenate(
            [np.array([phi0, phi1]), x[: nt - 4], np.array([phiN1, phiN])]
        ).reshape(nt, 1)
        tau = x[nt - 4 :].reshape(nt - 2, 1)
        system.set_tau(interp1d(t[1:-1], tau, axis=0, fill_value="extrapolate"))
        return np.concatenate(
            [
                system.M(t[i], q[i]) @ (q[i + 1] - 2 * q[i] + q[i - 1]) / dt
                - dt * system.h(t[i], q[i], np.zeros_like(q[i]))
                - dt
                * system.W_tau(t[i], q[i])
                @ system.la_tau(t[i], q[i], np.zeros_like(q[i]))
                for i in np.arange(1, nt - 1)
            ]
        )

    def cost(x):
        tau = x[nt - 4 :].reshape(nt - 2, 1)
        return np.sum(tau**2) / 2

    # choose optimizer and options
    method = "SLSQP"
    options = {"disp": True, "iprint": 2}
    opt_sol = minimize(
        cost,
        x0,
        method=method,
        constraints={"type": "eq", "fun": constraints},
        options=options,
    )
    if opt_sol.success:
        x_opt = opt_sol.x
        print(opt_sol.message)
    else:
        raise ValueError("optimization not converged.")

    q = np.concatenate(
        [np.array([phi0, phi1]), x_opt[: nt - 4], np.array([phiN1, phiN])]
    ).reshape(nt, 1)
    tau = x_opt[nt - 4 :].reshape(nt - 2, 1)
    system.set_tau(interp1d(t[1:-1], tau, axis=0, fill_value=0.0, bounds_error=False))

    return system, t, q, tau


if __name__ == "__main__":
    ############
    # parameters
    ############
    l = 1  # pendulum length
    m = 1  # pendulum mass
    g = 10  # gravitational acceleration
    phi0 = 0  # initial angle
    phi_dot0 = 0  # initial angular velocity

    tN = 1  # final time
    phiN = np.pi  # final angle
    phi_dotN = 0  # final angular velocity
    dt = 2e-2  # time step

    # solve optimal control problem using the direct method
    system, t, q, tau = optimal_control_pendulum(
        l, m, tN, phiN, phi_dotN, dt, g=g, phi0=phi0, phi_dot0=phi_dot0
    )

    ##############
    # plot results
    ##############
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 7))

    ax[0].plot(t, q)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("phi")
    ax[0].grid()

    ax[1].plot(t[1:-1], tau)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("tau")
    ax[1].grid()

    plt.tight_layout()
    plt.show()

    ###########
    # animation
    ###########

    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    width = 1.5 * l
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)
    ax.axis("equal")

    # prepare data for animation
    frames = len(t)
    target_frames = min(len(t), 200)
    frac = int(frames / target_frames)
    animation_time = 5
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t = t[::frac]
    q = q[::frac].flatten()

    (line,) = ax.plot([], [], "-ok")
    angles = np.linspace(0, 2 * np.pi, num=100, endpoint=True)
    ax.plot(np.cos(angles), np.sin(angles), "--k")

    def update(t, q, line):
        x, y = l * np.sin(q), -l * np.cos(q)
        line.set_data([0, x], [0, y])
        return (line,)

    def animate(i):
        update(t[i], q[i], line)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()
