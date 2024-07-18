import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

from cardillo.math import A_IB_basic
from cardillo import System
from cardillo.discrete import Frame, PointMass
from cardillo.constraints import FixedDistance
from cardillo.forces import Force
from cardillo.solver import (
    ScipyIVP,
    ScipyDAE,
    Moreau,
    BackwardEuler,
    Rattle,
)


class Mathematical_pendulum3D_excited:
    def __init__(self, m, L, e, e_t, e_tt):
        self.m = m
        self.L = L
        self.e = e
        self.e_t = e_t
        self.e_tt = e_tt

    def r_OP(self, t, q):
        L = self.L
        return A_IB_basic(q[0]).y @ np.array(
            [L * np.cos(q[1]), L * np.sin(q[1]) + self.e(t), 0]
        )

    def v_P(self, t, q, u):
        L = self.L
        B_v_C = np.array(
            [
                -L * np.sin(q[1]) * u[1],
                L * np.cos(q[1]) * u[1] + self.e_t(t),
                L * np.cos(q[1]) * u[0],
            ]
        )
        return A_IB_basic(q[0]).y @ B_v_C

    def eqm(self, t, x):
        dx = np.zeros(4)
        alpha = x[0]
        beta = x[1]
        alpha_dot = x[2]
        beta_dot = x[3]

        g = 9.81
        mL = self.m * self.L
        mL2 = self.m * self.L**2

        M = np.diag(np.array([mL2 * np.cos(beta) ** 2, mL2]))

        h = np.array(
            [
                2 * mL2 * np.cos(beta) * np.sin(beta) * alpha_dot * beta_dot,
                -mL * np.cos(beta) * self.e_tt(t)
                - mL2 * np.cos(beta) * np.sin(beta) * alpha_dot**2
                - mL * np.cos(beta) * g,
            ]
        )

        dx[:2] = x[2:]
        dx[2:] = np.linalg.solve(M, h)
        return dx


solvers_and_kwargs = [
    (ScipyIVP, {}),
    (ScipyDAE, {}),
    (Moreau, {}),
    (BackwardEuler, {}),
    (Rattle, {}),
]


@pytest.mark.parametrize("Solver, solver_kwargs", solvers_and_kwargs)
def test_fixed_distance(Solver, solver_kwargs, show=False):
    t0 = 0
    t1 = 1
    dt = 1e-3

    m = 0.1
    L = 0.2
    g = 9.81

    omega = 10
    A = L / 10

    e = lambda t: A * np.sin(omega * t)
    e_t = lambda t: A * omega * np.cos(omega * t)
    e_tt = lambda t: -A * omega * omega * np.sin(omega * t)

    # e = lambda t: A * t
    # e_t = lambda t: A
    # e_tt = lambda t: 0

    # e = lambda t: 0
    # e_t = lambda t: 0
    # e_tt = lambda t: 0

    r_OP = lambda t: np.array([0, e(t), 0])
    v_P = lambda t: np.array([0, e_t(t), 0])
    a_P = lambda t: np.array([0, e_tt(t), 0])

    # reference solution
    pendulum = Mathematical_pendulum3D_excited(m, L, e, e_t, e_tt)

    alpha0 = 1
    alpha_dot0 = 0
    beta0 = 0
    beta_dot0 = 0

    x0 = np.array([alpha0, beta0, alpha_dot0, beta_dot0])
    ref = solve_ivp(
        pendulum.eqm,
        [t0, t1],
        x0,
        method="RK45",
        t_eval=np.arange(t0, t1 + dt, dt),
        rtol=1e-8,
        atol=1e-12,
    )
    t_ref = ref.t
    q_ref = ref.y[:2].T

    # solutions with cardillo models
    r_OC0 = pendulum.r_OP(t0, np.array([alpha0, beta0]))
    v_C0 = pendulum.v_P(
        t0, np.array([alpha0, beta0]), np.array([alpha_dot0, beta_dot0])
    )

    PM = PointMass(m, q0=r_OC0, u0=v_C0)

    origin = Frame(r_OP, r_OP_t=v_P, r_OP_tt=a_P)
    # joint = FixedDistance(origin, PM)
    joint = FixedDistance(PM, origin)

    system = System()
    system.add(origin)
    system.add(PM)
    system.add(joint)
    system.add(Force(lambda t: np.array([0, -g * m, 0]), PM))

    system.assemble()

    solver = Solver(system, t1, dt, **solver_kwargs)

    sol = solver.solve()
    t = sol.t
    q = sol.q

    if show:
        x_ref_ = []
        y_ref_ = []
        for i, ti in enumerate(t_ref):
            x_ref_.append(pendulum.r_OP(ti, q_ref[i])[0])
            y_ref_.append(pendulum.r_OP(ti, q_ref[i])[1])

        x_ = []
        y_ = []
        for i, ti in enumerate(t):
            x_.append(PM.r_OP(ti, q[i])[0])
            y_.append(PM.r_OP(ti, q[i])[1])

        plt.plot(x_ref_, y_ref_, "-r")
        plt.plot(x_, y_, "--gx")
        scale_ = 1.2 * L
        plt.xlim(-scale_, scale_)
        plt.ylim(-scale_, scale_)
        plt.axis("equal")
        plt.xlabel("x_S [m]")
        plt.ylabel("y_S [m]")
        plt.show()

    if show:
        # animate configurations
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("z [m]")
        ax.set_ylabel("x [m]")
        ax.set_zlabel("y [m]")
        scale = L
        ax.set_xlim3d(left=-scale, right=scale)
        ax.set_ylim3d(bottom=-scale, top=scale)
        ax.set_zlim3d(bottom=-scale, top=scale)

        # prepare data for animation
        frames = len(t)
        target_frames = min(frames, 100)
        frac = int(frames / target_frames)
        animation_time = t1 - t0
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = q[::frac]

        def create(t, q):
            x_0, y_0, z_0 = r_OP(t)
            x_S, y_S, z_S = PM.r_OP(t, q)

            (COM,) = ax.plot([z_0, z_S], [x_0, x_S], [y_0, y_S], "-ok")

            return COM

        COM = create(0, q[0])

        def update(t, q, COM):
            x_0, y_0, z_0 = r_OP(t)
            x_S, y_S, z_S = PM.r_OP(t, q)

            COM.set_data([z_0, z_S], [x_0, x_S])
            COM.set_3d_properties([y_0, y_S])

            return COM

        def animate(i):
            update(t[i], q[i], COM)

        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=False
        )
        plt.show()


if __name__ == "__main__":
    test_fixed_distance(ScipyIVP, {}, show=True)
    test_fixed_distance(ScipyDAE, {}, show=True)
    test_fixed_distance(Moreau, {}, show=True)
    test_fixed_distance(BackwardEuler, {}, show=True)
