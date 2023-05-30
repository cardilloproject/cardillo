import numpy as np
from math import pi
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo import System
from cardillo.math import A_IK_basic, axis_angle2quat
from cardillo.discrete import *
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.visualization import Export
from cardillo.solver import (
    MoreauShifted,
    Rattle,
    NonsmoothBackwardEuler,
    NPIRK,
    MoreauShiftedNew,
    MoreauClassical,
)
from cardillo.solver._butcher_tableaus import RadauIIATableau

m = 1.25
r = 0.1
g = 10
path = Path(__file__).parents[2] / "geometry/box/box.stl"

Shape = {
    "ball": (Ball, {"radius": r}),
    "cube": (Box, {"dimensions": (r, r, r)}),
    "box": (Box, {"dimensions": (r / 2, r, 3 * r / 2)}),
    "cylinder": (Cylinder, {"length": r, "radius": r, "axis": 0}),
    "stl": (FromSTL, {"path": path, "K_r_SP": np.zeros(3), "K_Theta_S": None}),  # TODO
}

Parametrization = {
    "quaternion": RigidBodyQuaternion,
    "axis_angle": RigidBodyAxisAngle,
    "euler": RigidBodyEuler,
}

Solver = {
    "NPIRK": (NPIRK, "NPIRK", 5e-2, {"butcher_tableau": RadauIIATableau(2)}),
    "BackwardEuler": (NonsmoothBackwardEuler, "Euler backward", 1e-2, {}),
    "Rattle": (Rattle, "Rattle", 1e-2, {}),
    "MoreauShifted": (MoreauShifted, "MoreauShifted", 2e-2, {"alpha": 0.4}),
    "MoreauShiftedNew": (MoreauShiftedNew, "MoreauShiftedNew", 1e-2, {"alpha": 0.4}),
    "Moreau": (MoreauClassical, "MoreauClassical", 1e-2, {"alpha": 0.4}),
}


def run(
    parametrization: str,
    shape: str,
    solver1: str,
    solver2: str,
    plots=False,
    vtk_export=True,
):
    """Example 10.2 of Capobianco2021.

    References:
    -----------
    Capobianco2021: https://doi.org/10.1002/nme.6801
    """
    ##############################################################
    #               Ridid body
    ##############################################################
    x0 = -0.5
    y0 = 1
    phi0 = 0
    x_dot0 = 0
    y_dot0 = 0
    phi_dot0 = 0

    r_OS0 = np.array([x0, y0, 0], dtype=float)
    vS0 = np.array([x_dot0, y_dot0, 0], dtype=float)
    if parametrization == "quaternion":
        p = axis_angle2quat(np.array((1, 0.5, 0)), phi0)
        q0 = np.concatenate([r_OS0, p])
    else:
        q0 = np.concatenate([r_OS0, np.array([phi0, 0, 0], dtype=float)])
    u0 = np.concatenate([vS0, np.array([0, 0, phi_dot0], dtype=float)])
    Shp, kwargs = Shape[shape]
    RB = Shp(Parametrization[parametrization])(mass=m, q0=q0, u0=u0, **kwargs)
    F_G = Force(lambda t: np.array([0, -g * m, 0]), RB)

    ##############################################################
    #               Contact
    ##############################################################
    alpha = -pi / 4 * 1.1
    e1, e2, e3 = A_IK_basic(alpha).z().T
    frame_left = PlaneFixed(n=e2)
    mu1 = 0.3
    e_N1 = 0.0
    e_F1 = 0.0

    beta = pi / 4
    e1, e2, e3 = A_IK_basic(beta).z().T
    frame_right = PlaneFixed(n=e2)
    mu2 = 0.3
    # mu2 = 0
    e_N2 = 0.5
    # e_N2 = 0.0  # TODO: Remove this
    e_F2 = 0.0

    match shape:
        case "ball":
            contacts = [
                Sphere2Plane(frame_left, RB, r, mu1, e_N=e_N1, e_F=e_F1),
                Sphere2Plane(frame_right, RB, r, mu2, e_N=e_N2, e_F=e_F2),
            ]
        case "cube" | "box":
            contacts = []
            for point in RB.points:
                contacts.append(
                    Sphere2Plane(
                        frame_left, RB, 0, mu1, e_N=e_N1, e_F=e_F1, K_r_SP=point
                    )
                )
                contacts.append(
                    Sphere2Plane(
                        frame_right, RB, 0, mu2, e_N=e_N2, e_F=e_F2, K_r_SP=point
                    )
                )
        case "cylinder":
            # TODO implement contact
            raise NotImplementedError("Circle2PlaneContact not implemented.")
        case "stl":
            raise NotImplementedError("K_Theta_S missing.")
        case _:
            raise (RuntimeError, "Select correct shape.")

    ##############################################################
    #               System
    ##############################################################
    system = System()
    system.add(RB)
    system.add(F_G)
    system.add(*contacts)
    system.add(frame_left)
    system.add(frame_right)

    system.assemble()

    ##############################################################
    #               Solver
    ##############################################################
    t_final = 2

    Solver1, label1, dt1, kwargs1 = Solver[solver1]
    Solver2, label2, dt2, kwargs2 = Solver[solver2]

    sol1 = Solver1(system, t_final, dt1, **kwargs1).solve()
    t = sol1.t
    q = sol1.q
    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    P_N1 = sol1.P_N
    P_F1 = sol1.P_F

    sol2 = Solver2(system, t_final, dt2, **kwargs2).solve()
    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    P_N2 = sol2.P_N
    P_F2 = sol2.P_F

    ##############################################################
    #               Plots
    ##############################################################
    if plots:

        def plot_setup(
            ax,
            title,
            q1,
            q2,
            t1=t1,
            t2=t2,
            format1="-k",
            format2="--r",
            label1=label1,
            label2=label2,
        ):
            ax.set_title(title)
            ax.plot(t1, q1, format1, label=label1)
            ax.plot(t2, q2, format2, label=label2)
            ax.legend()

        fig, ax = plt.subplots(2, 3)

        plot_setup(ax[0, 0], "x(t)", q1[:, 0], q2[:, 0])
        plot_setup(ax[1, 0], "u_x(t)", u1[:, 0], u2[:, 0])
        plot_setup(ax[0, 1], "y(t)", q1[:, 1], q2[:, 1])
        plot_setup(ax[1, 1], "u_y(t)", u1[:, 1], u2[:, 1])
        plot_setup(ax[0, 2], "phi(t)", q1[:, 3], q2[:, 3])
        plot_setup(ax[1, 2], "u_phi(t)", u1[:, 3], u2[:, 3])

        plt.tight_layout()

        fig, ax = plt.subplots(2, 3)

        plot_setup(ax[0, 0], "P_N_left(t)", P_N1[:, 0], P_N2[:, 0])
        plot_setup(ax[1, 0], "P_N_right(t)", P_N1[:, 1], P_N2[:, 1])

        if mu1 > 0:
            plot_setup(ax[0, 1], "P_Fx_left(t)", P_F1[:, 0], P_F2[:, 0])
            plot_setup(ax[0, 2], "P_Fy_left(t)", P_F1[:, 1], P_F2[:, 1])
            plot_setup(ax[1, 1], "P_Fx_right(t)", P_F1[:, 2], P_F2[:, 2])
            plot_setup(ax[1, 2], "P_Fy_right(t)", P_F1[:, 3], P_F2[:, 3])

        plt.tight_layout()

        # matplotlib animation
        if shape == "ball":
            t = t1
            q = q1
            # animate configurations
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.axis("equal")
            ax.set_xlim(-2 * y0, 2 * y0)
            ax.set_ylim(-2 * y0, 2 * y0)

            # prepare data for animation
            frames = len(t)
            target_frames = min(len(t), 200)
            frac = int(frames / target_frames)
            animation_time = 5
            interval = animation_time * 1000 / target_frames

            frames = target_frames
            t = t[::frac]
            q = q[::frac]

            # inclined planes
            K_r_OPs = np.array([[-y0, 0, 0], [y0, 0, 0]]).T
            r_OPs_left = A_IK_basic(alpha).z() @ K_r_OPs
            r_OPs_right = A_IK_basic(beta).z() @ K_r_OPs
            ax.plot(*r_OPs_left[:2], "-k")
            ax.plot(*r_OPs_right[:2], "--k")

            def boundary(object, t, q, n=100):
                phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
                K_r_SP = object.radius * np.vstack(
                    [np.sin(phi), np.cos(phi), np.zeros(n)]
                )
                return (
                    np.repeat(object.r_OP(t, q), n).reshape(3, n)
                    + object.A_IK(t, q) @ K_r_SP
                )

            def create(t, q):
                x_S, y_S, _ = RB.r_OP(t, q)

                A_IK = RB.A_IK(t, q)
                d1 = A_IK[:, 0] * r
                d2 = A_IK[:, 1] * r

                (COM,) = ax.plot([x_S], [y_S], "ok")
                (bdry,) = ax.plot([], [], "-k")
                (d1_,) = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], "-r")
                (d2_,) = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], "-g")
                return COM, bdry, d1_, d2_

            COM, bdry, d1_, d2_ = create(0, q[0])

            def update(t, q, COM, bdry, d1_, d2_):
                x_S, y_S, _ = RB.r_OP(t, q)

                x_bdry, y_bdry, _ = boundary(RB, t, q)

                A_IK = RB.A_IK(t, q)
                d1 = A_IK[:, 0] * r
                d2 = A_IK[:, 1] * r

                COM.set_data([x_S], [y_S])
                bdry.set_data(x_bdry, y_bdry)

                d1_.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
                d2_.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])

                return COM, bdry, d1_, d2_

            def animate(i):
                update(t[i], q[i], COM, bdry, d1_, d2_)

            anim = animation.FuncAnimation(
                fig, animate, frames=frames, interval=interval, blit=False
            )

        plt.show()

    ##############################################################
    #               Export
    ##############################################################
    if vtk_export:
        file_path = Path(__file__)
        path = Path(file_path.parents[1], "sim_data")
        folder = file_path.stem
        e = Export(path, folder, overwrite=True, fps=30, solution=sol1)
        e.export_contr(frame_left, file_name="PlaneLeft")
        e.export_contr(frame_right, file_name="PlaneRight")
        e.export_contr(RB, base_export=True, file_name=f"COM{shape}")
        e.export_contr(RB, base_export=False, file_name=shape.strip("_"))
        e.export_contr(F_G, file_name="F_G")
        e.export_contr(contacts, file_name="Contacts")


if __name__ == "__main__":
    # parametrization = "quaternion"
    # parametrization = "euler"
    parametrization = "axis_angle"
    # shape = "box"
    # shape = "cube"
    # shape = "stl"
    shape = "ball"
    # shape = "cylinder"
    run(parametrization, shape, "Moreau", "MoreauShiftedNew")
