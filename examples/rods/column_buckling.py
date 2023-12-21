from cardillo.math import e1, e2
from cardillo.rods import (
    RectangularCrossSection,
    Simo1986,
)
from cardillo.discrete import Frame
from cardillo.constraints import Revolute
from cardillo.constraints._base import ProjectedPositionOrientationBase
from cardillo.rods import animate_beam
from cardillo.math import smoothstep1, smoothstep0

from cardillo.forces import Force
from cardillo import System
from cardillo.solver import Newton, Riks, SolverOptions
from cardillo.visualization import Export
from cardillo.rods.cosseratRod import make_CosseratRod_Quat

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def column_buckling(
    Rod,
    nelements=10,
    polynomial_degree=2,
    n_load_steps=10,
    VTK_export=False,
    reduced_integration=True,
):
    # geometry of the column
    length = 1  

    # cross section properties for visualization purposes
    width = 0.005
    height = 0.02
    cross_section = RectangularCrossSection(width, height)
    A = cross_section.area
    I1, I2, I3 = np.diag(cross_section.second_moment)

    # material model
    EE = 1.0
    GG = 0.5

    Ei = np.array([EE * A, GG * A, GG * A])
    Fi = np.array([GG * I1, EE * I2, EE * I3])

    material_model = Simo1986(Ei, Fi)

    # construct system
    system = System()

    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(
        nelements, length, polynomial_degree=polynomial_degree
    )
    # construct cantilever
    column = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
    )


    # left and right joint
    joint0 = Revolute(system.origin, column, axis=2, frame_ID2=(0,))

    r_OP1 = np.array([length, 0, 0], dtype=float)
    frame1 = Frame(r_OP1, np.eye(3))
    joint1 = ProjectedPositionOrientationBase(
        frame1,
        column,
        constrained_axes_translation=[1, 2],
        projection_pairs_rotation=[(1, 2), (2, 0)],
        frame_ID2=(1,),
    )

    # # force at the beam's tip
    # f_crit = np.pi**2 * EE * I3 / length**2
    # F1 = lambda t: -t * e1 * f_crit * 1.5
    # force1 = Force(F1, column, frame_ID=(1,))

    # F12 = lambda t: smoothstep1(1.5 * t, 0.999, 1.001) * e2 * f_crit * 5e-3
    # force12 = Force(F12, column, frame_ID=(0.5,))


    # force at the beam's tip
    overload_factor = 0.3 + 0.01
    f_crit = np.pi**2 * EE * I3 / length**2
    def F1(t):
        if t > 0:
            F = -(0.95 + t * overload_factor) * e1 * f_crit
        else:
            F = e1 * 0
        return F
    force1 = Force(F1, column, frame_ID=(1,))

    F12 = lambda t: smoothstep1(t * overload_factor, 0.049, 0.05) * e2 * f_crit * 5e-3
    force12 = Force(F12, column, frame_ID=(0.5,))

    # assemble the system
    system.add(column)
    system.add(joint0)
    system.add(frame1)
    system.add(joint1)
    system.add(force1)
    system.add(force12)
    system.assemble(
        options=SolverOptions(compute_consistent_initial_conditions=False)
    )
    
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=30, newton_atol=1.0e-12),
    )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol)
        e.export_contr(
            column,
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever_curve",
        )
        e.export_contr(
            column,
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="cantilever_volume",
        )

    #########################
    # visualize displacements
    #########################
    fig, ax = plt.subplots()

    r_OPs = np.array([column.centerline(qi, num=3) for qi in sol.q])
    forces = np.array([F1(ti) for ti in sol.t]) / f_crit

    ax.plot([-1, -1], [0, 0.6 * length], "-k", label="f_crit")
    # ax.plot(forces[:, 0], r_OPs[:, 0, 1], "-r", label="x(t, 0.5)")
    ax.plot(forces[:, 0], r_OPs[:, 1, 1], "-g", label="y(t, 0.5)")
    # ax.plot(forces[:, 0], r_OPs[:, 2, 1], "--b", label="z(t, 0.5)")
    ax.grid()
    ax.legend()

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [column],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=column.nelement + 1,
        repeat=True,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    Y_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    plt.show()

if __name__ == "__main__":
    column_buckling(Rod=make_CosseratRod_Quat(mixed=True, constraints=[0, 1, 2]),
        VTK_export=False,
        n_load_steps=20,)
