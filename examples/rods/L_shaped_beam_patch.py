from cardillo.rods import (
    RectangularCrossSection,
    Simo1986,
    Harsch2021,
    animate_beam,
)

from cardillo.rods.cosseratRod import (
    make_CosseratRod_Quat,
    make_CosseratRod_SE3,
)

from cardillo.discrete import Frame

from cardillo.constraints import RigidConnection
from cardillo.solver import Newton, SolverOptions
from cardillo.forces import Force, K_Moment

from cardillo.math import e2, e3, A_IK_basic, smoothstep2

from cardillo.visualization import Export

from cardillo import System

import numpy as np

from math import pi
import matplotlib.pyplot as plt
from pathlib import Path

""" L-shaped beam patch from:

Harsch, J., Capobianco, G. and Eugster, S. R., "Finite element formulations for constrained spatial nonlinear beam theories", 2021.
https://doi.org/10.1177/10812865211000790
4.4. Beam patches with slope discontinuity

Greco, L., "An iso-parametric G1-conforming finite element for the nonlinear analysis of Kirchhoff rod. Part I: the 2D case"
https://doi.org/10.1007/s00161-020-00861-9
4.2 L-shaped spring
"""


def L_shaped_beam_patch(
    Rod,
    nelements=5,
    polynomial_degree=2,
    n_load_steps=40,
    VTK_export=False,
    reduced_integration=True,
):
    # geometry of the rod
    length = 2 * np.pi

    # cross section properties for visualization purposes
    slenderness = 1.0e2
    width = length / slenderness
    cross_section = RectangularCrossSection(width, width)

    # material properties
    Ei = np.array([5, 1, 1])
    Fi = np.array([0.5, 2, 2])
    material_model = Simo1986(Ei, Fi)

    # construct system
    system = System()

    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(
        nelements, length, polynomial_degree=polynomial_degree
    )
    # construct cantilever
    cantilever1 = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
    )

    r_OP = np.array([length, 0, 0])
    A_IK = A_IK_basic(-pi / 2).z()

    q0 = Rod.straight_configuration(
        nelements, length, polynomial_degree=polynomial_degree, r_OP=r_OP, A_IK=A_IK
    )

    cantilever2 = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
    )

    A_IK = (
        lambda t: A_IK_basic(smoothstep2(t, 0.25, 0.5) * 2 * pi).x()
        @ A_IK_basic(smoothstep2(t, 0.5, 0.75) * 2 * pi).y()
        @ A_IK_basic(smoothstep2(t, 0.75, 1.0) * 2 * pi).z()
    )
    rotating_frame = Frame(A_IK=A_IK)
    clamping_left = RigidConnection(rotating_frame, cantilever1, frame_ID2=(0,))
    rod_rod_clamping = RigidConnection(
        cantilever1, cantilever2, frame_ID1=(1,), frame_ID2=(0,)
    )

    # assemble the system
    system.add(cantilever1)
    system.add(cantilever2)
    system.add(rotating_frame)
    system.add(clamping_left)
    system.add(rod_rod_clamping)

    # moment at cantilever tip
    m = material_model.Fi[2] * 2 * np.pi / length
    M = lambda t: smoothstep2(t, 0, 0.25) * e3 * m
    moment = K_Moment(M, cantilever2, frame_ID=(1,))
    system.add(moment)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=30, newton_atol=1.0e-8),
    )

    # solve nonlinear static equilibrium equations
    sol = solver.solve()

    # extract solutions
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 100, sol)
        rod_list = [cantilever1, cantilever2]
        for i, rod in enumerate(rod_list):
            e.export_contr(
                rod,
                level="centerline + directors",
                num=3 * nelements,
                file_name="rod_curve",
            )
            e.export_contr(
                rod,
                continuity="C0",
                level="volume",
                n_segments=nelements,
                num=3 * nelements,
                file_name="rod_volume",
            )

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [cantilever1, cantilever2],
        scale=1.6 * length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever1.nelement + 1,
        repeat=True,
    )

    path = Path(__file__)

    # add plane with z-direction as normal
    X_z = np.linspace(-1.6 * length, 1.6 * length, num=2)
    Y_z = np.linspace(-1.6 * length, 1.6 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    # add analytic solution to the animation
    nticks = 100
    ts = np.linspace(0, 1, num=nticks)
    xL_analytic = np.zeros(nticks)
    yL_analytic = np.zeros(nticks)
    zL_analytic = np.zeros(nticks)
    xL_analytic[0] = length  # for singulartiy avoidance
    xL_analytic[1:] = length / (2 * pi * ts[1:]) * np.sin(2 * pi * ts[1:])
    yL_analytic[1:] = length / (2 * pi * ts[1:]) * (1 - np.cos(2 * pi * ts[1:]))
    ax1.plot(xL_analytic, yL_analytic, zL_analytic, "-r")
    zL2_analytic = np.zeros(nticks)
    # Formula is wrong in Harsch2021
    # xL2_analytic = xL_analytic + np.cos(2 * pi * ts) * yL_analytic + np.sin(2 * pi * ts) * xL_analytic
    # yL2_analytic = yL_analytic + np.sin(2 * pi * ts) * yL_analytic - np.cos(2 * pi * ts) * xL_analytic
    xL2_analytic = (
        xL_analytic
        + np.cos(2 * pi * ts - pi / 2) * xL_analytic
        - np.sin(2 * pi * ts - pi / 2) * yL_analytic
    )
    yL2_analytic = (
        yL_analytic
        + np.sin(2 * pi * ts - pi / 2) * xL_analytic
        + np.cos(2 * pi * ts - pi / 2) * yL_analytic
    )
    ax1.plot(xL2_analytic, yL2_analytic, zL2_analytic, "-b")

    # plot animation
    ax1.azim = -90
    ax1.elev = 72

    E_pot_total_ref = np.loadtxt(
        Path(path.parent, "_data_L_shaped_beam_patch", "E_pot.csv"),
        delimiter=",",
        skiprows=1,
    )

    E_pot_total = np.zeros(len(t))

    for i, ti in enumerate(t):
        E_pot_total[i] += cantilever1.E_pot(ti, q[i, cantilever1.qDOF])
        E_pot_total[i] += cantilever2.E_pot(ti, q[i, cantilever2.qDOF])

    fig2, ax2 = plt.subplots()
    ax2.plot(t, E_pot_total, "-", color="red", label="Cosserat (numeric)")
    ax2.plot(
        E_pot_total_ref[:, 0],
        E_pot_total_ref[:, 1],
        "-.",
        color="blue",
        label="Director beam (numeric)",
    )

    ax2.set_ylabel("potential Energy E")
    ax2.set_xlabel("t")
    ax2.legend()
    ax2.grid()

    plt.show()


if __name__ == "__main__":
    # Quaternion interpolation:
    L_shaped_beam_patch(Rod=make_CosseratRod_Quat(mixed=True))
    # SE3 interpolation:
    # L_shaped_beam_patch(Rod=make_CosseratRod_SE3(mixed=True))
