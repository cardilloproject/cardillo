from cardillo.beams import (
    CosseratRodPG_SE3,
    CosseratRodPG_Quat,
    CosseratRodPG_R12,
    RectangularCrossSection,
    CircularCrossSection,
    Simo1986,
    animate_beam,
)

from cardillo.beams.cosseratRodPGMixed import (
    CosseratRodPG_R12Mixed,
    CosseratRodPG_QuatMixed,
    CosseratRodPG_SE3Mixed,
)

from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.solver import Newton
from cardillo.forces import Force, Moment, K_Force, K_Moment

from cardillo.math import e1, e2, e3, A_IK_basic
from cardillo.visualization import Export

from cardillo import System

from math import pi
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path


def _bent_45_beam(
    load_type="moment", rod_hypothesis_penalty="shear_deformable", VTK_export=False
):
    # Rod = CosseratRodPG_R12Mixed
    Rod = CosseratRodPG_QuatMixed
    # Rod = CosseratRodPG_SE3Mixed

    nelements_Lagrangian = 8
    polynomial_degree = 1

    # number of elements
    if Rod is CosseratRodPG_SE3Mixed:
        nelements = nelements_Lagrangian * polynomial_degree
    else:
        nelements = nelements_Lagrangian

    # geometry of the rod
    length = 2 * pi * 100 / 8
    width = 1

    # cross section
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, width)

    A = cross_section.area
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    atol = 1e-10

    # # material model
    # if rod_hypothesis_penalty == "shear_deformable":
    #     Ei = np.array([1e7, 1e7/2, 1e7/2])
    # elif rod_hypothesis_penalty == "shear_rigid":
    #     Ei = np.array([1e7, 1e10*1e12, 1e10*1e12])
    # elif rod_hypothesis_penalty == "inextensilbe_shear_rigid":
    #     Ei = np.array([1e6, 1e6, 1e6]) * 1e10

    Ei = np.array([1.0e7, 5.0e6, 5.0e6])
    Fi = np.array([1.0, 1.0, 1.0]) * 1.0e7 / 12
    material_model = Simo1986(Ei, Fi)

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    q0_ph1 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        r_OP=r_OP0,
        A_IK=A_IK0,
        mixed=True,
    )

    cantilever_ph1 = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        Q=q0_ph1,
        q0=q0_ph1,
        polynomial_degree=polynomial_degree,
        reduced_integration=False,
        mixed=True,
    )

    # generate the constraint on the beam
    A_IK_clamping_ph1 = lambda t: A_IK_basic(0.0).z()
    clamping_point_ph1 = Frame(A_IK=A_IK_clamping_ph1)
    clamping_left_ph1 = RigidConnection(
        clamping_point_ph1, cantilever_ph1, frame_ID2=(0,)
    )

    m = Fi[2] * np.pi / length * 0.25
    M = lambda t: t * e3 * m
    moment = K_Moment(M, cantilever_ph1, (1,))

    # assemble the system
    system_ph1 = System()
    system_ph1.add(cantilever_ph1)
    system_ph1.add(clamping_point_ph1)
    system_ph1.add(clamping_left_ph1)
    system_ph1.add(moment)
    system_ph1.assemble()

    solver = Newton(
        system_ph1,
        n_load_steps=2,
        max_iter=30,
        atol=atol,
    )

    sol_ph1 = solver.solve()
    q_ph1 = sol_ph1.q
    nt_ph1 = len(q_ph1)
    t_ph1 = sol_ph1.t[:nt_ph1]

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t_ph1,
        q_ph1,
        [cantilever_ph1],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever_ph1.nelement + 1,
        repeat=False,
    )

    plt.show()

    """
    second phase
    """

    system_ph2 = System()

    q0_ph2 = q_ph1[-1].copy()
    q0_ph2[cantilever_ph1.elDOF_m] = 0.0
    q0_ph2[cantilever_ph1.elDOF_n] = 0.0

    cantilever_ph2 = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        Q=q0_ph2,
        q0=q0_ph2,
        polynomial_degree=polynomial_degree,
        reduced_integration=False,
        mixed=True,
    )

    system_ph2.q0 = q0_ph2

    # generate the constraint on the beam
    A_IK_clamping_ph2 = lambda t: A_IK_basic(0.0).z()
    clamping_point_ph2 = Frame(A_IK=A_IK_clamping_ph2)
    clamping_left_ph2 = RigidConnection(
        clamping_point_ph2, cantilever_ph2, frame_ID2=(0,)
    )

    F = lambda t: 600 * t * e3
    force = Force(F, cantilever_ph2, frame_ID=(1,))
    system_ph2.add(force)

    system_ph2.add(cantilever_ph2)
    system_ph2.add(clamping_point_ph2)
    system_ph2.add(clamping_left_ph2)
    system_ph2.assemble()

    solver = Newton(
        system_ph2,
        n_load_steps=6,
        max_iter=30,
        atol=atol,
    )

    # solve nonlinear static equilibrium equations
    sol_ph2 = solver.solve()

    # extract solutions
    q_ph2 = sol_ph2.q
    nt_ph2 = len(q_ph2)
    t_ph2 = sol_ph2.t[:nt_ph2]

    # matplotlib visualization
    # construct animation of beam
    fig2, ax2, anim2 = animate_beam(
        t_ph2,
        q_ph2,  # nuova configurazione derivata dal linearSolve
        [cantilever_ph2],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever_ph2.nelement + 1,
        repeat=True,
    )

    plt.show()

    print(q_ph2[-1, cantilever_ph2.qDOF[cantilever_ph2.elDOF_r]][-1, -1])


if __name__ == "__main__":
    _bent_45_beam(load_type="moment", VTK_export=False)
