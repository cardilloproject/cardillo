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



def _bent_45_beam(load_type="moment", rod_hypothesis_penalty="shear_deformable", VTK_export=False):
    
    Rod = CosseratRodPG_R12Mixed
    # Rod = CosseratRodPG_QuatMixed
    # Rod = CosseratRodPG_SE3Mixed

    nelements_Lagrangian = 30
    polynomial_degree = 2

     # number of elements
    if Rod is CosseratRodPG_SE3Mixed:
        nelements = nelements_Lagrangian * polynomial_degree
    else:
        nelements = nelements_Lagrangian


    # geometry of the rod
    length = 100
    width = 1

    # cross section
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, width)
    
    A = cross_section.area
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    atol = 1e-10
   
    # material model
    if rod_hypothesis_penalty == "shear_deformable":
        Ei = np.array([1e7, 1e7/2, 1e7/2])
    elif rod_hypothesis_penalty == "shear_rigid":
        Ei = np.array([1e7, 1e10*1e12, 1e10*1e12])
    elif rod_hypothesis_penalty == "inextensilbe_shear_rigid":
        Ei = np.array([1e6, 1e6, 1e6]) * 1e10

    Ei = np.array([1e7, 1e7/2, 1e7/2])
    Fi = np.array([705000., 833333., 833333.])
    material_model = Simo1986(Ei, Fi)


    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    q0 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        r_OP=r_OP0,
        A_IK=A_IK0,
        mixed=True,
    )

    cantilever = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=False,
        mixed=True
    )

    # generate the constraint on the beam
    A_IK_clamping= lambda t: A_IK_basic(0.).z()
    clamping_point = Frame(A_IK=A_IK_clamping)
    clamping_left = RigidConnection(clamping_point, cantilever, frame_ID2=(0,))

    # frame_left = Frame(r_OP=r_OP0, A_IK=A_IK0)
    # clamping_left = RigidConnection(frame_left, cantilever, r_OP0, frame_ID2=(0,))

    # starting moment to obtain the deformed configuration
    
    m = Fi[2] * np.pi / length * 0.25 
    M = lambda t: t * e3 * m
    moment = K_Moment(M, cantilever, (1,))
        
    # assemble the system
    system = System()
    system.add(cantilever)
    system.add(clamping_point)
    system.add(clamping_left)
    system.add(moment)
    system.assemble()


    solver = Newton(
        system,
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
    [cantilever],
    scale=length,
    scale_di=0.05,
    show=False,
    n_frames=cantilever.nelement + 1,
    repeat=False,
    )

    system_p = system.deepcopy(sol_ph1)

    system.remove(moment)

    F = lambda t: 600 * t * e3
    force = Force(F, cantilever, frame_ID=(1,))
    system.add(force)

    system.assemble()


    solver = Newton(
        system,
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
        q_ph2, # nuova configurazione derivata dal linearSolve
        [cantilever],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever.nelement + 1,
        repeat=True,
    )

    plt.show()

if __name__ == "__main__":
    _bent_45_beam(load_type="moment", VTK_export=False)
