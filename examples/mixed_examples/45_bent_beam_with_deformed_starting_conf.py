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

from cardillo.math import e1, e2, e3, A_IK_basic, norm, cross3, Log_SO3_quat

from cardillo.visualization import Export

from cardillo import System

from math import pi
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path


def _bent_45_beam(load_type="moment", rod_hypothesis_penalty="shear_deformable", VTK_export=False):
    
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

    Ei = np.array([1.e7, 5.e6, 5.e6])
    Fi = np.array([1., 1., 1.])*1.e7/12
    material_model = Simo1986(Ei, Fi)

    R = 100
    angle = pi/4

    curve = lambda xi: np.array([R - R * np.cos(xi), R * np.sin(xi), 0])
    dcurve = lambda xi: np.array([R * np.sin(xi), R * np.cos(xi), 0])
    ddcurve = lambda xi: np.array([R * np.cos(xi), -R * np.sin(xi), 0])

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    q0 = Rod.deformed_configuration(
        nelements,
        curve,
        dcurve,
        ddcurve,
        angle,
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
    
    F = lambda t: 600 * t * e3
    force = Force(F, cantilever, frame_ID=(1,))
    
    # assemble the system
    system = System()
    system.add(cantilever)
    system.add(clamping_point)
    system.add(clamping_left)
    system.add(force)
    system.assemble()


    solver = Newton(
        system,
        n_load_steps=6,
        max_iter=30,
        atol=atol,
    )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
    t,
    q, 
    [cantilever],
    scale=length,
    scale_di=0.05,
    show=False,
    n_frames=cantilever.nelement + 1,
    repeat=True,
    )

    plt.show()

   
    print(q[-1, cantilever.elDOF_r][-1, -1])

if __name__ == "__main__":
    _bent_45_beam(load_type="moment", VTK_export=False)
 