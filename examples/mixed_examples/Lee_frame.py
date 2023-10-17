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
from cardillo.constraints import RigidConnection, Cylindrical, Revolute
from cardillo.solver import Newton
from cardillo.forces import Force, K_Force, K_Moment

from cardillo.math import e1, e2, e3, A_IK_basic
from cardillo.visualization import Export

from cardillo import System

from math import pi
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

''' L beam example from Jelenic', G. and Crisfield, M. A., "Geometrically exact 3D beam theory: implementation of a strain-invariant finite element for statics and dynamics", 1999. 
https://sci-hub.hkvisa.net/10.1016/s0045-7825(98)00249-7
'''

def cantilever(load_type="force", rod_hypothesis_penalty="shear_deformable", VTK_export=False):
    # interpolation of Ansatz/trial functions
    Rod = CosseratRodPG_R12Mixed
    # Rod = CosseratRodPG_QuatMixed
    # Rod = CosseratRodPG_SE3Mixed

    # Ghosh and Roy use a mesh with 5 element for beam
    nelements_Lagrangian = 5
    polynomial_degree = 2
    
    # number of elements
    if Rod is CosseratRodPG_SE3Mixed:
        nelements = nelements_Lagrangian * polynomial_degree
    else:
        nelements = nelements_Lagrangian

    # geometry of the rod
    length = 120
    # slenderness = 1.0e2
    width = 1

    # cross section
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, width)
    #     cross_section = CircularCrossSection(line_density, width)

    A = cross_section.area
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    atol = 1e-8

    Gg = 720 / (2 * (1 + 0.3))

    # material model
    if rod_hypothesis_penalty == "shear_deformable":
        Ei = np.array([720 * 6, Gg * 6, Gg * 6])
    elif rod_hypothesis_penalty == "shear_rigid":
        Ei = np.array([720 * 6, Gg * 1e6, Gg * 1e6])
    elif rod_hypothesis_penalty == "inextensilbe_shear_rigid":
        Ei = np.array([1e6, 1e6, 1e6]) * 1e10
            
    Fi = np.array([720 * 2, 720 * 2, 720 * 2])
    
    material_model = Simo1986(Ei, Fi)

    # position and orientation of left point
    r_OP01 = np.zeros(3, dtype=float)
    A_IK01 = np.eye(3, dtype=float)
    # angolo_rad_0 = np.radians(90)
    # A_IK01 = A_IK_basic(angolo_rad_0).x() 

    r_OP02 = np.zeros(3, dtype=float)
    r_OP02[0]= length
    # angolo_rad = np.radians(-90)
    # A_IK02 = A_IK_basic(angolo_rad).x() 
    angolo_rad = np.radians(90)
    A_IK02 = A_IK_basic(angolo_rad).y() 


    # construct system
    system = System()

    q01 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        r_OP=r_OP01,
        A_IK=A_IK01,
        mixed=True,
    )
    cantilever1 = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        Q=q01,
        q0=q01,
        polynomial_degree=polynomial_degree,
        reduced_integration=False,
        mixed=True
    )

    q02 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        r_OP=r_OP02,
        A_IK=A_IK02,
        mixed=True,
    )
    cantilever2 = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        Q=q02,
        q0=q02,
        polynomial_degree=polynomial_degree,
        reduced_integration=False,
        mixed=True
    )

    # rigid connection between beams
    clamping_c1_c2 = RigidConnection(cantilever1, cantilever2, frame_ID1=(1,), frame_ID2=(0,))

    # hinge on the left of beam 1
    # x-axis rotation
    frame_left_c1= Frame(r_OP01, A_IK01)
    hinge_left_c1= Revolute(frame_left_c1, cantilever1, 1, frame_ID2=(0,))

    # hinge on the right of beam 2
    r_OP02_right = np.zeros(3, dtype=float)
    r_OP02_right[0] = length
    r_OP02_right[2] = -length
    frame_right_c2= Frame(r_OP02_right, A_IK02)
    hinge_right_c2= Revolute(frame_right_c2, cantilever2, 1, frame_ID2=(1,))
    # hinge_right_c2= Revolute(cantilever2, system.origin, 0, frame_ID1=(1,))
    
    
    # concentrated force
    F = lambda t: -1 * t * e1
    force = Force(F, cantilever2, frame_ID=(0.2,))
    
    # assemble the system
    system.add(cantilever1)
    system.add(cantilever2)
    system.add(clamping_c1_c2)
    system.add(frame_left_c1)
    system.add(frame_right_c2)
    system.add(hinge_left_c1)
    system.add(hinge_right_c2)
    system.add(force)

    system.assemble()

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=10,
        max_iter=30,
        atol=atol,
    )

    # solve nonlinear static equilibrium equations
    sol = solver.solve()

    # extract solutions
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q, # nuova configurazione derivata dal linearSolve
        [cantilever1, cantilever2],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever1.nelement + 1,
        repeat=False,
    )

    # E_pot_total_ph1 = np.zeros(len(t_ph1))
    # E_pot_total_ph2 = np.zeros(len(t_ph2))
    # vertical_tip_displacement_ph1 = np.zeros(len(t_ph1))
    # vertical_tip_displacement_ph2 = np.zeros(len(t_ph2))

    # for i in range(len(t_ph1)):
    #     E_pot_total_ph1[i] = cantilever1.E_pot(t_ph1[i], q_ph1[i])
    #     E_pot_total_ph1[i] += cantilever2.E_pot(t_ph1[i], q_ph1[i])
    #     vertical_tip_displacement_ph1[i] = q_ph1[i,cantilever2.qDOF[cantilever2.elDOF_r[nelements-1,-1]]]

    # for i in range(len(t_ph2)):
    #     E_pot_total_ph2[i] = cantilever1.E_pot(t_ph2[i], q_ph2[i])
    #     E_pot_total_ph2[i] += cantilever2.E_pot(t_ph2[i], q_ph2[i])
    #     vertical_tip_displacement_ph2[i] = q_ph2[i,cantilever2.qDOF[cantilever2.elDOF_r[nelements-1,-1]]]

    # E_pot_total = np.concatenate((E_pot_total_ph1, E_pot_total_ph2))
    # vertical_tip_displacement = np.concatenate((vertical_tip_displacement_ph1, vertical_tip_displacement_ph2))
    # t = np.linspace(0, 1, len(E_pot_total))
    
    # fig3, ax3 = plt.subplots()
    # ax3.plot(t, E_pot_total, '-', color='black', label='Cosserat (numeric)')

    # fig4, ax4 = plt.subplots()
    # ax4.plot(t, vertical_tip_displacement, '-', color='black', label='Cosserat (numeric)')

    plt.show()



    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol)
        e.export_contr(
            cantilever1,
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever1_curve",
        )
        e.export_contr(
            cantilever1,
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="cantilever1_volume",
        )
        e.export_contr(
            cantilever2,
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever2_curve",
        )
        e.export_contr(
            cantilever2,
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="cantilever2_volume",
        )


if __name__ == "__main__":
    cantilever(load_type="force", VTK_export=False)