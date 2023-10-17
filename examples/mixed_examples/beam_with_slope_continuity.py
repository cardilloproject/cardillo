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
from cardillo.forces import Force, K_Force, K_Moment

from cardillo.math import e1, e2, e3, A_IK_basic
from cardillo.visualization import Export

from cardillo import System

from math import pi
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

""" Beam with slope continuity example from Romero, I., 
"A comparison of finite elements for nonlinear beams: the absolute nodal coordinate and geometrically
exact formulations", 2008. 
"""

def cantilever(load_type="force", rod_hypothesis_penalty="shear_deformable", VTK_export=False):
    # interpolation of Ansatz/trial functions
    Rod = CosseratRodPG_R12Mixed
    # Rod = CosseratRodPG_QuatMixed
    # Rod = CosseratRodPG_SE3Mixed

    # Ghosh and Roy use a mesh with 5 element for beam
    nelements_Lagrangian = 20
    polynomial_degree = 2
    
    # number of elements
    if Rod is CosseratRodPG_SE3Mixed:
        nelements = nelements_Lagrangian * polynomial_degree
    else:
        nelements = nelements_Lagrangian

    # geometry of the rod
    length = 1
    # slenderness = 1.0e2
    width = 0.1

    # cross section
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, width)
    #     cross_section = CircularCrossSection(line_density, width)

    A = cross_section.area
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment # it is a diagonal matrix of the inertia moment

    atol = 1e-8

    # material model
    if rod_hypothesis_penalty == "shear_deformable":
        Ei = np.array([1e6 * A, 5 * 1e5 * A,  5 * 1e5 * A])
    elif rod_hypothesis_penalty == "shear_rigid":
        Ei = np.array([1e6, 1e10*1e6, 1e10*1e6])
    elif rod_hypothesis_penalty == "inextensilbe_shear_rigid":
        Ei = np.array([1e6, 1e8, 1e8]) * 1e10
            
    Fi = np.array([5 * 1e5 * K_I_rho0[0,0], 1e6 * K_I_rho0[1,1], 1e6 * K_I_rho0[2,2]])

    material_model = Simo1986(Ei, Fi)

    # position and orientation of left point
    r_OP01 = np.zeros(3, dtype=float)
    A_IK01 = np.eye(3, dtype=float)

    r_OP02 = np.zeros(3, dtype=float)
    r_OP02[0]= length
    angolo_rad = np.radians(90)
    A_IK02 = A_IK_basic(angolo_rad).z() 

    r_OP03 = np.zeros(3, dtype=float)
    r_OP03[0]= length
    r_OP03[1]= length
    angolo_rad = np.radians(-90)
    A_IK03 = A_IK_basic(angolo_rad).y() 

    # construct system
    system = System() # è una classe,

    # construct cantilever1 in a straight initial configuration
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

    q03 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        r_OP=r_OP03,
        A_IK=A_IK03,
        mixed=True,
    )
    cantilever3 = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        Q=q03,
        q0=q03,
        polynomial_degree=polynomial_degree,
        reduced_integration=False,
        mixed=True
    )

    clamping_left_c1 = RigidConnection(system.origin, cantilever1, frame_ID2=(0,))
    clamping_left_c2 = RigidConnection(cantilever1, cantilever2, frame_ID1=(1,), frame_ID2=(0,))
    clamping_left_c3 = RigidConnection(cantilever2, cantilever3, frame_ID1=(1,), frame_ID2=(0,))

    F1 = lambda t: - 10 * t * e1
    force_1 = Force(F1, cantilever3, frame_ID=(1,))

    F2 = lambda t: - 10 * t * e3
    force_2 = Force(F2, cantilever3, frame_ID=(1,))
    
    # assemble the system
    system.add(cantilever1)
    system.add(cantilever2)
    system.add(cantilever3)
    system.add(clamping_left_c1)
    system.add(clamping_left_c2)
    system.add(clamping_left_c3)
    system.add(force_1)
    system.add(force_2)

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
        [cantilever1, cantilever2, cantilever3],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever1.nelement + 1,
        repeat=True,
    )

    x_tip_displacement = np.zeros(len(t))
    y_tip_displacement = np.zeros(len(t))
    z_tip_displacement = np.zeros(len(t))

    for i in range(len(t)):
        x_tip_displacement[i] = q[i,cantilever3.qDOF[cantilever3.elDOF_r[nelements-1][polynomial_degree]]] - q03[cantilever3.elDOF_r[nelements-1][polynomial_degree]]
        y_tip_displacement[i] = q[i,cantilever3.qDOF[cantilever3.elDOF_r[nelements-1][polynomial_degree * 2 + 1]]] - q03[cantilever3.elDOF_r[nelements-1][polynomial_degree * 2 + 1]]
        z_tip_displacement[i] = q[i,cantilever3.qDOF[cantilever3.elDOF_r[nelements-1][polynomial_degree * 3 + 2]]] - q03[cantilever3.elDOF_r[nelements-1][polynomial_degree * 3 + 1]]


    fig2, ax2 = plt.subplots()
    ax2.plot(10 * t, x_tip_displacement, '-', color='black', label='Cosserat (numeric)')

    fig3, ax3 = plt.subplots()
    ax3.plot(10 * t, y_tip_displacement, '-', color='black', label='Cosserat (numeric)')

    fig4, ax4 = plt.subplots()
    ax4.plot(10 * t, z_tip_displacement, '-', color='black', label='Cosserat (numeric)')

    
    fig5, ax = plt.subplots()

    ax.plot(10 * t, x_tip_displacement, '-', color='blue', label='X Tip Displacement')
    ax.plot(10 * t, y_tip_displacement, '-', color='red', label='Y Tip Displacement')
    ax.plot(10 * t, z_tip_displacement, '-', color='green', label='Z Tip Displacement')

    # Aggiungi una legenda
    ax.legend()

    # Personalizza il titolo e le label degli assi se necessario
    ax.set_title('Tip Displacements Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Displacement')


    plt.show()

    # # VTK export
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
    cantilever(load_type="force", VTK_export=True)