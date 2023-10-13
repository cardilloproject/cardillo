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

""" L beam example from Jelenic', G. and Crisfield, M. A., "Geometrically exact 3D beam theory: implementation of a strain-invariant finite element for statics and dynamics", 1999. 
https://sci-hub.hkvisa.net/10.1016/s0045-7825(98)00249-7

Example 4: Elbow cantilever subject to prescribed rotation and point load
Case I: tip load + rotation of pi/4 around (for increment) z-axis        
Case II: tip load + rotation of pi/30 (for increment) around x-axis 
Case III: tip moment + rotation of pi/25 (for increment) around x-axis
do 100 revolutions
"""

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
    length = 10
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

    # material model
    if rod_hypothesis_penalty == "shear_deformable":
        Ei = np.array([1e6, 1e6, 1e6])
    elif rod_hypothesis_penalty == "shear_rigid":
        Ei = np.array([1e6,1e10*1e6, 1e10*1e6])
    elif rod_hypothesis_penalty == "inextensilbe_shear_rigid":
        Ei = np.array([1e6, 1e6, 1e6]) * 1e10
            
    Fi = np.array([1e3, 1e3, 1e3])

    """ if load_type == "follower_force":
        E = 1e6
        A = 1
        I = 1
        nu = 0.
        G = E/(2 + 2 * nu)
        # G*=1e1
        length = 10
        Ei = np.array([E*A, G*A, G*A])
        Fi = np.array([G*I, E*I, E*I]) """ # per commentare più righe contemporaneamente 
                                           # puoi usare la shortcut "Alt+Shift+A"
    
    material_model = Simo1986(Ei, Fi)

    # position and orientation of left point
    r_OP01 = np.zeros(3, dtype=float)
    A_IK01 = np.eye(3, dtype=float)

    r_OP02 = np.zeros(3, dtype=float)
    r_OP02[0]= length
    angolo_rad = np.radians(90)
    A_IK02 = A_IK_basic(angolo_rad).z() # rotazione base nello spazio Euclideo (file _rotations.py)
                                         # A_IK_basic è una classe
                                         # z è un metodo della classe, indica una rotazione attorno all'asse z

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

    # A_IK_clamping = lambda t: A_IK_basic(0 / 0.9 * t * 2 * pi * np.maximum(t - 1 / 10, 0)).z()
    # A_IK_clamping = lambda t: A_IK_basic(3/2 * t * pi/2 * np.maximum(t - 1/3, 0)).z() # rotation of pi/2
    # A_IK_clamping = lambda t: A_IK_basic(1/0.9 * t * 4 * pi/2 * np.maximum(t - 1/10, 0)).z() # 2 revolutions
    # A_IK_clamping = lambda t: A_IK_basic(1/0.9 * t * 4 * pi/2 * np.maximum(t - 1/10, 0)).z()
    
    A_IK_clamping_1= lambda t: A_IK_basic(0.).z()
    

    clamping_point_ph1 = Frame(A_IK=A_IK_clamping_1)
    clamping_left_c1_ph1 = RigidConnection(clamping_point_ph1, cantilever1, frame_ID2=(0,))
    clamping_left_c2 = RigidConnection(cantilever1, cantilever2, frame_ID1=(1,), frame_ID2=(0,))

    F = lambda t: 5 * t * e3
    force_ph1 = Force(F, cantilever2, frame_ID=(1,))
    
    # assemble the system
    system.add(cantilever1)
    system.add(cantilever2)
    system.add(clamping_point_ph1)
    system.add(clamping_left_c1_ph1)
    system.add(clamping_left_c2)
    system.add(force_ph1)

    system.assemble()

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=2,
        max_iter=30,
        atol=atol,
    )

    # solve nonlinear static equilibrium equations
    sol_ph1 = solver.solve()

    # extract solutions
    q_ph1 = sol_ph1.q
    nt_ph1 = len(q_ph1)
    t_ph1 = sol_ph1.t[:nt_ph1]

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t_ph1,
        q_ph1, # nuova configurazione derivata dal linearSolve
        [cantilever1, cantilever2],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever1.nelement + 1,
        repeat=False,
    )

    system_ph1 = system.deepcopy(sol_ph1)

    system.remove(clamping_point_ph1)
    system.remove(clamping_left_c1_ph1)
    system.remove(force_ph1)

    n_load_steps = 8
    A_IK_clamping_ph2 = lambda t: A_IK_basic(t * n_load_steps * pi / 4).z()
    clamping_point_ph2 = Frame(A_IK=A_IK_clamping_ph2)
    system.add(clamping_point_ph2)
    clamping_left_c1_ph2 = RigidConnection(clamping_point_ph2, cantilever1, frame_ID2=(0,))
    system.add(clamping_left_c1_ph2)
    F_ph2 = lambda t: F(1)
    force_ph2 = Force(F_ph2, cantilever2, frame_ID=(1,))
    system.add(force_ph2)

    system.assemble()



    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=n_load_steps+1,
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
        [cantilever1, cantilever2],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever1.nelement + 1,
        repeat=True,
    )

    E_pot_total_ph1 = np.zeros(len(t_ph1))
    E_pot_total_ph2 = np.zeros(len(t_ph2))
    vertical_tip_displacement_ph1 = np.zeros(len(t_ph1))
    vertical_tip_displacement_ph2 = np.zeros(len(t_ph2))

    for i in range(len(t_ph1)):
        E_pot_total_ph1[i] = cantilever1.E_pot(t_ph1[i], q_ph1[i])
        E_pot_total_ph1[i] += cantilever2.E_pot(t_ph1[i], q_ph1[i])
        vertical_tip_displacement_ph1[i] = q_ph1[i,cantilever2.qDOF[cantilever2.elDOF_r[nelements-1,-1]]]

    for i in range(len(t_ph2)):
        E_pot_total_ph2[i] = cantilever1.E_pot(t_ph2[i], q_ph2[i])
        E_pot_total_ph2[i] += cantilever2.E_pot(t_ph2[i], q_ph2[i])
        vertical_tip_displacement_ph2[i] = q_ph2[i,cantilever2.qDOF[cantilever2.elDOF_r[nelements-1,-1]]]

    E_pot_total = np.concatenate((E_pot_total_ph1, E_pot_total_ph2))
    vertical_tip_displacement = np.concatenate((vertical_tip_displacement_ph1, vertical_tip_displacement_ph2))
    t = np.linspace(0, 1, len(E_pot_total))
    
    fig3, ax3 = plt.subplots()
    ax3.plot(t, E_pot_total, '-', color='black', label='Cosserat (numeric)')

    fig4, ax4 = plt.subplots()
    ax4.plot(t, vertical_tip_displacement, '-', color='black', label='Cosserat (numeric)')



    plt.show()



    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol_ph1)
        e.export_contr(
            [cantilever1, cantilever2],
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever_curve",
        )
        e.export_contr(
            [cantilever1, cantilever2],
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="cantilever_volume",
        )

    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol_ph2)
        e.export_contr(
            [cantilever1, cantilever2],
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever_curve",
        )
        e.export_contr(
            [cantilever1, cantilever2],
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="cantilever_volume",
        )


if __name__ == "__main__":
    cantilever(load_type="force", VTK_export=False)