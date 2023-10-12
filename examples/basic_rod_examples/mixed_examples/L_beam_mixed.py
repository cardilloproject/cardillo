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
    nelements_Lagrangian = 100
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
    

    def angle(t, t0=0.1, angle_max=10 * np.pi):
        if t > t0:
            return angle_max * (t - t0) / (1 - t0)
        else:
            return 0
    A_IK_clamping = lambda t: A_IK_basic(0).z()

    clamping_point = Frame(A_IK=A_IK_clamping)
    # clamping_left1 = RigidConnection(system.origin, cantilever1, frame_ID2=(0,))
    clamping_left1 = RigidConnection(clamping_point, cantilever1, frame_ID2=(0,))
    clamping_left2 = RigidConnection(cantilever1, cantilever2,frame_ID1=(1,), frame_ID2=(0,))

    # assemble the system
    system.add(clamping_point)
    system.add(cantilever1)
    system.add(clamping_left1)
    system.add(cantilever2)
    system.add(clamping_left2)

    if load_type == "moment":
        # moment at cantilever tip
        m = material_model.Fi[2] * 2 * np.pi / length
        M = lambda t: 10 * np.minimum(t, 0.1) * e3 * m
        moment = K_Moment(M, cantilever2, (1,))
        system.add(moment)
    elif load_type == "force":
        # force at the beam's tip
        #f = m / L * 10e-1
        # F = lambda t: 5. * e3
        F = lambda t: 9 * 5.* min(t, 1 / 9) * e3
        # min(t, 1/number load step)
        force = Force(F, cantilever2, frame_ID=(1,))
        system.add(force)
    elif load_type == "dead_load":
        # spatially fixed load at cantilever tip
        P = lambda t: material_model.Fi[2] * (10 * t) / length**2
        F = lambda t: - P(t) * e2
        force = Force(F, cantilever2, (1,))
        system.add(force)
    elif load_type == "dead_load_and_moment":
        # spatially fixed load at cantilever tip
        P = lambda t: material_model.Fi[2] * (10 * t) / length**2
        F = lambda t: - P(t) * e2
        force = Force(F, cantilever, (1,))
        system.add(force)
        # moment at cantilever tip
        M = lambda t: 2.5 * P(t) * e3
        moment = K_Moment(M, cantilever2, (1,))
        system.add(moment)
    elif load_type == "follower_force":
        # spatially fixed load at cantilever tip
        F = lambda t: - 3e-3 * t * e2
        force = K_Force(F, cantilever2, (1,))
        system.add(force)
    else:
        raise NotImplementedError

    system.assemble()

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=2,
        max_iter=30,
        atol=atol,
    )

    # solve nonlinear static equilibrium equations
    sol = solver.solve()

    system_firstphase = system.deepcopy(sol)



    # extract solutions
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol)
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

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q, # nuova configurazione derivata dal linearSolve
        [cantilever1,cantilever2],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever1.nelement + 1,
        repeat=True,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    Y_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    path = Path(__file__)

    if load_type == "force":
        E_pot_total = np.zeros(len(t))

        for i in range(len(t)):
            E_pot_total[i] = cantilever1.E_pot(t[i], q[i])
            E_pot_total[i] += cantilever2.E_pot(t[i], q[i])

        # Kirchhoff rod
        fig2, ax2 = plt.subplots()
        ax2.plot(t, E_pot_total, '-', color='black', label='Cosserat (numeric)')

    if load_type == "force":
        vertical_tip_displacement = np.zeros(len(t))

        for i in range(len(t)):
            vertical_tip_displacement[i] = q[i,cantilever2.qDOF[cantilever2.elDOF_r[nelements-1,-1]]]

        # Kirchhoff rod
        fig2, ax2 = plt.subplots()
        ax2.plot(t, vertical_tip_displacement, '-', color='black', label='Cosserat (numeric)')


    elif load_type == "dead_load":

        centerline_T = np.loadtxt(Path(path.parent, "cantilever_data","dead_load_centerline_T.txt"), delimiter=",", skiprows=1)
        ax1.plot(centerline_T[:, 0], centerline_T[:, 1], np.zeros_like(centerline_T[:, 0]), "-b")
        centerline_EB = np.loadtxt(Path(path.parent, "cantilever_data", "dead_load_centerline_EB.txt"), delimiter=",", skiprows=1)
        ax1.plot(centerline_EB[:, 0], centerline_EB[:, 1], np.zeros_like(centerline_EB[:, 0]), "-g")
        centerline_IEB = np.loadtxt(Path(path.parent, "cantilever_data", "dead_load_centerline_IEB.txt"), delimiter=",", skiprows=1)
        ax1.plot(centerline_IEB[:, 0], centerline_IEB[:, 1], np.zeros_like(centerline_IEB[:, 0]), "-r")


        fig2, ax2 = plt.subplots()
        Deltas_T = np.loadtxt(Path(path.parent, "cantilever_data", "dead_load_Deltas_T.txt"), delimiter=",", skiprows=1)
        ax2.plot(Deltas_T[:, 0], Deltas_T[:, 1],  's', color='blue', label='Timoshenko (numeric)')
        ax2.plot(Deltas_T[:, 0], Deltas_T[:, 2],  's', color='blue')
        
        Deltas_EB = np.loadtxt(Path(path.parent, "cantilever_data", "dead_load_Deltas_EB.txt"), delimiter=",", skiprows=1)
        ax2.plot(Deltas_EB[:, 0], Deltas_EB[:, 1],  'o', color='green', label='Euler-Bernoulli (numeric)')
        ax2.plot(Deltas_EB[:, 0], Deltas_EB[:, 2],  'o', color='green')

        # elastica: analytical solution of Bisshopp, K.E. and Drucker, D.C. "Large deflection of cantilever beams", 1945
        Deltas_IEB_A = np.loadtxt(Path(path.parent, "cantilever_data", "dead_load_Deltas_IEB_A.txt"), delimiter=",", skiprows=1)
        ax2.plot(Deltas_IEB_A[:, 0], Deltas_IEB_A[:, 1],  's', color='red', label='Elastica (analytic)')
        ax2.plot(Deltas_IEB_A[:, 0], Deltas_IEB_A[:, 2],  's', color='red')

        Delta_num = np.zeros(len(t))
        delta_num = np.zeros(len(t))

        for i in range(len(t)):
            r_OP_L = cantilever.nodes(q[i])[:, -1]
            Delta_num[i] = r_OP_L[0] / length
            delta_num[i] = - r_OP_L[1] / length

        # Kirchhoff rod
        ax2.plot(10 * t, delta_num, '-', color='black', label='Cosserat (numeric)')
        ax2.plot(10 * t, Delta_num, '-', color='black')
        

        ax2.set_xlabel("alpha^2")
        ax2.set_ylabel("Delta=x(L)/L, delta=-y(L)/L")
        ax2.legend()
        ax2.grid()

        # plot animation
        ax1.azim = -90
        ax1.elev = 72
        ax1.dist = 8


    plt.show()
    

if __name__ == "__main__":
    # # load: moment at cantilever tip
    cantilever(load_type="force", VTK_export=False)

    # load: dead load at cantilever tip
    # cantilever(load_type="dead_load", rod_hypothesis_penalty="shear_deformable", VTK_export=False)
    # cantilever(load_type="dead_load", rod_hypothesis_penalty="shear_rigid", VTK_export=False)
    # cantilever(load_type="dead_load", rod_hypothesis_penalty="inextensilbe_shear_rigid", VTK_export=False)

    # cantilever(load_type="follower_force", VTK_export=False)
