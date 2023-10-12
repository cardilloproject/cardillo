from cardillo.beams import (
    CosseratRodPG_R12,
    CosseratRodPG_Quat,
    K_R12_PetrovGalerkin_Quaternion,
    K_SE3_PetrovGalerkin_Quaternion,
    K_PetrovGalerkinQuaternionInterpolation,
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

""" Cantilever beam examples from Harsch, J. and Eugster, S. R., "Finite element analysis of planar nonlinear classical beam theories", 2020. 
https://simonreugster.com/literature/book_contributions/Harsch2020a.pdf
load_type = "moment":         6.1 Pure bending of a cantilever beam
load_type = "dead_load":      6.2 Cantilever beam subjected to constant end load
load_type = "follower_force": 6.3 Cantilever beam subject to follower end load
"""

def cantilever(load_type="moment", rod_hypothesis_penalty="shear_deformable", VTK_export=False):
    # interpolation of Ansatz/trial functions
    Rod = CosseratRodPG_R12Mixed
    # Rod = CosseratRodPG_QuatMixed
    # Rod = CosseratRodPG_SE3Mixed
    # nelements_Lagrangian = 5
    nelements_Lagrangian = 5
    polynomial_degree = 2
    
    # number of elements
    if Rod is CosseratRodPG_SE3Mixed:
        nelements = nelements_Lagrangian * polynomial_degree
    else:
        nelements = nelements_Lagrangian

    # geometry of the rod
    length = 2 * np.pi
    slenderness = 1.0e2
    width = length / slenderness

    # cross section
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, width)

    A = cross_section.area
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    atol = 1e-8

    # material model
    if rod_hypothesis_penalty == "shear_deformable":
        Ei = np.array([5, 1, 1])
    elif rod_hypothesis_penalty == "shear_rigid":
        Ei = np.array([5, 1e3, 1e3])
    elif rod_hypothesis_penalty == "inextensilbe_shear_rigid":
        Ei = np.array([5, 1, 1]) * 1e4
            
    Fi = np.array([0.5, 2, 2])

    if load_type == "follower_force":
        E = 2100
        A = 20e-4
        I = 1.667e-8
        nu = 0.3
        G = E/(2 + 2 * nu)
        # G*=1e1
        length = 1
        Ei = np.array([E*A, G*A, G*A])
        Fi = np.array([2*G*I, E*I, E*I])
    material_model = Simo1986(Ei, Fi)

    # position and orientation of left point
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    # construct system
    system = System()

    # construct cantilever in a straight initial configuration
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

    clamping_left = RigidConnection(system.origin, cantilever, frame_ID2=(0,))

    # assemble the system
    system.add(cantilever)
    system.add(clamping_left)

    if load_type == "moment":
        # moment at cantilever tip
        m = material_model.Fi[2] * 2 * np.pi / length
        M = lambda t: 0.1 * t * e3 * m
        moment = K_Moment(M, cantilever, (1,))
        system.add(moment)
    elif load_type == "dead_load":
        # spatially fixed load at cantilever tip
        P = lambda t: material_model.Fi[2] * (10 * t) / length**2
        F = lambda t: - P(t) * e2
        force = Force(F, cantilever, (1,))
        system.add(force)
    elif load_type == "follower_force":
        # spatially fixed load at cantilever tip
        F = lambda t: - 3e-3 * t * e2 * 3 / 7
        force = K_Force(F, cantilever, (1,))
        system.add(force)
    elif load_type == "dead_load_and_moment":
        # spatially fixed load at cantilever tip
        P = lambda t: material_model.Fi[2] * (10 * t) / length**2
        F = lambda t: - P(t) * e2
        force = Force(F, cantilever, (1,))
        system.add(force)
        # moment at cantilever tip
        M = lambda t: 2.5 * P(t) * e3
        moment = K_Moment(M, cantilever, (1,))
        system.add(moment)
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

    # extract solutions
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol)
        e.export_contr(
            cantilever,
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever_curve",
        )
        e.export_contr(
            cantilever,
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
        q,
        [cantilever],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever.nelement + 1,
        repeat=False,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    Y_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    path = Path(__file__)

    if load_type == "moment":
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

        # camera settings for 3D plot
        ax1.azim = -90
        ax1.elev = 47
        ax1.dist = 8

    elif load_type == "dead_load":
        # add reference solution to the animation
        centerline_T = np.loadtxt(Path(path.parent, "cantilever_data","dead_load_centerline_T.txt"), delimiter=",", skiprows=1)
        ax1.plot(centerline_T[:, 0], centerline_T[:, 1], np.zeros_like(centerline_T[:, 0]), "-b")
        centerline_EB = np.loadtxt(Path(path.parent, "cantilever_data", "dead_load_centerline_EB.txt"), delimiter=",", skiprows=1)
        ax1.plot(centerline_EB[:, 0], centerline_EB[:, 1], np.zeros_like(centerline_EB[:, 0]), "-g")
        centerline_IEB = np.loadtxt(Path(path.parent, "cantilever_data", "dead_load_centerline_IEB.txt"), delimiter=",", skiprows=1)
        ax1.plot(centerline_IEB[:, 0], centerline_IEB[:, 1], np.zeros_like(centerline_IEB[:, 0]), "-r")

        # add normalized force-displacement diagram
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

    elif load_type == "follower_force":
        # add reference solution to the animation
        # Timoshenko beam
        centerline_T = np.loadtxt(Path(path.parent, "cantilever_data","follower_force_centerline_T_lambda_3_7.txt"), delimiter=",", skiprows=1)
        ax1.plot(centerline_T[:, 0], centerline_T[:, 1], np.zeros_like(centerline_T[:, 0]), "-b")
        # # Euler-Bernoulli beam
        # centerline_T = np.loadtxt(Path(path.parent, "cantilever_data","follower_force_centerline_EB_lambda_3_7.txt"), delimiter=",", skiprows=1)
        # ax1.plot(centerline_T[:, 0], centerline_T[:, 1], np.zeros_like(centerline_T[:, 0]), "-r")

        # camera settings for 3D plot
        ax1.azim = -90
        ax1.elev = 72
        ax1.dist = 8

        # add normalized force-displacement diagram
        fig2, ax2 = plt.subplots()
        # Timoshenko beam
        Deltas_T = np.loadtxt(Path(path.parent, "cantilever_data", "follower_force_Deltas_T.txt"), delimiter=",", skiprows=1)
        ax2.plot(Deltas_T[:43, 1], Deltas_T[:43, 0], 'o', color='blue', label='Timoshenko (numeric)')
        ax2.plot(Deltas_T[:43, 2], Deltas_T[:43, 0], 'o', color='blue')
        # Euler-Bernoulli beam
        Deltas_EB = np.loadtxt(Path(path.parent, "cantilever_data", "follower_force_Deltas_EB.txt"), delimiter=",", skiprows=1)
        ax2.plot(Deltas_EB[:43, 1], Deltas_EB[:43, 0], 'o', color='green', label='Euler-Bernoulli (numeric)')
        ax2.plot(Deltas_EB[:43, 2], Deltas_EB[:43, 0], 'o', color='green')

        # solution Argyris
        Delta_x_Argyris = np.loadtxt(Path(path.parent, "cantilever_data", "follower_force_Delta_x_Argyris.csv"), delimiter=";", skiprows=0)
        ax2.plot(Delta_x_Argyris[:-3, 0] / 10, Delta_x_Argyris[:-3, 1] / 300,  'x', color='red', label='- u_x / L (Argyris)')
        Delta_y_Argyris = np.loadtxt(Path(path.parent, "cantilever_data", "follower_force_Delta_y_Argyris.csv"), delimiter=";", skiprows=0)
        ax2.plot(Delta_y_Argyris[:-3, 0] / 10, Delta_y_Argyris[:-3, 1] / 300,  'x', color='red', label='- u_y / L (Argyris)')

        Delta_num = np.zeros(len(t))
        delta_num = np.zeros(len(t))

        for i in range(len(t)):
            r_OP_L = cantilever.nodes(q[i])[:, -1]
            Delta_num[i] = -(r_OP_L[0] - length) / length
            delta_num[i] = - r_OP_L[1] / length

        # Kirchhoff rod
        ax2.plot(delta_num, t * 3 / 7, '-', color='black', label='Cosserat (numeric)')
        ax2.plot(Delta_num, t * 3 / 7, '-', color='black')
        
        ax2.set_ylabel("lambda")
        ax2.set_xlabel("- u_x / L, - u_y / L")
        ax2.legend()
        ax2.grid()


    plt.show()
    

if __name__ == "__main__":
    # # load: moment at cantilever tip
    cantilever(load_type="moment", VTK_export=False)

    # load: dead load at cantilever tip
    # cantilever(load_type="dead_load", rod_hypothesis_penalty="shear_deformable", VTK_export=False)
    # cantilever(load_type="dead_load", rod_hypothesis_penalty="shear_rigid", VTK_export=False)
    # cantilever(load_type="dead_load", rod_hypothesis_penalty="inextensilbe_shear_rigid", VTK_export=False)

    # cantilever(load_type="follower_force", VTK_export=False)
