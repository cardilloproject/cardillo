import numpy as np
from math import pi, ceil, sin, cos
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from cardillo.model import Model
from cardillo.model.classical_beams.planar import Euler_bernoulli, Hooke
from cardillo.model.bilateral_constraints.implicit import Spherical_joint2D, Rigid_connection2D 
from cardillo.solver.newton import Newton
from cardillo.solver.euler_backward import Euler_backward
from cardillo.solver import Generalized_alpha_1, Scipy_ivp
from cardillo.discretization.B_spline import uniform_knot_vector
from cardillo.model.frame import Frame
from cardillo.math.algebra import A_IK_basic_z

if __name__ == "__main__":
    statics = False
    t1 = 1e-2
    dt = 1e-4
    # physical parameters
    gamma = pi/4
    # nRow = 6
    # nCol = 10
    nRow = 10
    nCol = 100
    L = 0.2041
    H = L / nCol * nRow
    LBeam = L / (nCol * cos(gamma))
    EA = 1.6e9 * 1.6e-3 * 0.9e-3
    EI = 1.6e9 * (1.6e-3) * (0.9e-3)**3/12

    displacementX_l = 0 #-0.0567/4
    # displacementX = 0.02
    displacementY_l = 0.0
    rotationZ_l = 0# -np.pi/10

    displacementX_r = 0# 0.0567/4
    # displacementX = 0.02
    displacementY_r = 0.0
    
    rotationZ_r = 0 #np.pi/10

    # r_OP_l = lambda t: np.array([0, H / 2, 0]) + np.array([t * displacementX_l, t * displacementY_r, 0])

    A = LBeam
    frq = 1/2 # frq times oscillation during simulation
    Omega = 2 * np.pi * frq / t1
    r_OP_l = lambda t: np.array([0, H / 2, 0]) + A * np.array([sin(Omega *t), 0, 0])

    A_IK_l = lambda t: A_IK_basic_z(t * rotationZ_l)

    r_OP_r = lambda t: np.array([L, H / 2, 0]) +  np.array([t * displacementX_r, t * displacementY_r, 0])
    A_IK_r = lambda t: A_IK_basic_z(t * rotationZ_r)
    
    A_rho0 = 0.1
    material_model = Hooke(EA, EI)

    ###################
    # create pantograph
    ###################
    p = 3
    assert p >= 2
    nQP = int(np.ceil((p + 1)**2 / 2))
    # nQP = p

    print(f'nQP: {nQP}')
    nEl = 1

    # projections of beam length
    Lx = LBeam * cos(gamma)
    Ly = LBeam * sin(gamma)

    # upper left node
    xUL = 0         
    yUL = Ly*nRow

    # build reference configuration
    nNd = nEl + p
    X0 = np.linspace(0, L, nNd)
    Xi = uniform_knot_vector(p, nEl)
    for i in range(nNd):
        X0[i] = np.sum(Xi[i+1:i+p+1])
    Y1 = -np.copy(X0) * Ly / p
    X1 = X0 * Lx / p

    X2 = np.copy(X1)
    Y2 = -np.copy(Y1)
    
    # create model
    model = Model()

    # create beams
    beams = []
    ID_mat = np.zeros((nRow, nCol)).astype(int)
    ID = 0
    for brow in range(0, nRow, 2):
        for bcol in range(0, nCol, 2):
            X = X1 + xUL + Lx * bcol
            Y = Y1 + yUL - Ly * brow

            # beam 1
            Q = np.concatenate([X, Y])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol] = ID
            ID = ID + 1
            
            # beam 2
            Q = np.concatenate([X2 + X[-1], Y2 + Y[-1]])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol + 1] = ID
            ID = ID + 1

            
    for brow in range(1, nRow, 2):
        for bcol in range(0, nCol, 2):
            X = X2 + xUL + Lx * bcol
            Y = Y2 + yUL - Ly * (brow + 1)
            # beam 1
            Q = np.concatenate([X, Y])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol] = ID
            ID = ID + 1
            # beam 2
            Q = np.concatenate([X1 + X[-1], Y1 + Y[-1]])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol + 1] = ID
            ID = ID + 1

    # junctions in the beam families 

    frame_ID1 = (1,)
    frame_ID2 = (0,)
            
    # odd colums
    for bcol in range(0, nCol, 2):
        for brow in range(0, nRow, 2):
            beam1 = beams[ID_mat[brow, bcol]]
            beam2 = beams[ID_mat[brow + 1, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Rigid_connection2D(beam1, beam2, r_OB, frame_ID1=frame_ID1, frame_ID2=frame_ID2))

            beam1 = beams[ID_mat[brow + 1, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Rigid_connection2D(beam1, beam2, r_OB, frame_ID1=frame_ID1, frame_ID2=frame_ID2))

    # even columns
    for bcol in range(1, nCol - 1, 2):
        for brow in range(1, nRow - 1, 2):
            beam1 = beams[ID_mat[brow, bcol]]
            beam2 = beams[ID_mat[brow + 1, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Rigid_connection2D(beam1, beam2, r_OB, frame_ID1=frame_ID1, frame_ID2=frame_ID2))

            beam1 = beams[ID_mat[brow + 1, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Rigid_connection2D(beam1, beam2, r_OB, frame_ID1=frame_ID1, frame_ID2=frame_ID2))

    # pivots between beam families
            
    # internal pivots
    for brow in range(0, nRow, 2):
        for bcol in range(0, nCol - 1):
            beam1 = beams[ID_mat[brow, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Spherical_joint2D(beam1, beam2, r_OB, frame_ID1=frame_ID1, frame_ID2=frame_ID2))

    # lower boundary pivots
    for bcol in range(1, nCol - 1, 2):
        beam1 = beams[ID_mat[-1, bcol]]
        beam2 = beams[ID_mat[-1, bcol + 1]]
        r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
        model.add(Spherical_joint2D(beam1, beam2, r_OB, frame_ID1=frame_ID1, frame_ID2=frame_ID2))

    # clamping at the left hand side
    frame_l = Frame(r_OP=r_OP_l, A_IK=A_IK_l)
    model.add(frame_l)
    for idx in ID_mat[:, 0]:
        beam = beams[idx]
        r_OB = beam.r_OP(0, beam.q0[beam.qDOF_P(frame_ID2)], frame_ID=frame_ID2)
        model.add(Rigid_connection2D(frame_l, beam, r_OB, frame_ID2=frame_ID2))

    # clamping at the right hand side
    frame_r = Frame(r_OP=r_OP_r, A_IK = A_IK_r)
    model.add(frame_r)
    for idx in ID_mat[:, -1]:
        beam = beams[idx]
        r_OB = beam.r_OP(0, beam.q0[beam.qDOF_P(frame_ID1)], frame_ID=frame_ID1)
        model.add(Rigid_connection2D(beam, frame_r, r_OB, frame_ID1=frame_ID1))

    # assemble model
    model.assemble()

    # plot initial configuration
    # fig, ax = plt.subplots()
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # ax.set_xlim([-Ly, Ly*(nCol+1)])
    # ax.set_ylim([-Ly, Ly*(nRow+1)])
    # ax.grid(linestyle='-', linewidth='0.5')
    # ax.set_aspect('equal')


    # for bdy in beams:
    #     x, y, z = bdy.centerline(model.q0).T
    #     ax.plot(x, y, '--k')

    # plt.show()

    ######################
    # solve static problem
    ######################
    if statics:
        solver = Newton(model, n_load_steps=3, max_iter=20, tol=1.0e-6, numerical_jacobian=False)
    else:
        solver = Euler_backward(model, t1, dt, newton_max_iter=50, numerical_jacobian=False, debug=False)
        # solver = Generalized_alpha_1(model, t1, dt, variable_dt=False, rho_inf=0.5)
        # solver = Scipy_ivp(model, t1, dt, atol=1e-6)


    sol = solver.solve()

    if statics:
        fig, ax = plt.subplots()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        # ax.set_xlim([-Ly + H/2 * sin(rotationZ_l), Ly*(nCol+1) + displacementX + H/2 * sin(rotationZ_r)])
        # ax.set_ylim([-Ly, Ly*(nRow+1) + displacementY])
        ax.grid(linestyle='-', linewidth='0.5')
        ax.set_aspect('equal')

        for bdy in beams:
            x, y, z = bdy.centerline(sol.q[-1]).T
            ax.plot(x, y, '-b')

        plt.show()
    else:
        # animate configurations
        fig, ax = plt.subplots()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim([-Ly + H/2 * sin(rotationZ_l), Ly*(nCol+1) + displacementX_r + H/2 * sin(rotationZ_r)])
        ax.set_ylim([-Ly, Ly*(nRow+1) + displacementY_r])
        ax.grid(linestyle='-', linewidth='0.5')
        ax.set_aspect('equal')

        # prepare data for animation
        t = sol.t
        frames = len(t)
        target_frames = min(len(t), 100)
        frac = int(frames / target_frames)
        animation_time = 5
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = sol.q[::frac]

        centerlines = []
        # lobj, = ax.plot([], [], '-k')
        for bdy in beams:
            lobj, = ax.plot([], [], '-k')
            centerlines.append(lobj)
            
        def animate(i):
            for idx, bdy in enumerate(beams):
                x, y, _ = bdy.centerline(q[i], n=2).T
                centerlines[idx].set_data(x, y)

            return centerlines

        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)

        plt.show()