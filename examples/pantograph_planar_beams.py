import numpy as np
from math import pi, ceil, sin, cos
import matplotlib.pyplot as plt

from cardillo.model import Model
from cardillo.model.classical_beams.planar import Euler_bernoulli, Hooke
from cardillo.model.bilateral_constraints.implicit import Spherical_joint2D, Rigid_connection2D 
from cardillo.solver.newton import Newton
from cardillo.discretization.B_spline import uniform_knot_vector
from cardillo.model.frame import Frame

if __name__ == "__main__":
    # physical parameters
    gamma = pi/4
    nRow = 6
    nCol = 10
    # nRow = 10
    # nCol = 20
    L = 0.2041
    LBeam = L / (nCol * cos(gamma))
    EA = 1.6e9 * 1.6e-3 * 0.9e-3
    EI = 1.6e9 * (1.6e-3) * (0.9e-3)**3/12
    # displacementX = 0.0567

    displacementX = 0.03
    displacementY = 0.01

    A_rho0 = 0.1
    material_model = Hooke(EA, EI)

    ###################
    # create pantograph
    ###################
    p = 4
    assert p >= 2
    nQP = int(np.ceil((p + 1)**2 / 2))
    print(f'nQP: {nQP}')
    nEl = 2

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
    for idx in ID_mat[:, 0]:
        beam = beams[idx]
        r_OB = beam.r_OP(0, beam.q0[beam.qDOF_P(frame_ID2)], frame_ID=frame_ID2)
        frame_left = Frame(r_OB)
        model.add(frame_left)
        model.add(Rigid_connection2D(frame_left, beam, r_OB, frame_ID2=frame_ID2))

    # clamping at the right hand side
    for idx in ID_mat[:, -1]:
        beam = beams[idx]
        r_OB = beam.r_OP(0, beam.q0[beam.qDOF_P(frame_ID1)], frame_ID=frame_ID1)
        frame_right = Frame(lambda t, r=r_OB: r + np.array([t * displacementX, t * displacementY, 0]))
        model.add(frame_right)
        model.add(Rigid_connection2D(beam, frame_right, r_OB, frame_ID1=frame_ID1))

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
    solver = Newton(model, n_load_steps=20, max_iter=20, tol=1.0e-6, numerical_jacobian=False)
    sol = solver.solve()

    fig, ax = plt.subplots()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_xlim([-Ly, Ly*(nCol+1) + displacementX])
    ax.set_ylim([-Ly, Ly*(nRow+1)])
    ax.grid(linestyle='-', linewidth='0.5')
    ax.set_aspect('equal')

    for bdy in beams:
        x, y, z = bdy.centerline(sol.q[-1]).T
        ax.plot(x, y, '--r')

    plt.show()
    # # solve system and measure time (includes a dummy solver step, thus compilation time will not be measured)
    # # TODO: we need to build a better assembler/ a better sparse Newton-Raphson method in order to decreaase computation
    # #       time in the assembling of the sparse structures. SuperLu takes less than 10% of the computation time!
    # sol = Newton(model, nLoadSteps=2, tol=tol).integrate()
    # pr.enable()
    # sol = Newton(model, nLoadSteps=nLoadSteps, tol=tol).integrate()
    # pr.disable()

    # # plot deformed configuration
    # for bdy in assembler.bodyList:
    #     plotBeamConfiguration(bdy, ax[1], sol.q[-1], nPlot=100, nodeStyle='--o')

    # plt.show()

    # sortby = 'cumulative'
    # ps = pstats.Stats(pr).sort_stats(sortby)
    # ps.print_stats(0.10) # print only first 10% of the list