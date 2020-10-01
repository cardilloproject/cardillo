from numpy.lib.npyio import save
from cardillo.solver.solution import load_solution, save_solution
import numpy as np
from math import pi, sin, cos, exp, atan2, sqrt, ceil, cosh
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from cardillo.model import Model
from cardillo.model.classical_beams.planar import Euler_bernoulli, Hooke, Inextensible_Euler_bernoulli
from cardillo.model.bilateral_constraints.implicit import Rigid_connection2D, Linear_guidance_x_2D
from cardillo.model.scalar_force_interactions.force_laws import Linear_spring
from cardillo.model.scalar_force_interactions import add_rotational_forcelaw
from cardillo.solver.newton import Newton
from cardillo.solver import Generalized_alpha_index3_panto
from cardillo.discretization.B_spline import uniform_knot_vector
from cardillo.model.frame import Frame
from cardillo.math.algebra import A_IK_basic_z
from cardillo.utility.post_processing_vtk import post_processing
from scipy.interpolate import interp1d

from cardillo.discretization.B_spline import B_spline_basis1D
class Junction():
    def __init__(self, beam1, beam2, la_g0=None):
        # rigid connection between to consecutive beams. End of beam1 is connected to start of beam2.
        self.nla_g = 3
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.beam1 = beam1
        self.beam2 = beam2

        self.frame_ID1 = (1,)
        self.frame_ID2 = (0,)
       
        N, N_xi = B_spline_basis1D(beam1.polynomial_degree, 1, beam1.knot_vector.data, 1).T
        self.beam1_N = self.stack_shapefunctions(N, beam1.nq_el)
        self.beam1_N_xi = self.stack_shapefunctions(N_xi, beam1.nq_el)

        N, N_xi = B_spline_basis1D(beam2.polynomial_degree, 1, beam2.knot_vector.data, 0).T
        self.beam2_N = self.stack_shapefunctions(N, beam2.nq_el)
        self.beam2_N_xi_perp = self.stack_shapefunctions_perp(N_xi, beam2.nq_el)
        

    def assembler_callback(self):
        qDOF1 = self.beam1.qDOF_P(self.frame_ID1)
        qDOF2 = self.beam2.qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate([self.beam1.qDOF[qDOF1], self.beam2.qDOF[qDOF2]])
        self.nq1 = nq1 = len(qDOF1)
        self.nq2 = len(qDOF2)
        self._nq = self.nq1 + self.nq2
        
        uDOF1 = self.beam1.uDOF_P(self.frame_ID1)
        uDOF2 = self.beam2.uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate([self.beam1.uDOF[uDOF1], self.beam2.uDOF[uDOF2]])
        self.nu1 = nu1 = len(uDOF1)
        self.nu2 = len(uDOF2)
        self._nu = self.nu1 + self.nu2

    def g(self, t, q):
        nq1 = self.nq1
        r_OP1 = self.beam1_N @ q[:nq1]
        r_OP2 = self.beam2_N @ q[nq1:]
        # tangent vector beam 1
        t = self.beam1_N_xi @ q[:nq1]
        # normal vector beam 2
        n = self.beam2_N_xi_perp @ q[nq1:]

        return np.concatenate([r_OP2 - r_OP1, [t @ n]]) 
        
    def g_q_dense(self, t, q):
        nq1 = self.nq1
        g_q = np.zeros((self.nla_g, self._nq))
        g_q[:2, :nq1] = - self.beam1_N
        g_q[:2, nq1:] = self.beam2_N

        # tangent vector beam 1
        t = self.beam1_N_xi @ q[:nq1]
        # normal vector beam 2
        n = self.beam2_N_xi_perp @ q[nq1:]

        g_q[2, :nq1] = n @ self.beam1_N_xi
        g_q[2, nq1:] = t @ self.beam2_N_xi_perp
        return g_q

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q).T, (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        # dense_num = Numerical_derivative(lambda t, q: self.g_q_dense(t, q).T @ la_g, order=2)._x(t, q)
        # [la_g[0], la_g[1]] @ (self.beam2_N - self.beam1_N) independent of q
        # [la_g[2] * self.beam1_N_xi.T @ n , la_g[2] * self.beam2_N_xi_perp.T @ t]
        nq1 = self.nq1
        nu1 = self.nu1

        dense = np.zeros((self._nu, self._nq))
        dense[:nu1, nq1:] = la_g[2] * self.beam1_N_xi.T @ self.beam2_N_xi_perp
        dense[nu1:, :nq1] = la_g[2] * self.beam2_N_xi_perp.T @ self.beam1_N_xi
        
        coo.extend( dense, (self.uDOF, self.qDOF))

    def stack_shapefunctions(self, N, nq_el):
        # return np.kron(np.eye(2), N)
        n2 = int(nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, :n2] = N
        NN[1, n2:] = N
        return NN

    def stack_shapefunctions_perp(self, N, nq_el):
        # return np.kron(np.array([[0, -1], [1, 0]]), N)
        n2 = int(nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, n2:] = -N
        NN[1, :n2] = N
        return NN

class Pivot_w_spring():
    def __init__(self, beam1, beam2, force_law, la_g0=None):
        # pivot between to consecutive beams. End of beam1 is connected to start of beam2.
        self.nla_g = 2
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.beam1 = beam1
        self.beam2 = beam2

        self.frame_ID1 = (1,)
        self.frame_ID2 = (0,)

        self.force_law = force_law
       
        N, N_xi = B_spline_basis1D(beam1.polynomial_degree, 1, beam1.knot_vector.data, 1).T
        self.beam1_N = self.stack_shapefunctions(N, beam1.nq_el)
        self.beam1_N_xi = self.stack_shapefunctions(N_xi, beam1.nq_el)
        self.beam1_N_xi_perp = self.stack_shapefunctions_perp(N_xi, beam1.nq_el)

        N, N_xi = B_spline_basis1D(beam2.polynomial_degree, 1, beam2.knot_vector.data, 0).T
        self.beam2_N = self.stack_shapefunctions(N, beam2.nq_el)
        self.beam2_N_xi = self.stack_shapefunctions(N_xi, beam2.nq_el)
        self.beam2_N_xi_perp = self.stack_shapefunctions_perp(N_xi, beam2.nq_el)

    def assembler_callback(self):
        qDOF1 = self.beam1.qDOF_P(self.frame_ID1)
        qDOF2 = self.beam2.qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate([self.beam1.qDOF[qDOF1], self.beam2.qDOF[qDOF2]])
        self.nq1 = nq1 = len(qDOF1)
        self.nq2 = len(qDOF2)
        self._nq = self.nq1 + self.nq2
        
        uDOF1 = self.beam1.uDOF_P(self.frame_ID1)
        uDOF2 = self.beam2.uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate([self.beam1.uDOF[uDOF1], self.beam2.uDOF[uDOF2]])
        self.nu1 = nu1 = len(uDOF1)
        self.nu2 = len(uDOF2)
        self._nu = self.nu1 + self.nu2

        q0_beam1 = self.beam1.q0[qDOF1]
        q0_beam2 = self.beam2.q0[qDOF2]

        t0_beam1 = self.beam1_N_xi @ q0_beam1
        theta0_beam1 = atan2(t0_beam1[1], t0_beam1[0])
        
        t0_beam2 = self.beam2_N_xi @ q0_beam2
        theta0_beam2 = atan2(t0_beam2[1], t0_beam2[0])

        # undeformed angle for torsional spring
        self.delta_theta0 = theta0_beam1 - theta0_beam2

        if self.force_law.g0 is None:
            self.force_law.g0 = self.delta_theta0

    def g(self, t, q):
        nq1 = self.nq1
        r_OP1 = self.beam1_N @ q[:nq1]
        r_OP2 = self.beam2_N @ q[nq1:]
        return r_OP2 - r_OP1
        
    def g_q_dense(self, t, q):
        return np.hstack([-self.beam1_N, self.beam2_N])

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q).T, (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        pass

    def stack_shapefunctions(self, N, nq_el):
        # return np.kron(np.eye(2), N)
        n2 = int(nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, :n2] = N
        NN[1, n2:] = N
        return NN

    def stack_shapefunctions_perp(self, N, nq_el):
        # return np.kron(np.array([[0, -1], [1, 0]]), N)
        n2 = int(nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, n2:] = -N
        NN[1, :n2] = N
        return NN

    def pot(self, t, q):
        return self.force_law.pot(t, self.__g(t, q))

    def f_pot(self, t, q):
        return - self.g_spring_q(t, q) * self.force_law.pot_g(t, self.g_spring(t, q))

    def g_spring(self, t, q):
        nq1 = self.nq1

        T1 = self.beam1_N_xi @ q[:nq1]
        T2 = self.beam2_N_xi @ q[nq1:]

        theta1 = atan2(T1[1], T1[0])
        theta2 = atan2(T2[1], T2[0])

        return theta1 - theta2

    def g_spring_q(self, t, q):
        nq1 = self.nq1
        T1 = self.beam1_N_xi @ q[:nq1]
        T2 = self.beam2_N_xi @ q[nq1:]
        
        g_q1 = (T1[0] * self.beam1_N_xi[1] - T1[1] * self.beam1_N_xi[0]) / (T1 @ T1)
        g_q2 = (T2[0] * self.beam2_N_xi[1] - T2[1] * self.beam2_N_xi[0]) / (T2 @ T2)

        W = np.hstack([g_q1, -g_q2])

        return W

    def f_pot_q(self, t, q, coo):
        # dense_num = Numerical_derivative(lambda t, q: self.f_pot(t, q), order=2)._x(t, q)
        dense = np.zeros((self._nu, self._nq))

        # current tangent vector
        nq1 = self.nq1
        T1 = self.beam1_N_xi @ q[:nq1]
        T2 = self.beam2_N_xi @ q[nq1:]

        # angle stiffness
        tmp1_1 = np.outer(self.beam1_N_xi[1], self.beam1_N_xi[0]) + np.outer(self.beam1_N_xi[0], self.beam1_N_xi[1])
        tmp1_2 = np.outer(self.beam1_N_xi[0], self.beam1_N_xi[0]) - np.outer(self.beam1_N_xi[1], self.beam1_N_xi[1])
        
        tmp2_1 = np.outer(self.beam2_N_xi[1], self.beam2_N_xi[0]) + np.outer(self.beam2_N_xi[0], self.beam2_N_xi[1])
        tmp2_2 = np.outer(self.beam2_N_xi[0], self.beam2_N_xi[0]) - np.outer(self.beam2_N_xi[1], self.beam2_N_xi[1])

        g_qq = np.zeros((self._nq, self._nq))
        g_qq[:nq1, :nq1] =   ((T1[1]**2 - T1[0]**2) * tmp1_1 + 2 * T1[0] * T1[1] * tmp1_2) / (T1 @ T1)**2
        g_qq[nq1:, nq1:] = - ((T2[1]**2 - T2[0]**2) * tmp2_1 + 2 * T2[0] * T2[1] * tmp2_2) / (T2 @ T2)**2
   
        W = self.g_spring_q(t, q)

        dense = - g_qq * self.force_law.pot_g(t, self.g_spring(t,q)) \
            - self.force_law.pot_gg(t, self.g_spring(t,q)) * np.outer(W, W)
                    
        coo.extend(dense, (self.uDOF, self.qDOF))

def create_pantograph(gamma, nRow, nCol, H, EA, EI, GI, A_rho0, p, nEl, nQP, r_OP_l, A_IK_l, r_OP_r, A_IK_r):
    
    assert p >= 2
    LBeam = H / (nRow * sin(gamma))
    L = nCol * LBeam * cos(gamma)

    ###################################################
    # create reference configuration individual beams #
    ###################################################
    
    # projections of beam length
    Lx = LBeam * cos(gamma)
    Ly = LBeam * sin(gamma)
    # upper left node
    xUL = 0         
    yUL = Ly*nRow
    # build reference configuration beam family 1
    nNd = nEl + p
    X0 = np.linspace(0, LBeam, nNd)
    Xi = uniform_knot_vector(p, nEl)
    for i in range(nNd):
        X0[i] = np.sum(Xi[i+1:i+p+1])
    Y1 = -np.copy(X0) * Ly / p
    X1 = X0 * Lx / p
    # build reference configuration beam family 2
    X2 = np.copy(X1)
    Y2 = -np.copy(Y1)
    
    #############
    # add beams #
    #############

    model = Model()

    material_model = Hooke(EA, EI)
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
            ID += 1
            
            # beam 2
            Q = np.concatenate([X2 + X[-1], Y2 + Y[-1]])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol + 1] = ID
            ID += 1
      
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
            ID += 1
            # beam 2
            Q = np.concatenate([X1 + X[-1], Y1 + Y[-1]])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol + 1] = ID
            ID += 1

    ######################################
    # add junctions within beam families #
    ######################################

    frame_ID1 = (1,)
    frame_ID2 = (0,)
            
    # odd colums
    for bcol in range(0, nCol, 2):
        for brow in range(0, nRow, 2):
            beam1 = beams[ID_mat[brow, bcol]]
            beam2 = beams[ID_mat[brow + 1, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Junction(beam1, beam2))

            beam1 = beams[ID_mat[brow + 1, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Junction(beam1, beam2))

    # even columns
    for bcol in range(1, nCol - 1, 2):
        for brow in range(1, nRow - 1, 2):
            beam1 = beams[ID_mat[brow, bcol]]
            beam2 = beams[ID_mat[brow + 1, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Junction(beam1, beam2))

            beam1 = beams[ID_mat[brow + 1, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            model.add(Junction(beam1, beam2))

    ##########################################################
    # add pivots and torsional springs between beam families #
    ##########################################################
    
    # internal pivots
    for brow in range(0, nRow, 2):
        for bcol in range(0, nCol - 1):
            beam1 = beams[ID_mat[brow, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            spring = Linear_spring(GI)
            model.add(Pivot_w_spring(beam1, beam2, spring))

    # lower boundary pivots
    for bcol in range(1, nCol - 1, 2):
        beam1 = beams[ID_mat[-1, bcol]]
        beam2 = beams[ID_mat[-1, bcol + 1]]
        r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
        spring = Linear_spring(GI)
        model.add(Pivot_w_spring(beam1, beam2, spring))

    # not used for convergence study
    # left boundary pivots
    for brow in range(1, nRow - 2, 2):
        beam1 = beams[ID_mat[brow, 0]]
        beam2 = beams[ID_mat[brow + 1, 0]]
        r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID2)], frame_ID2)
        spring = Linear_spring(GI)
        model.add(Pivot_w_spring(beam1, beam2, spring))

    # right boundary pivots
    for brow in range(1, nRow - 2, 2):
        beam1 = beams[ID_mat[brow, -1]]
        beam2 = beams[ID_mat[brow + 1, -1]]
        r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
        spring = Linear_spring(GI)
        model.add(Pivot_w_spring(beam1, beam2, spring))
    

    ###########################
    # add boundary conditions #
    ###########################

    # clamping at the left hand side

    # frame_l = Frame(r_OP=r_OP_l, A_IK=A_IK_l)
    # model.add(frame_l)
    # for idx in ID_mat[:, 0]:
    #     beam = beams[idx]
    #     r_OB = beam.r_OP(0, beam.q0[beam.qDOF_P(frame_ID2)], frame_ID=frame_ID2)
    #     # model.add(Rigid_connection2D(frame_l, beam, r_OB, frame_ID2=frame_ID2))
    #     model.add(Linear_guidance_x_2D(frame_l, beam, r_OB, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), frame_ID2=frame_ID2))

    # excitation even beams
    exc_frames = []
    ex_ID = 0
    for i in range(0, nRow, 2):        
        exc_frames.append(Frame(r_OP=r_OP_l[ceil(i / 2)], A_IK=A_IK_l))
        model.add(exc_frames[ex_ID])
        beam = beams[ID_mat[i, 0]]
        r_OB = beam.r_OP(0, beam.q0[beam.qDOF_P(frame_ID2)], frame_ID=frame_ID2)
        # model.add(Rigid_connection2D(exc_frames[ex_ID], beam, r_OB, frame_ID2=frame_ID2))
        model.add(Linear_guidance_x_2D(exc_frames[ex_ID], beam, r_OB, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), frame_ID2=frame_ID2))
        ex_ID += 1

    # excitation odd beams
    for i in range(1, nRow, 2):        
        exc_frames.append(Frame(r_OP=r_OP_l[ceil((i + 1) / 2)], A_IK=A_IK_l))
        model.add(exc_frames[ex_ID])
        beam = beams[ID_mat[i, 0]]
        r_OB = beam.r_OP(0, beam.q0[beam.qDOF_P(frame_ID2)], frame_ID=frame_ID2)
        # model.add(Rigid_connection2D(exc_frames[ex_ID], beam, r_OB, frame_ID2=frame_ID2))
        model.add(Linear_guidance_x_2D(exc_frames[ex_ID], beam, r_OB, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), frame_ID2=frame_ID2))

        ex_ID += 1
    

    # clamping at the right hand side
    frame_r = Frame(r_OP=r_OP_r, A_IK = A_IK_r)
    model.add(frame_r)
    for idx in ID_mat[:, -1]:
        beam = beams[idx]
        r_OB = beam.r_OP(0, beam.q0[beam.qDOF_P(frame_ID1)], frame_ID=frame_ID1)
        # model.add(Rigid_connection2D(beam, frame_r, r_OB, frame_ID1=frame_ID1))
        model.add(Linear_guidance_x_2D(beam, frame_r, r_OB, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), frame_ID1=frame_ID1))

    # assemble model
    model.assemble()

    return model, beams, ID_mat

if __name__ == "__main__":
    load_excitation = False
    dynamics = True
    solve_problem = True
    time_displacement_diagram = True
    position_displacement_diagram = True
    paraview_export = True

###############################################################################################
    # time simulation parameters
    t1 = 2e-2                      # simulation time
    dt = 1e-5                           # time step
    rho_inf = 0.8                       # damping parameter generalized-alpha integrator

    # beam finite element parameters
    p = 3                               # polynomial degree
    nQP = p + 2                         # number of quadrature points
    nEl = 1                             # number of elements per beam

    # geometric parameters
    gamma = pi/4                        # angle between fiber families
    nRow = 20                           # number of rows = 2 * number of fibers per height
    nCol = 200                        # number of columns = 2 * number of fibers per length

    H = 0.07                            # height of pantographic sheet
    LBeam = H / (nRow * sin(gamma))     # length of individual beam
    L = nCol * LBeam * cos(gamma)       # length of pantographic sheet
    
    # material prameters
    Yb = 500e6                          # Young's modulus material
    Gb = Yb / (2 * (1 + 0.4))           # Shear modulus material
    a = 1.6e-3                          # height of beam cross section
    b = 1e-3                            # width of beam cross section
    rp = 0.45e-3                        # radius of pivots
    hp = 1e-3                           # height of pivots
    
    EA = Yb * a * b                     # axial stiffness
    EI = Yb * (a * b**3) / 12           # bending stiffness
    GI = Gb * 0.5*(np.pi * rp**4) / hp  # shear stiffness

    A_rho0 = 930 * a * b                # density per unit length

    # boundary conditions
    # excitation function
    displ = H / 5

    n_excited_fibers = ceil(nRow / 2 + 1)

    r_OP_l = []

    if load_excitation:
        u_x = load_solution(f'displacement_x')
        u_y = load_solution(f'displacement_y')
        time = load_solution(f'time')


        fig, ax = plt.subplots(2, 1)
        fig.suptitle('displacements', fontsize=16)

        for i in range(len(u_x)):
            ax[0].plot(time, u_x[i], label=f'Row {i}')
            ax[1].plot(time, u_y[i], label=f'Row {i}')


        ax[0].set_xlabel('time t [s]')
        ax[0].set_ylabel('displacement x-direction [m]')
        ax[0].grid()
        ax[0].legend()
        
        ax[1].set_xlabel('time t [s]')
        ax[1].set_ylabel('displacement y-direction [m]')
        ax[1].grid()
        ax[1].legend()

        plt.show()

        for i in range(n_excited_fibers):       
            r_OP_l.append(lambda t, k=i: np.array([interp1d(time, u_x[k])([t])[0],
                                      interp1d(time, u_y[k])([t])[0],
                                      0]))
    else:

        c1 = 0.001
        c2 = 0.005     # center of bell

        fcn = lambda t: displ * np.exp(-(t-c2)**2/c1**2)

        fig, ax = plt.subplots()
        ax.set_xlabel('time t [s]')
        ax.set_ylabel('displacement [m]')
        x = np.linspace(0, t1, 1000)
        y = []

        for t in x:
            y.append(fcn(t))

        ax.plot(x, y)
        plt.show()
        for i in range(n_excited_fibers):       
            r_OP_l.append(lambda t, k=i: np.array([fcn(t), 0, 0]))
    

    rotationZ_l = 0
    rotationZ_r = 0

    # r_OP_l = lambda t: np.array([0, H / 2, 0]) + fcn(t) * np.array([1, 0, 0])
    A_IK_l = lambda t: A_IK_basic_z(t * rotationZ_l)

    r_OP_r = lambda t: np.array([L, H / 2, 0]) - fcn(t) * np.array([1, 0, 0])
    # r_OP_r = lambda t: np.array([L, H / 2, 0])
    A_IK_r = lambda t: A_IK_basic_z(t * rotationZ_r)
    ######################################################################################

    # create pantograph
    model, beams, ID_mat = create_pantograph(gamma, nRow, nCol, H, EA, EI, GI, A_rho0, p, nEl, nQP, r_OP_l, A_IK_l, r_OP_r, A_IK_r)

    # create .vtu file for initial configuration
    # post_processing(beams, np.array([0]), model.q0.reshape(1, model.q0.shape[0]), 'Pantograph_initial_configuration', binary=True)

    # choose solver
    if dynamics:
        solver = Generalized_alpha_index3_panto(model, t1, dt, rho_inf=rho_inf)
    else:
        solver = Newton(model, n_load_steps=5, max_iter=50, tol=1.0e-10, numerical_jacobian=False)
        
    # solve or load problem
    if solve_problem:
        # import cProfile, pstats
        # pr = cProfile.Profile()
        # pr.enable()
        sol = solver.solve()
        # pr.disable()

        # sortby = 'cumulative'
        # ps = pstats.Stats(pr).sort_stats(sortby)
        # ps.print_stats(0.1) # print only first 10% of the list
        if dynamics == True:
            save_solution(sol, f'Pantographic_sheet_{nRow}x{nCol}_rho_{rho_inf}_nEl_{nEl}_nQP_{nQP}_')
        else:
            save_solution(sol, f'Pantographic_sheet_{nRow}x{nCol}_statics')
    else:
        if dynamics == True:
            sol = load_solution(f'Pantographic_sheet_{nRow}x{nCol}_rho_{rho_inf}_nEl_{nEl}_nQP_{nQP}_')
        else:
            sol = load_solution(f'Pantographic_sheet_{nRow}x{nCol}_statics')

    # prepare data for animation and paraview export
    slow_motion_factor = 100
    animation_time = t1 * slow_motion_factor
    fps = 25
    frames = int(min(sol.q.shape[0], fps * animation_time))
    frac = int(sol.q.shape[0] / frames)
    interval = animation_time * 1000 / frames # delay between frams in milliseconds
    
    tt = sol.t[::frac]
    qq = sol.q[::frac]
    uu = sol.u[::frac]

    # time-displacement diagramm
    if time_displacement_diagram:
        cs_idx = 100

        u_x_centerline = []

        u_x_crosssection = []
        u_y_crosssection = []
        u_zero_crosssection = []

        for i in range(0, nRow, 2):
            u_x_c = []
            u_x = []
            u_y = []
            beam = beams[ID_mat[i, cs_idx]]

            pos_0 = beam.r_OP(0,sol.q[0, beam.qDOF[beam.qDOF_P((0,))]], (0,))
            for k, t in enumerate(sol.t):
                pos_t = beam.r_OP(t,sol.q[k, beam.qDOF[beam.qDOF_P((0,))]], (0,))
                displacement = pos_t - pos_0 + pos_0[1]
                u_x.append(displacement[0])
                u_y.append(displacement[1])
                if i == 10:
                    u_x_c.append((pos_t - pos_0)[0])

            if i == 10:
                u_x_centerline.extend([u_x_c])

            u_x_crosssection.extend([u_x])
            u_y_crosssection.extend([u_y])

        i = nRow-1
        u_x = []
        u_y = []
        beam = beams[ID_mat[i, cs_idx]]
        pos_0 = beam.r_OP(0,sol.q[0, beam.qDOF[beam.qDOF_P((0,))]], (0,))
        for k, t in enumerate(sol.t):
            pos_t = beam.r_OP(t,sol.q[k, beam.qDOF[beam.qDOF_P((0,))]], (0,))
            displacement = pos_t - pos_0 + pos_0[1]
            u_x.append(displacement[0])
            u_y.append(displacement[1])
            

        u_x_crosssection.extend([u_x])
        u_y_crosssection.extend([u_y])

        # post processing data
        fig2, ax2 = plt.subplots(2, 1)
        fig2.suptitle(f'displacements at x={cs_idx/nCol}L', fontsize=16)

        for i in range(len(u_x_crosssection)):
            ax2[0].plot(sol.t, u_x_crosssection[i], label=f'Row {i}')
            ax2[1].plot(sol.t, u_y_crosssection[i], label=f'Row {i}')

        ax2[0].set_xlabel('time t [s]')
        ax2[0].set_ylabel('displacement x-direction [m]')
        ax2[0].grid()
        ax2[0].legend()
        
        ax2[1].set_xlabel('time t [s]')
        ax2[1].set_ylabel('displacement y-direction [m]')
        ax2[1].grid()
        ax2[1].legend()

        # new_time = sol.t - sol.t[0]
        # save_solution(u_x_crosssection, f'displacement_x')
        # save_solution(u_y_crosssection, f'displacement_y')
        # save_solution(new_time, f'time')

        c1 = 0.00475
        c2 = 0.01865 # center of bell
        fcn2 = lambda t: displ * np.exp(-(t-c2)**2/c1**2) #*(t*(t<c1)+c1*(t>=c1))/c1
        # fcn2 = lambda t: displ * 1 / cosh((t-c2)*350)*(t*(t<c1)+c1*(t>=c1))/c1

        y = []
        for t in sol.t:
            y.append(fcn2(t))


        fig3, ax3 = plt.subplots()
        ax3.set_xlabel('time t [s]')
        ax3.set_ylabel('displacement [m]')

        fig3.suptitle(f'displacement in x-direction of centerline at x={cs_idx/nCol}L', fontsize=16)
        ax3.plot(sol.t, u_x_centerline[0], label=f'num')
        # ax3.plot(sol.t, y, label=f'moved excitation')

    # position-displacement diagramm centerline
    if position_displacement_diagram:

        u_x_time = []
        u_y_time = []
        time = []
        
        for t_idx, t in enumerate(tt):
            u_x = []
            u_y = []
            X_0 = []

            for i in range(0, nCol, 2):
                beam = beams[ID_mat[10, i]]
                pos_0 = beam.r_OP(0 , qq[0, beam.qDOF[beam.qDOF_P((0,))]], (0,))
                pos_t = beam.r_OP(t , qq[t_idx, beam.qDOF[beam.qDOF_P((0,))]], (0,))
                displacement = pos_t - pos_0
                u_x.append(displacement[0])
                u_y.append(displacement[1])
                X_0.append(pos_0[0])

            beam = beams[ID_mat[10, nCol-1]]
            pos_0 = beam.r_OP(0 , qq[0, beam.qDOF[beam.qDOF_P((1,))]], (1,))
            pos_t = beam.r_OP(t , qq[t_idx, beam.qDOF[beam.qDOF_P((1,))]], (1,))
            displacement = pos_t - pos_0
            u_x.append(displacement[0])
            u_y.append(displacement[1])
            X_0.append(pos_0[0])
            
            u_x_time.extend([u_x])
            u_y_time.extend([u_y])
            time.append(t)


        # animate configurations
        fig4, ax4 = plt.subplots()
        fig4.suptitle(f'displacement in x-direction of centerline', fontsize=16)

        # ax.axis('equal')
        # scale = 2.5 * R
        # ax.axis('equal')
        ax4.set_xlim([0, L])
        ax4.set_ylim([-1.2 * displ, 1.2 * displ])
        ax4.set_xlabel('position x [m]')
        ax4.set_ylabel('displacement x-direction [m]')

        center_line, = ax4.plot([], [], '-k')

        def animate(i):
            center_line.set_data(X_0, u_x_time[i])
            return center_line, 

        anim = animation.FuncAnimation(fig4, animate, frames=frames, interval=interval, blit=False)

        # post processing data
        fig5, ax5 = plt.subplots(2, 1)
        fig5.suptitle(f'displacement in x-direction of centerline', fontsize=16)

        for i in range(len(time)):
            ax5[0].plot(X_0, u_x_time[i], label=f'time {time[i]}')
            ax5[1].plot(X_0, u_y_time[i], label=f'time {time[i]}')

        ax5[0].set_xlabel('position x [m]')
        ax5[0].set_ylabel('displacement x-direction [m]')
        ax5[0].grid()
        
        ax5[1].set_xlabel('position x [m]')
        ax5[1].set_ylabel('displacement y-direction [m]')
        ax5[1].grid()

    plt.show()

    # post processing for paraview
    if paraview_export:
        if dynamics:
            post_processing(beams, tt, qq, f'Pantographic_sheet_{nRow}x{nCol}_rho_{rho_inf}_nEl_{nEl}_nQP_{nQP}_', u = uu, binary=True)
        else:
            post_processing(beams, tt, qq, f'Pantographic_sheet_{nRow}x{nCol}_statics', binary=True)