from pickle import load
from cardillo.solver.solution import load_solution, save_solution
import numpy as np
from math import pi, ceil, sin, cos, exp, atan2, sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.core.function_base import linspace
from numpy.lib.function_base import disp
import meshio
import os

from cardillo.model import Model
from cardillo.model.classical_beams.planar import EulerBernoulli, Hooke, Inextensible_Euler_bernoulli
from cardillo.model.bilateral_constraints.implicit import Spherical_joint2D, Rigid_connection2D, Revolute_joint2D
# from cardillo.model.bilateral_constraints.implicit import Rigid_beam_beam_connection2D as junction
from cardillo.model.scalar_force_interactions.force_laws import Linear_spring, Power_spring
from cardillo.model.scalar_force_interactions import add_rotational_forcelaw
from cardillo.solver.Newton import Newton
from cardillo.solver.EulerBackward import Euler_backward
from cardillo.solver import Generalized_alpha_1, Scipy_ivp, Generalized_alpha_4_index3, Generalized_alpha_index3_panto
from cardillo.discretization.B_spline import uniform_knot_vector
from cardillo.model.frame import Frame
from cardillo.math.algebra import A_IK_basic_z, norm2
from cardillo.utility.post_processing_vtk import post_processing
from cardillo.math import Numerical_derivative


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
       
        N, N_xi, _ = beam1.basis_functions(1)
        self.beam1_N = self.stack_shapefunctions(N, beam1.nq_el)
        self.beam1_N_xi = self.stack_shapefunctions(N_xi, beam1.nq_el)

        N, N_xi, _ = beam2.basis_functions(0)
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
       
        N, N_xi, _ = beam1.basis_functions(1)
        self.beam1_N = self.stack_shapefunctions(N, beam1.nq_el)
        self.beam1_N_xi = self.stack_shapefunctions(N_xi, beam1.nq_el)
        self.beam1_N_xi_perp = self.stack_shapefunctions_perp(N_xi, beam1.nq_el)

        N, N_xi, _ = beam2.basis_functions(0)
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
        # coo.extend(dense_num, (self.uDOF, self.qDOF))

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

if __name__ == "__main__":
    solveProblem = True
    E_B_beam = True

    # physical parameters
    gamma = pi/4
    # nRow = 20
    # nCol = 400
    nRow = 20
    nCol = 60
    nf = nRow / 2

    H = 0.0048 * 10 *sqrt(2)
    L = nCol / nRow * H
    LBeam = H / (nRow * sin(gamma))

    EA = 1.34e5 * LBeam
    EI = 1.92e-2 * LBeam
    GI = 1.59e2 * LBeam**2
    spring_power = 2

    # Yb = 500e6
    # Gb = Yb / (2 * (1 + 0.4))
    # a = 1.6e-3
    # b = 1e-3
    # rp = 0.45e-3
    # hp = 1e-3

    # Jg = (a * b**3) / 12
    
    # EA = Yb * a * b
    # EI = Yb * Jg
    # GI = Gb * 0.5*(np.pi * rp**4)/hp

    # EA = 1.6e9 * 1.6e-3 * 0.9e-3
    # EI = 1.6e9 * (1.6e-3) * (0.9e-3)**3 / 12
    # GI = 0.1 * 1/3 * 1.6e9 * np.pi * ((0.9e-3)**4)/32 * 1e3 

    # EA = 1.34e5
    # EI = 1.92e-2
    # GI = 1.59e2 * LBeam**2

    # EA = 2304
    # EI = 1.555e-4
    # GI = 0.004

    A_rho0 = 0

    displacementX_l = 0#displ #-0.0567/4
    displacementY_l = 0.0
    rotationZ_l = 0 #-np.pi/10

    displacementX_r = 0.0567 * 1.1
    displacementY_r = 0.00
    
    rotationZ_r = 0 #np.pi/10

    r_OP_l = lambda t: np.array([0, H / 2, 0]) +  np.array([t * displacementX_l, t * displacementY_l, 0])
    A_IK_l = lambda t: A_IK_basic_z(t * rotationZ_l)

    r_OP_r = lambda t: np.array([L, H / 2, 0]) +  np.array([t * displacementX_r, t * displacementY_r, 0])
    A_IK_r = lambda t: A_IK_basic_z(t * rotationZ_r)
    
    material_model = Hooke(EA, EI)

    ###################
    # create pantograph
    ###################
    p = 3
    assert p >= 2
    # nQP = int(np.ceil((p + 1)**2 / 2))
    nQP = p + 1

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
    X0 = np.linspace(0, LBeam, nNd)
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
            beams.append(EulerBernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            # beams.append(Inextensible_Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol] = ID
            ID = ID + 1
            
            # beam 2
            Q = np.concatenate([X2 + X[-1], Y2 + Y[-1]])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(EulerBernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            # beams.append(Inextensible_Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
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
            beams.append(EulerBernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            # beams.append(Inextensible_Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            model.add(beams[ID])
            ID_mat[brow, bcol] = ID
            ID = ID + 1
            # beam 2
            Q = np.concatenate([X1 + X[-1], Y1 + Y[-1]])
            q0 = np.copy(Q)
            u0 = np.zeros_like(Q)
            beams.append(EulerBernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
            # beams.append(Inextensible_Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q, q0=Q, u0=u0))
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
            if E_B_beam:
                model.add(Junction(beam1, beam2))
            else:
                model.add(Rigid_connection2D(beam1, beam2, r_OB, frame_ID1=frame_ID1, frame_ID2=frame_ID2))

            beam1 = beams[ID_mat[brow + 1, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            if E_B_beam:
                model.add(Junction(beam1, beam2))
            else:
                model.add(Rigid_connection2D(beam1, beam2, r_OB, frame_ID1=frame_ID1, frame_ID2=frame_ID2))

    # even columns
    for bcol in range(1, nCol - 1, 2):
        for brow in range(1, nRow - 1, 2):
            beam1 = beams[ID_mat[brow, bcol]]
            beam2 = beams[ID_mat[brow + 1, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            if E_B_beam:
                model.add(Junction(beam1, beam2))
            else:
                model.add(Rigid_connection2D(beam1, beam2, r_OB, frame_ID1=frame_ID1, frame_ID2=frame_ID2))

            beam1 = beams[ID_mat[brow + 1, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            if E_B_beam:
                model.add(Junction(beam1, beam2))
            else:
                model.add(Rigid_connection2D(beam1, beam2, r_OB, frame_ID1=frame_ID1, frame_ID2=frame_ID2))

    # pivots and torsional springs between beam families
            
    # internal pivots
    for brow in range(0, nRow, 2):
        for bcol in range(0, nCol - 1):
            beam1 = beams[ID_mat[brow, bcol]]
            beam2 = beams[ID_mat[brow, bcol + 1]]
            r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
            # revolute joint without shear stiffness
            # model.add(Revolute_joint2D(beam1, beam2, r_OB, np.eye(3), frame_ID1=frame_ID1, frame_ID2=frame_ID2))
            # revolute joint with shear stiffness
            # spring = Linear_spring(GI)
            spring = Power_spring(GI, spring_power)
            if E_B_beam:
                model.add(Pivot_w_spring(beam1, beam2, spring))
            else:
                model.add(add_rotational_forcelaw(spring, Revolute_joint2D)(beam1, beam2, r_OB, np.eye(3), frame_ID1=frame_ID1, frame_ID2=frame_ID2))

    # lower boundary pivots
    for bcol in range(1, nCol - 1, 2):
        beam1 = beams[ID_mat[-1, bcol]]
        beam2 = beams[ID_mat[-1, bcol + 1]]
        r_OB = beam1.r_OP(0, beam1.q0[beam1.qDOF_P(frame_ID1)], frame_ID1)
        # revolute joint without shear stiffness
        # model.add(Revolute_joint2D(beam1, beam2, r_OB, np.eye(3), frame_ID1=frame_ID1, frame_ID2=frame_ID2))
        # revolute joint with shear stiffness
        # spring = Linear_spring(GI)
        spring = Power_spring(GI, spring_power)
        if E_B_beam:
            model.add(Pivot_w_spring(beam1, beam2, spring))
        else:
            model.add(add_rotational_forcelaw(spring, Revolute_joint2D)(beam1, beam2, r_OB, np.eye(3), frame_ID1=frame_ID1, frame_ID2=frame_ID2))
        
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

    ######################
    # solve static problem
    ######################
    solver = Newton(model, n_load_steps=3, max_iter=50, tol=1.0e-2, numerical_jacobian=False)

    if solveProblem == True:
        # import cProfile, pstats
        # pr = cProfile.Profile()
        # pr.enable()
        sol = solver.solve()
        # pr.disable()

        # sortby = 'cumulative'
        # ps = pstats.Stats(pr).sort_stats(sortby)
        # ps.print_stats(0.1) # print only first 10% of the list
        save_solution(sol, f'Pantographic_sheet_{nRow}x{nCol}_statics')
    else:
        sol = load_solution(f'Pantographic_sheet_{nRow}x{nCol}_statics')

    post_processing(beams, sol.t, sol.q, f'Pantographic_sheet_{nRow}x{nCol}_statics', binary=True)
    # if statics:
    #     fig, ax = plt.subplots()
    #     ax.set_xlabel('x [m]')
    #     ax.set_ylabel('y [m]')
    #     # ax.set_xlim([-Ly + H/2 * sin(rotationZ_l), Ly*(nCol+1) + displacementX + H/2 * sin(rotationZ_r)])
    #     # ax.set_ylim([-Ly, Ly*(nRow+1) + displacementY])
    #     ax.grid(linestyle='-', linewidth='0.5')
    #     ax.set_aspect('equal')

    #     for bdy in beams:
    #         x, y, z = bdy.centerline(sol.q[-1]).T
    #         ax.plot(x, y, '-b')

    #     plt.show()
    # else:
    #     # animate configurations
    #     fig, ax = plt.subplots()
    #     ax.set_xlabel('x [m]')
    #     ax.set_ylabel('y [m]')
    #     ax.set_xlim([-Ly + H/2 * sin(rotationZ_l), Ly*(nCol+1) + displacementX_r + H/2 * sin(rotationZ_r)])
    #     ax.set_ylim([-Ly, Ly*(nRow+1) + displacementY_r])
    #     ax.grid(linestyle='-', linewidth='0.5')
    #     ax.set_aspect('equal')

    #     # prepare data for animation
    #     t = sol.t
    #     frames = len(t)
    #     target_frames = min(len(t), 100)
    #     frac = int(frames / target_frames)
    #     animation_time = 5
    #     interval = animation_time * 1000 / target_frames

    #     frames = target_frames
    #     t = t[::frac]
    #     q = sol.q[::frac]

    #     centerlines = []
    #     # lobj, = ax.plot([], [], '-k')
    #     for bdy in beams:
    #         lobj, = ax.plot([], [], '-k')
    #         centerlines.append(lobj)
            
    #     def animate(i):
    #         for idx, bdy in enumerate(beams):
    #                 # q_body = q[i][bdy.qDOF]
    #                 # r = []
    #                 # for i, xi in enumerate(xi_plt):
    #                 #     qp = q_body[bdy_qDOF_P[i]]
    #                 #     r.append(NN[i] @ qp)

    #                 # x, y = np.array(r).T
    #                 # centerlines[idx].set_data(x, y)

    #             x, y, _ = bdy.centerline(q[i], n=2).T
    #             centerlines[idx].set_data(x, y)

    #         return centerlines

    #     anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)

    #     plt.show()