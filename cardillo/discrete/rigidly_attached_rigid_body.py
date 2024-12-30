import numpy as np
from cardillo.math import cross3, ax2skew


class RigidlyAttachedRigidBody:
    def __init__(
        self,
        mass,
        B_Theta_C,
        body,
        xi=np.zeros(3),
        r_OC0=np.zeros(3),
        A_IB0=np.eye(3),
    ):
        self.mass = mass
        self.B_Theta_C = B_Theta_C
        self.r_OC0 = r_OC0
        self.A_IB0 = A_IB0

        self.parent = body
        self.xip = xi

    def assembler_callback(self):

        qDOFp = self.parent.local_qDOF_P(self.xip)
        self.qDOF =self.parent.qDOF[qDOFp]
        # self.nqp = nqp = len(qDOFp)
        self.q0 = self.parent.q0[qDOFp]
        self.__nq = len(self.qDOF) 

        uDOFp = self.parent.local_uDOF_P(self.xip)
        self.uDOF = self.parent.uDOF[uDOFp]
        # self.nup = nup = len(uDOFp)
        self.u0 = self.parent.u0[uDOFp]
        self.__nu = len(self.uDOF)

        # initial orientation of parent
        A_IBp0 = self.parent.A_IB(
            self.parent.t0, self.parent.q0[qDOFp], xi=self.xip
        )
        # relative orientation between this body (B) and parent (Bp) (constant)
        self.A_BpB = A_IBp0.T @ self.A_IB0
        # initial position of the c.o.m. of the parent
        r_OCp0 = self.parent.r_OP(self.parent.t0, self.parent.q0[qDOFp], xi=self.xip)
        # relative position of the c.o.m. (C) of this body w.r.t. the c.o.m. (Cp) of its parent 
        self.B_r_CpC = (A_IBp0 @ self.A_BpB).T @ (self.r_OC0 - r_OCp0)

    def M(self, t, q):
        J_C = self.J_P(t, q)
        B_J_R = self.B_J_R(t, q)
        return self.mass * J_C.T @ J_C + B_J_R.T @ self.B_Theta_C @ B_J_R

    def Mu_q(self, t, q, u):
        J_C = self.J_P(t, q)
        B_J_R = self.B_J_R(t, q)
        J_C_q = self.J_P_q(t, q)
        B_J_R_q = self.B_J_R_q(t, q)

        return (
            np.einsum("ijl,ik,k->jl", J_C_q, J_C, self.mass * u)
            + np.einsum("ij,ikl,k->jl", J_C, J_C_q, self.mass * u)
            + np.einsum("ijl,ik,k->jl", B_J_R_q, self.B_Theta_C @ B_J_R, u)
            + np.einsum("ij,jkl,k->il", B_J_R.T @ self.B_Theta_C, B_J_R_q, u)
        )

    def h(self, t, q, u):
        Omega = self.B_Omega(t, q, u)
        return -(
            self.mass * self.J_P(t, q).T @ self.kappa_P(t, q, u)
            + self.B_J_R(t, q).T
            @ (
                self.B_Theta_C @ self.B_kappa_R(t, q, u)
                + cross3(Omega, self.B_Theta_C @ Omega)
            )
        )

    def h_q(self, t, q, u):
        Omega = self.B_Omega(t, q, u)
        Omega_q = self.B_Omega_q(t, q, u)
        J_P_q = self.J_P_q(t, q)
        tmp1 = self.B_Theta_C @ self.B_kappa_R(t, q, u)
        tmp1_q = self.B_Theta_C @ self.B_kappa_R_q(t, q, u)
        tmp2 = cross3(Omega, self.B_Theta_C @ Omega)
        tmp2_q = (
            ax2skew(Omega) @ self.B_Theta_C - ax2skew(self.B_Theta_C @ Omega)
        ) @ Omega_q

        f_gyr_q = -(
            np.einsum("jik,j->ik", J_P_q, self.mass * self.kappa_P(t, q, u))
            + self.mass * self.J_P(t, q).T @ self.kappa_P_q(t, q, u)
            + np.einsum("jik,j->ik", J_P_q, tmp1 + tmp2)
            + self.B_J_R(t, q).T @ (tmp1_q + tmp2_q)
        )
        return f_gyr_q

    def h_u(self, t, q, u):
        Omega = self.B_Omega(t, q, u)
        Omega_u = self.B_J_R(t, q)
        tmp1_u = self.B_Theta_C @ self.B_kappa_R_u(t, q, u)
        tmp2_u = (
            ax2skew(Omega) @ self.B_Theta_C - ax2skew(self.B_Theta_C @ Omega)
        ) @ Omega_u

        f_gyr_u = -(
            self.mass * self.J_P(t, q).T @ self.kappa_P_u(t, q, u)
            + self.B_J_R(t, q).T @ (tmp1_u + tmp2_u)
        )
        return f_gyr_u

        # f_gyr_u_num = Numerical_derivative(self.f_gyr, order=2)._y(t, q, u)
        # print(f'f_gyr_u error = {np.linalg.norm(f_gyr_u - f_gyr_u_num)}')

    #########################################
    # helper functions
    #########################################

    def local_qDOF_P(self, xi=None):
        return np.arange(self.__nq)

    def local_uDOF_P(self, xi=None):
        return np.arange(self.__nu)

    def A_IB(self, t, q, xi=None):
        return self.parent.A_IB(t, q, xi=self.xip) @ self.A_BpB

    def A_IB_q(self, t, q, xi=None):
        return  np.einsum(
            "ijk,jl->ilk", self.parent.A_IB(t, q, xi=self.xip),  self.A_BpB
        )

    def r_OP(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        return self.parent.r_OP(
            t, q, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )

    def r_OP_q(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        return self.parent.r_OP_q(
            t, q, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )
    
    def v_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return self.parent.v_P(
            t, q, u, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )
    def v_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return self.parent.v_P_q(
            t, q, u, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )

    def J_P(self, t, q, xi=None, B_r_CP=np.zeros(3)):

        return self.parent.J_P(
            t, q, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )

    def J_P_q(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        return self.parent.J_P_q(
            t, q, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )

    def a_P(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3)):
        return self.parent.a_P(
            t, q, u, u_dot, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )

    def a_P_q(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3)):
        return self.parent.a_P_q(
            t, q, u, u_dot, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )

    def a_P_u(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3)):
        return self.parent.a_P_u(
            t, q, u, u_dot, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )

    def kappa_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return self.parent.kappa_P(
            t, q, u, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )

    def kappa_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return self.parent.kappa_P_q(
            t, q, u, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )
    def kappa_P_u(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return self.parent.kappa_P_u(
            t, q, u, self.xip, B_r_CP=self.A_BpB @ (self.B_r_CpC + B_r_CP)
        )
    
    def B_Omega(self, t, q, u, xi=None):
        return self.A_BpB.T @ self.parent.B_Omega(t, q, u, self.xip)

    def B_Omega_q(self, t, q, u, xi=None):
        return  np.einsum("ij,jkl->ikl", self.A_BpB.T, self.parent.B_Omega_q(t, q, u, self.xip))
        
    def B_J_R(self, t, q, xi=None):
        return self.A_BpB.T @ self.parent.B_J_R(t, q, self.xip)

    def B_J_R_q(self, t, q, xi=None):
        return  np.einsum("ij,jkl->ikl", self.A_BpB.T, self.parent.B_J_R_q(t, q, self.xip))
        
    def B_Psi(self, t, q, u, u_dot, xi=None):
        return self.A_BpB.T @ self.parent.B_Psi(t, q, u, u_dot, self.xip)

    def B_kappa_R(self, t, q, u, xi=None):
        # return self.B_Psi(t, q, u, np.zeros(self.__nu))
        return self.A_BpB.T @ self.parent.B_kappa_R(t, q, u, self.xip)

    def B_kappa_R_q(self, t, q, u, xi=None):
        return np.einsum("ij,jkl->ikl", self.A_BpB.T, self.parent.B_kappa_R_q(t, q, u, self.xip))
        

    def B_kappa_R_u(self, t, q, u, xi=None):
        return np.einsum("ij,jkl->ikl", self.A_BpB.T, self.parent.B_kappa_R_u(t, q, u, self.xip))
        
