import numpy as np
from cardillo.math import cross3, ax2skew, ax2skew_a
from cardillo.math.approx_fprime import approx_fprime # TODO: will this function stay in cardillo?

from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey


class RigidBodyRelKinematics:
    def __init__(
        self,
        mass,
        B_Theta_C,
        joint,
        predecessor,
        xip=np.zeros(3),
        r_OC0=np.zeros(3),
        A_IB0=np.eye(3),
        max_cache_size=1,
    ):
        self.mass = mass
        self.B_Theta_C = B_Theta_C
        self.r_OC0 = r_OC0
        self.A_IB0 = A_IB0

        self.predecessor = predecessor
        self.xip = xip

        self.joint = joint

        self.is_assembled = False

        self.A_IB_cache = LRUCache(maxsize=max_cache_size)
        self.J_P_cache = LRUCache(maxsize=max_cache_size)
        self.B_Omega_cache = LRUCache(maxsize=max_cache_size)
        self.B_J_R_cache = LRUCache(maxsize=max_cache_size)

    def assembler_callback(self):
        # TODO: add this check again
        # if not self.joint.is_assembled:
        #     raise RuntimeError("Joint is not assembled; maybe not added to the model.")

        # if not self.predecessor.is_assembled:
        #     raise RuntimeError(
        #         "Predecessor is not assembled; maybe not added to the model."
        #     )

        qDOFp = self.predecessor.local_qDOF_P(self.xip)
        self.qDOF = np.concatenate([self.predecessor.qDOF[qDOFp], self.joint.qDOF])
        self.nqp = nqp = len(qDOFp)
        self.q0 = np.concatenate([self.predecessor.q0[qDOFp], self.joint.q0])
        self.__nq = nqp + self.joint.nq

        uDOFp = self.predecessor.local_uDOF_P(self.xip)
        self.uDOF = np.concatenate([self.predecessor.uDOF[uDOFp], self.joint.uDOF])
        self.nup = nup = len(uDOFp)
        self.u0 = np.concatenate([self.predecessor.u0[uDOFp], self.joint.u0])
        self.__nu = nup + self.joint.nu

        A_IBp = self.predecessor.A_IB(
            self.predecessor.t0, self.predecessor.q0[qDOFp], xi=self.xip
        )
        A_BpB1 = A_IBp.T @ self.joint.A_IB1
        A_BB2 = (
            self.A_IB0.T @ self.joint.A_IB1 @ self.joint.A_B1B2(self.t0, self.q0[nqp:])
        )
        self.A_B2B = A_BB2.T
        r_OCp = self.predecessor.r_OP(
            self.predecessor.t0, self.predecessor.q0[qDOFp], xi=self.xip
        )
        B_r_CpB1 = A_IBp.T @ (self.joint.r_OB1 - r_OCp)
        self.B_r_CB2 = self.A_IB0.T @ (
            self.joint.r_OB1
            + self.joint.A_IB1 @ self.joint.B1_r_B1B2(self.t0, self.q0[nqp:])
            - self.r_OC0
        )

        self.r_OB1 = lambda t, q: self.predecessor.r_OP(
            t, q[:nqp], self.xip, B_r_CpB1
        )
        self.r_OB1_qp = lambda t, q: self.predecessor.r_OP_q(
            t, q[:nqp], self.xip, B_r_CpB1
        )
        self.v_B1 = lambda t, q, u: self.predecessor.v_P(
            t, q[:nqp], u[:nup], self.xip, B_r_CpB1
        )
        self.v_B1_qp = lambda t, q, u: self.predecessor.v_P_q(
            t, q[:nqp], u[:nup], self.xip, B_r_CpB1
        )
        self.B1_Omegap = lambda t, q, u: A_BpB1.T @ self.predecessor.B_Omega(
            t, q[:nqp], u[:nup], self.xip
        )
        self.B1_Psip = lambda t, q, u, u_dot: A_BpB1.T @ self.predecessor.B_Psi(
            t, q[:nqp], u[:nup], u_dot[:nup], self.xip
        )
        self.a_B1 = lambda t, q, u, u_dot: self.predecessor.a_P(
            t, q[:nqp], u[:nup], u_dot[:nup], self.xip, B_r_CpB1
        )
        self.kappa_B1 = lambda t, q, u: self.predecessor.kappa_P(
            t, q[:nqp], u[:nup], self.xip, B_r_CpB1
        )
        self.kappa_B1_qp = lambda t, q, u: self.predecessor.kappa_P_q(
            t, q[:nqp], u[:nup], self.xip, B_r_CpB1
        )
        self.kappa_B1_up = lambda t, q, u: self.predecessor.kappa_P_u(
            t, q[:nqp], u[:nup], self.xip, B_r_CpB1
        )
        self.B1_kappa_Rp = lambda t, q, u: A_BpB1.T @ self.predecessor.B_kappa_R(
            t, q[:nqp], u[:nup], self.xip
        )
        self.B1_kappa_Rp_qp = lambda t, q, u: A_BpB1.T @ self.predecessor.B_kappa_R_q(
            t, q[:nqp], u[:nup], self.xip
        )
        self.B1_kappa_Rp_up = lambda t, q, u: A_BpB1.T @ self.predecessor.B_kappa_R_u(
            t, q[:nqp], u[:nup], self.xip
        )
        self.J_B1 = lambda t, q: self.predecessor.J_P(
            t, q[:nqp], self.xip, B_r_CpB1
        )
        self.J_B1_qp = lambda t, q: self.predecessor.J_P_q(
            t, q[:nqp], self.xip, B_r_CpB1
        )
        self.A_IB1 = (
            lambda t, q: self.predecessor.A_IB(t, q[:nqp], xi=self.xip) @ A_BpB1
        )
        self.A_IB1_qp = lambda t, q: np.einsum(
            "ijl,jk->ikl", self.predecessor.A_IB_q(t, q[:nqp], xi=self.xip), A_BpB1
        )
        self.B1_J_Rp = lambda t, q: A_BpB1.T @ self.predecessor.B_J_R(
            t, q[:nqp], self.xip
        )
        self.B1_J_Rp_qp = lambda t, q: np.einsum(
            "ij,jkl->ikl",
            A_BpB1.T,
            self.predecessor.B_J_R_q(t, q[:nqp], self.xip),
        )
        self.B1_Omegap_qp = lambda t, q, u: A_BpB1.T @ self.predecessor.B_Omega_q(
            t, q[:nqp], u[:nup]
        )

        self.A_B1B2 = lambda t, q: self.joint.A_B1B2(t, q[nqp:])
        self.A_B1B2_q2 = lambda t, q: self.joint.A_B1B2_q(t, q[nqp:])

        self.B1_r_B1B2 = lambda t, q: self.joint.B1_r_B1B2(t, q[nqp:])
        self.B1_r_B1B2_q2 = lambda t, q: self.joint.B1_r_B1B2_q(t, q[nqp:])
        self.B1_v_B1B2 = lambda t, q, u: self.joint.B1_v_B1B2(t, q[nqp:], u[nup:])
        self.B1_v_B1B2_q2 = lambda t, q, u: self.joint.B1_v_B1B2_q(t, q[nqp:], u[nup:])
        self.B1_J_B1B2 = lambda t, q: self.joint.B1_J_B1B2(t, q[nqp:])
        self.B1_J_B1B2_q2 = lambda t, q: self.joint.B1_J_B1B2_q(t, q[nqp:])
        self.B1_Omega_B1B2 = lambda t, q, u: self.joint.B1_Omega_B1B2(
            t, q[nqp:], u[nup:]
        )
        self.B1_Omega_B1B2_q2 = lambda t, q, u: self.joint.B1_Omega_B1B2_q(
            t, q[nqp:], u[nup:]
        )
        self.B1_a_B1B2 = lambda t, q, u, u_dot: self.joint.B1_a_B1B2(
            t, q[nqp:], u[nup:], u_dot[nup:]
        )
        self.B1_kappa_B1B2 = lambda t, q, u: self.joint.B1_kappa_B1B2(
            t, q[nqp:], u[nup:]
        )
        self.B1_kappa_B1B2_q2 = lambda t, q, u: self.joint.B1_kappa_B1B2_q(
            t, q[nqp:], u[nup:]
        )
        self.B1_kappa_B1B2_u2 = lambda t, q, u: self.joint.B1_kappa_B1B2_u(
            t, q[nqp:], u[nup:]
        )
        self.B1_Psi_B1B2 = lambda t, q, u, u_dot: self.joint.B1_Psi_B1B2(
            t, q[nqp:], u[nup:], u_dot[nup:]
        )
        self.B1_kappa_R_B1B2 = lambda t, q, u: self.joint.B1_kappa_R_B1B2(
            t, q[nqp:], u[nup:]
        )
        self.B1_kappa_R_B1B2_q2 = lambda t, q, u: self.joint.B1_kappa_R_B1B2_q(
            t, q[nqp:], u[nup:]
        )
        self.B1_kappa_R_B1B2_u2 = lambda t, q, u: self.joint.B1_kappa_R_B1B2_u(
            t, q[nqp:], u[nup:]
        )
        self.B1_J_R_B1B2 = lambda t, q: self.joint.B1_J_R_B1B2(t, q[nqp:])
        self.B1_J_R_B1B2_q2 = lambda t, q: self.joint.B1_J_R_B1B2_q(t, q[nqp:])

        self.is_assembled = True

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

    @cachedmethod(
        lambda self: self.A_IB_cache, 
        key = lambda self, t, q, xi=None: hashkey(t, tuple(q.tolist()))
    )
    def A_IB(self, t, q, xi=None):
        return self.A_IB1(t, q) @ self.A_B1B2(t, q) @ self.A_B2B

    def A_IB_q(self, t, q, xi=None):
        A_IB_q = np.zeros((3, 3, self.__nq))
        A_IB_q[:, :, : self.nqp] = np.einsum(
            "ijk,jl,lm->imk", self.A_IB1_qp(t, q), self.A_B1B2(t, q), self.A_B2B
        )
        A_IB_q[:, :, self.nqp :] = np.einsum(
            "ij,jkl,km->iml", self.A_IB1(t, q), self.A_B1B2_q2(t, q), self.A_B2B
        )
        return A_IB_q

    def r_OP(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        return (
            self.r_OB1(t, q)
            + self.A_IB1(t, q) @ self.B1_r_B1B2(t, q)
            + self.A_IB(t, q) @ (B_r_CP - self.B_r_CB2)
        )

    def r_OP_q(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        r_OP_q = np.einsum("ijk,j->ik", self.A_IB_q(t, q), (B_r_CP - self.B_r_CB2))
        r_OP_q[:, : self.nqp] += self.r_OB1_qp(t, q) + np.einsum(
            "ijk,j->ik", self.A_IB1_qp(t, q), self.B1_r_B1B2(t, q)
        )
        r_OP_q[:, self.nqp :] += self.A_IB1(t, q) @ self.B1_r_B1B2_q2(t, q)
        return r_OP_q

    def r_OP_qq(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        raise NotImplementedError

    def v_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        # v_B1 + A_IB1 B1_v_B1B2 + A_IB ( B_Omega x (B_r_CP - self.B_r_CB2) )
        v_B2 = self.v_B1(t, q, u) + self.A_IB1(t, q) @ (
            self.B1_v_B1B2(t, q, u)
            + cross3(self.B1_Omegap(t, q, u), self.B1_r_B1B2(t, q))
        )
        v_B2P = self.A_IB(t, q) @ cross3(self.B_Omega(t, q, u), B_r_CP - self.B_r_CB2)
        return v_B2 + v_B2P

    def v_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return approx_fprime(
            q,
            lambda q: self.v_P(t, q, u, xi=xi, B_r_CP=B_r_CP),
            method="3-point",
        )
    @cachedmethod(
        lambda self: self.J_P_cache, 
        key = lambda self, t, q, xi=None, B_r_CP=np.zeros(3): hashkey(t, tuple(q.tolist()), tuple(B_r_CP.tolist()))
    )
    def J_P(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        # J_P_num = np.zeros((3, self.__nu))
        # nu_P = self.v_P(t, q, np.zeros(self.__nu), B_r_CP=B_r_CP)
        # I = np.eye(self.__nu)
        # for i in range(self.__nu):
        #     J_P_num[:, i] = self.v_P(t, q, I[i], B_r_CP=B_r_CP) - nu_P
        # return J_P_num

        B_r_B2P = B_r_CP - self.B_r_CB2
        A_IB1 = self.A_IB1(t, q)
        J_P = -self.A_IB(t, q) @ ax2skew(B_r_B2P) @ self.B_J_R(t, q)
        J_P[:, : self.nup] += self.J_B1(t, q) - A_IB1 @ ax2skew(
            self.B1_r_B1B2(t, q)
        ) @ self.B1_J_Rp(t, q)
        J_P[:, self.nup :] += A_IB1 @ self.B1_J_B1B2(t, q)
        return J_P

    def J_P_q(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        # return Numerical_derivative(lambda t, q: self.J_P(t, q, B_r_CP=B_r_CP))._x(t, q)

        B_r_B2P = B_r_CP - self.B_r_CB2
        A_IB1 = self.A_IB1(t, q)
        B1_r_B1B2 = self.B1_r_B1B2(t, q)
        A_IB1_qp = self.A_IB1_qp(t, q)
        B1_J_Rp = self.B1_J_Rp(t, q)

        J_P_q = np.einsum(
            "ij,jkl->ikl", -self.A_IB(t, q) @ ax2skew(B_r_B2P), self.B_J_R_q(t, q)
        )
        J_P_q -= np.einsum(
            "ijk,jl->ilk", self.A_IB_q(t, q), ax2skew(B_r_B2P) @ self.B_J_R(t, q)
        )
        J_P_q[:, : self.nup, : self.nqp] += (
            self.J_B1_qp(t, q)
            - np.einsum(
                "ij,jkl->ikl",
                A_IB1 @ ax2skew(B1_r_B1B2),
                self.B1_J_Rp_qp(t, q),
            )
            - np.einsum(
                "ijl,jk->ikl",
                A_IB1_qp,
                ax2skew(B1_r_B1B2) @ B1_J_Rp,
            )
        )

        J_P_q[:, : self.nup, self.nqp :] -= np.einsum(
            "ijk,km,jl->ilm",
            A_IB1 @ ax2skew_a(),
            self.B1_r_B1B2_q2(t, q),
            B1_J_Rp,
        )

        J_P_q[:, self.nup :, : self.nqp] += np.einsum(
            "ijk,jl->ilk", A_IB1_qp, self.B1_J_B1B2(t, q)
        )
        J_P_q[:, self.nup :, self.nqp :] += np.einsum(
            "ij,jlk->ilk", A_IB1, self.B1_J_B1B2_q2(t, q)
        )

        return J_P_q

    def a_P(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3)):
        B_r_B2P = B_r_CP - self.B_r_CB2
        B1_r_B1B2 = self.B1_r_B1B2(t, q)
        B1_Omegap = self.B1_Omegap(t, q, u)
        B_Omega = self.B_Omega(t, q, u)

        a_B2 = self.a_B1(t, q, u, u_dot) + self.A_IB1(t, q) @ (
            self.B1_a_B1B2(t, q, u, u_dot)
            + cross3(self.B1_Psip(t, q, u, u_dot), B1_r_B1B2)
            + cross3(
                B1_Omegap, 2 * self.B1_v_B1B2(t, q, u) + cross3(B1_Omegap, B1_r_B1B2)
            )
        )
        a_B2P = self.A_IB(t, q) @ (
            cross3(self.B_Psi(t, q, u, u_dot), B_r_B2P)
            + cross3(B_Omega, cross3(B_Omega, B_r_B2P))
        )
        return a_B2 + a_B2P

    def a_P_q(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3)):
        return approx_fprime(
            q,
            lambda q: self.a_P(t, q, u, u_dot, xi=xi, B_r_CP=B_r_CP),
            method="3-point",
        )

    def a_P_u(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3)):
        return approx_fprime(
            u,
            lambda u: self.a_P(t, q, u, u_dot, xi=xi, B_r_CP=B_r_CP),
            method="3-point",
        )

    def kappa_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        # return self.a_P(t, q, u, np.zeros(self.__nu), B_r_CP=B_r_CP)

        B_r_B2P = B_r_CP - self.B_r_CB2
        B1_r_B1B2 = self.B1_r_B1B2(t, q)
        B1_Omegap = self.B1_Omegap(t, q, u)
        B_Omega = self.B_Omega(t, q, u)

        kappa_P = self.kappa_B1(t, q, u) + self.A_IB1(t, q) @ (
            self.B1_kappa_B1B2(t, q, u)
            + cross3(self.B1_kappa_Rp(t, q, u), B1_r_B1B2)
            + cross3(
                B1_Omegap, 2 * self.B1_v_B1B2(t, q, u) + cross3(B1_Omegap, B1_r_B1B2)
            )
        )
        kappa_P += self.A_IB(t, q) @ (
            cross3(self.B_kappa_R(t, q, u), B_r_B2P)
            + cross3(B_Omega, cross3(B_Omega, B_r_B2P))
        )

        return kappa_P

    def kappa_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        # raise RuntimeError("Some terms are still missing!")
        B_r_B2P = B_r_CP - self.B_r_CB2
        B_Omega = self.B_Omega(t, q, u)
        B_Omega_q = self.B_Omega_q(t, q, u)
        B1_r_B1B2 = self.B1_r_B1B2(t, q)
        B1_r_B1B2_q2 = self.B1_r_B1B2_q2(t, q)
        B1_Omegap = self.B1_Omegap(t, q, u)
        B1_Omegap_qp = self.B1_Omegap_qp(t, q, u)
        B1_kappa_Rp = self.B1_kappa_Rp(t, q, u)
        A_IB1 = self.A_IB1(t, q)

        tmp1 = cross3(self.B_kappa_R(t, q, u), B_r_B2P)
        tmp1_q = -ax2skew(B_r_B2P) @ self.B_kappa_R_q(t, q, u)
        tmp2 = cross3(B_Omega, cross3(B_Omega, B_r_B2P))
        tmp2_q = (
            -(ax2skew(cross3(B_Omega, B_r_B2P)) + ax2skew(B_Omega) @ ax2skew(B_r_B2P))
            @ B_Omega_q
        )

        kappa_P_q = np.einsum("ijk,j->ik", self.A_IB_q(t, q), tmp1 + tmp2) + self.A_IB(
            t, q
        ) @ (tmp1_q + tmp2_q)

        tmp4 = 2 * self.B1_v_B1B2(t, q, u) + cross3(B1_Omegap, B1_r_B1B2)
        tmp3 = (
            self.B1_kappa_B1B2(t, q, u)
            + cross3(B1_kappa_Rp, B1_r_B1B2)
            + cross3(B1_Omegap, tmp4)
        )
        tmp3_qp = (
            -ax2skew(B1_r_B1B2) @ self.B1_kappa_Rp_qp(t, q, u)
            - (ax2skew(tmp4) + ax2skew(B1_Omegap) @ ax2skew(B1_r_B1B2)) @ B1_Omegap_qp
        )
        tmp3_q2 = (
            self.B1_kappa_B1B2_q2(t, q, u)
            + ax2skew(B1_kappa_Rp) @ B1_r_B1B2_q2
            + ax2skew(B1_Omegap)
            @ (2 * self.B1_v_B1B2_q2(t, q, u) + ax2skew(B1_Omegap) @ B1_r_B1B2_q2)
        )

        kappa_P_q[:, : self.nqp] += (
            self.kappa_B1_qp(t, q, u)
            + np.einsum("ijk,j->ik", self.A_IB1_qp(t, q), tmp3)
            + A_IB1 @ tmp3_qp
        )
        kappa_P_q[:, self.nqp :] += A_IB1 @ tmp3_q2
        return kappa_P_q

    def kappa_P_u(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        B_r_B2P = B_r_CP - self.B_r_CB2
        B_Omega = self.B_Omega(t, q, u)
        B_Omega_u = self.B_J_R(t, q)
        B1_r_B1B2 = self.B1_r_B1B2(t, q)
        B1_Omegap = self.B1_Omegap(t, q, u)
        A_IB1 = self.A_IB1(t, q)

        tmp1_u = -ax2skew(B_r_B2P) @ self.B_kappa_R_u(t, q, u)
        tmp2_u = (
            -(ax2skew(cross3(B_Omega, B_r_B2P)) + ax2skew(B_Omega) @ ax2skew(B_r_B2P))
            @ B_Omega_u
        )
        # kappa_P = self.kappa_B1(t, q, u) + A_IB1 @ (
        #     self.B1_kappa_B1B2(t, q, u)
        #     + cross3(self.B1_kappa_Rp(t, q, u), B1_r_B1B2)
        #     + cross3(
        #         B1_Omegap, 2 * self.B1_v_B1B2(t, q, u) + cross3(B1_Omegap, B1_r_B1B2)
        #     )
        # )
        # tmp3 = (
        #     self.B1_kappa_B1B2(t, q, u)
        #     + cross3(self.B1_kappa_Rp(t, q, u), B1_r_B1B2)
        #     + cross3(
        #         B1_Omegap, 2 * self.B1_v_B1B2(t, q, u) + cross3(B1_Omegap, B1_r_B1B2)
        #     )
        # )
        tmp3_up = -ax2skew(B1_r_B1B2) @ self.B1_kappa_Rp_up(t, q, u) - (
            ax2skew(B1_Omegap) @ ax2skew(B1_r_B1B2)
            + ax2skew(2 * self.B1_v_B1B2(t, q, u) + cross3(B1_Omegap, B1_r_B1B2))
        ) @ self.B1_J_Rp(t, q)
        tmp3_u2 = self.B1_kappa_B1B2_u2(t, q, u) + 2 * ax2skew(
            B1_Omegap
        ) @ self.B1_J_B1B2(t, q)
        kappa_P_u = self.A_IB(t, q) @ (tmp1_u + tmp2_u)
        kappa_P_u[:, : self.nup] += self.kappa_B1_up(t, q, u) + A_IB1 @ tmp3_up
        kappa_P_u[:, self.nup :] += A_IB1 @ tmp3_u2
        return kappa_P_u
    
    @cachedmethod(
        lambda self: self.B_Omega_cache, 
        key = lambda self, t, q, u, xi=None: hashkey(t, tuple(q.tolist()), tuple(u.tolist()))
    )
    def B_Omega(self, t, q, u, xi=None):
        return (
            (self.B1_Omegap(t, q, u) + self.B1_Omega_B1B2(t, q, u))
            @ self.A_B1B2(t, q)
            @ self.A_B2B
        )

    def B_Omega_q(self, t, q, u, xi=None):
        # return Numerical_derivative(self.B_Omega)._x(t, q, u)
        B_BB1 = (self.A_B1B2(t, q) @ self.A_B2B).T
        B_BB1_q2 = np.einsum("ijk,jl->lik", self.A_B1B2_q2(t, q), self.A_B2B)
        B1_Omega_pB2 = self.B1_Omegap(t, q, u) + self.B1_Omega_B1B2(t, q, u)

        B_Omega_q = np.zeros((3, self.__nq))
        B_Omega_q[:, : self.nqp] = B_BB1 @ self.B1_Omegap_qp(t, q, u)
        B_Omega_q[:, self.nqp :] = np.einsum(
            "ijk,j->ik", B_BB1_q2, B1_Omega_pB2
        ) + B_BB1 @ self.B1_Omega_B1B2_q2(t, q, u)

        return B_Omega_q
   
    @cachedmethod(
        lambda self: self.B_J_R_cache, 
        key = lambda self, t, q, xi=None: hashkey(t, tuple(q.tolist()))
    )
    def B_J_R(self, t, q, xi=None):
        # B_J_R = np.zeros((3, self.__nu))
        # nu_R = self.B_Omega(t, q, np.zeros(self.__nu))
        # I = np.eye(self.__nu)
        # for i in range(self.__nu):
        #     B_J_R[:, i] = self.B_Omega(t, q, I[i]) - nu_R
        # return B_J_R

        B_J_R = np.zeros((3, self.__nu))
        B_BB1 = (self.A_B1B2(t, q) @ self.A_B2B).T
        B_J_R[:, : self.nup] = B_BB1 @ self.B1_J_Rp(t, q)
        B_J_R[:, self.nup :] = B_BB1 @ self.B1_J_R_B1B2(t, q)

        return B_J_R

    def B_J_R_q(self, t, q, xi=None):
        # return Numerical_derivative(lambda t, q: self.B_J_R(t, q))._x(t, q)

        B_J_R_q = np.zeros((3, self.__nu, self.__nq))

        B_BB1 = (self.A_B1B2(t, q) @ self.A_B2B).T
        B_BB1_q2 = np.einsum("ijk,jl->lik", self.A_B1B2_q2(t, q), self.A_B2B)

        B_J_R_q[:, : self.nup, : self.nqp] = np.einsum(
            "ij,jkl->ikl", B_BB1, self.B1_J_Rp_qp(t, q)
        )
        B_J_R_q[:, : self.nup, self.nqp :] = np.einsum(
            "ijk,jl->ilk", B_BB1_q2, self.B1_J_Rp(t, q)
        )
        B_J_R_q[:, self.nup :, self.nqp :] = np.einsum(
            "ijk,jl->ilk", B_BB1_q2, self.B1_J_R_B1B2(t, q)
        ) + np.einsum("ij,jkl->ikl", B_BB1, self.B1_J_R_B1B2_q2(t, q))
        return B_J_R_q

    def B_Psi(self, t, q, u, u_dot, xi=None):
        return (
            (self.B1_Psip(t, q, u, u_dot) + self.B1_Psi_B1B2(t, q, u, u_dot))
            @ self.A_B1B2(t, q)
            @ self.A_B2B
        )

    def B_kappa_R(self, t, q, u, xi=None):
        # return self.B_Psi(t, q, u, np.zeros(self.__nu))
        return (
            (self.B1_kappa_Rp(t, q, u) + self.B1_kappa_R_B1B2(t, q, u))
            @ self.A_B1B2(t, q)
            @ self.A_B2B
        )

    def B_kappa_R_q(self, t, q, u, xi=None):
        B_kappa_R_q = np.zeros((3, self.__nq))
        B_BB1 = (self.A_B1B2(t, q) @ self.A_B2B).T
        B_BB1_q2 = np.einsum("ijk,jl->lik", self.A_B1B2_q2(t, q), self.A_B2B)

        B_kappa_R_q[:, : self.nqp] = B_BB1 @ self.B1_kappa_Rp_qp(t, q, u)
        B_kappa_R_q[:, self.nqp :] = B_BB1 @ self.B1_kappa_R_B1B2_q2(
            t, q, u
        ) + np.einsum(
            "ijk,j->ik",
            B_BB1_q2,
            self.B1_kappa_Rp(t, q, u) + self.B1_kappa_R_B1B2(t, q, u),
        )

        return B_kappa_R_q

    def B_kappa_R_u(self, t, q, u, xi=None):
        B_BB1 = (self.A_B1B2(t, q) @ self.A_B2B).T
        B_kappa_R_u = np.zeros((3, self.__nu))
        B_kappa_R_u[:, : self.nup] = B_BB1 @ self.B1_kappa_Rp_up(t, q, u)
        B_kappa_R_u[:, self.nup :] = B_BB1 @ self.B1_kappa_R_B1B2_u2(t, q, u)
        return B_kappa_R_u
