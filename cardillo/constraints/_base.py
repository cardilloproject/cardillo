import numpy as np


def concatenate_qDOF(object):
    qDOF1 = object.subsystem1.qDOF
    qDOF2 = object.subsystem2.qDOF
    local_qDOF1 = object.subsystem1.local_qDOF_P(object.frame_ID1)
    local_qDOF2 = object.subsystem2.local_qDOF_P(object.frame_ID2)

    object.qDOF = np.concatenate((qDOF1[local_qDOF1], qDOF2[local_qDOF2]))
    object._nq1 = nq1 = len(local_qDOF1)
    object._nq2 = len(local_qDOF2)
    object._nq = object._nq1 + object._nq2

    return local_qDOF1, local_qDOF2


def concatenate_uDOF(object):
    uDOF1 = object.subsystem1.uDOF
    uDOF2 = object.subsystem2.uDOF
    local_uDOF1 = object.subsystem1.local_uDOF_P(object.frame_ID1)
    local_uDOF2 = object.subsystem2.local_uDOF_P(object.frame_ID2)

    object.uDOF = np.concatenate((uDOF1[local_uDOF1], uDOF2[local_uDOF2]))
    object._nu1 = nu1 = len(local_uDOF1)
    object._nu2 = len(local_uDOF2)
    object._nu = object._nu1 + object._nu2

    return local_uDOF1, local_uDOF2


def auxiliary_functions(
    object,
    K1_r_P1B0,
    K2_r_P2B0,
    A_K1B0=None,
    A_K2B0=None,
):
    nq1 = object._nq1
    nu1 = object._nu1

    # auxiliary functions for subsystem 1
    object.r_OB1 = lambda t, q: object.subsystem1.r_OP(
        t, q[:nq1], object.frame_ID1, K1_r_P1B0
    )
    object.r_OB1_q1 = lambda t, q: object.subsystem1.r_OP_q(
        t, q[:nq1], object.frame_ID1, K1_r_P1B0
    )
    object.v_B1 = lambda t, q, u: object.subsystem1.v_P(
        t, q[:nq1], u[:nu1], object.frame_ID1, K1_r_P1B0
    )
    object.v_B1_q1 = lambda t, q, u: object.subsystem1.v_P_q(
        t, q[:nq1], u[:nu1], object.frame_ID1, K1_r_P1B0
    )
    object.a_B1 = lambda t, q, u, u_dot: object.subsystem1.a_P(
        t, q[:nq1], u[:nu1], u_dot[:nu1], object.frame_ID1, K1_r_P1B0
    )
    object.a_B1_q1 = lambda t, q, u, u_dot: object.subsystem1.a_P_q(
        t, q[:nq1], u[:nu1], u_dot[:nu1], object.frame_ID1, K1_r_P1B0
    )
    object.a_B1_u1 = lambda t, q, u, u_dot: object.subsystem1.a_P_u(
        t, q[:nq1], u[:nu1], u_dot[:nu1], object.frame_ID1, K1_r_P1B0
    )
    object.J_B1 = lambda t, q: object.subsystem1.J_P(
        t, q[:nq1], object.frame_ID1, K1_r_P1B0
    )
    object.J_B1_q1 = lambda t, q: object.subsystem1.J_P_q(
        t, q[:nq1], object.frame_ID1, K1_r_P1B0
    )
    object.A_IB1 = (
        lambda t, q: object.subsystem1.A_IK(t, q[:nq1], object.frame_ID1) @ A_K1B0
    )
    object.A_IB1_q1 = lambda t, q: np.einsum(
        "ijl,jk->ikl", object.subsystem1.A_IK_q(t, q[:nq1], object.frame_ID1), A_K1B0
    )
    object.Omega1 = lambda t, q, u: object.subsystem1.A_IK(
        t, q[:nq1], object.frame_ID1
    ) @ object.subsystem1.K_Omega(t, q[:nq1], u[:nu1], object.frame_ID1)
    object.Omega1_q1 = lambda t, q, u: np.einsum(
        "ijk,j->ik",
        object.subsystem1.A_IK_q(t, q[:nq1], frame_ID=object.frame_ID1),
        object.subsystem1.K_Omega(t, q[:nq1], u[:nu1], object.frame_ID1),
    ) + object.subsystem1.A_IK(
        t, q[:nq1], frame_ID=object.frame_ID1
    ) @ object.subsystem1.K_Omega_q(
        t, q[:nq1], u[:nu1], object.frame_ID1
    )

    object.Psi1 = lambda t, q, u, u_dot: object.subsystem1.A_IK(
        t, q[:nq1], object.frame_ID1
    ) @ object.subsystem1.K_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], object.frame_ID1)
    object.Psi1_q1 = lambda t, q, u, u_dot: np.einsum(
        "ijk,j->ik",
        object.subsystem1.A_IK_q(t, q[:nq1], frame_ID=object.frame_ID1),
        object.subsystem1.K_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], object.frame_ID1),
    ) + object.subsystem1.A_IK(
        t, q[:nq1], frame_ID=object.frame_ID1
    ) @ object.subsystem1.K_Psi_q(
        t, q[:nq1], u[:nu1], u_dot[:nu1], object.frame_ID1
    )
    object.Psi1_u1 = lambda t, q, u, u_dot: object.subsystem1.A_IK(
        t, q[:nq1], frame_ID=object.frame_ID1
    ) @ object.subsystem1.K_Psi_u(t, q[:nq1], u[:nu1], u_dot[:nu1], object.frame_ID1)

    object.J_R1 = lambda t, q: object.subsystem1.A_IK(
        t, q[:nq1], object.frame_ID1
    ) @ object.subsystem1.K_J_R(t, q[:nq1], object.frame_ID1)
    object.J_R1_q1 = lambda t, q: np.einsum(
        "ijk,jl->ilk",
        object.subsystem1.A_IK_q(t, q[:nq1], object.frame_ID1),
        object.subsystem1.K_J_R(t, q[:nq1], object.frame_ID1),
    ) + np.einsum(
        "ij,jkl->ikl",
        object.subsystem1.A_IK(t, q[:nq1], object.frame_ID1),
        object.subsystem1.K_J_R_q(t, q[:nq1], object.frame_ID1),
    )

    # auxiliary functions for subsystem 2
    object.r_OB2 = lambda t, q: object.subsystem2.r_OP(
        t, q[nq1:], object.frame_ID2, K2_r_P2B0
    )
    object.r_OB2_q2 = lambda t, q: object.subsystem2.r_OP_q(
        t, q[nq1:], object.frame_ID2, K2_r_P2B0
    )
    object.v_B2 = lambda t, q, u: object.subsystem2.v_P(
        t, q[nq1:], u[nu1:], object.frame_ID2, K2_r_P2B0
    )
    object.v_B2_q2 = lambda t, q, u: object.subsystem2.v_P_q(
        t, q[nq1:], u[nu1:], object.frame_ID2, K2_r_P2B0
    )
    object.a_B2 = lambda t, q, u, u_dot: object.subsystem2.a_P(
        t, q[nq1:], u[nu1:], u_dot[nu1:], object.frame_ID2, K2_r_P2B0
    )
    object.a_B2_q2 = lambda t, q, u, u_dot: object.subsystem2.a_P_q(
        t, q[nq1:], u[nu1:], u_dot[nu1:], object.frame_ID2, K2_r_P2B0
    )
    object.a_B2_u2 = lambda t, q, u, u_dot: object.subsystem2.a_P_u(
        t, q[nq1:], u[nu1:], u_dot[nu1:], object.frame_ID2, K2_r_P2B0
    )
    object.J_B2 = lambda t, q: object.subsystem2.J_P(
        t, q[nq1:], object.frame_ID2, K2_r_P2B0
    )
    object.J_B2_q2 = lambda t, q: object.subsystem2.J_P_q(
        t, q[nq1:], object.frame_ID2, K2_r_P2B0
    )
    object.A_IB2 = (
        lambda t, q: object.subsystem2.A_IK(t, q[nq1:], object.frame_ID2) @ A_K2B0
    )
    object.A_IB2_q2 = lambda t, q: np.einsum(
        "ijk,jl->ilk", object.subsystem2.A_IK_q(t, q[nq1:], object.frame_ID2), A_K2B0
    )
    object.Omega2 = lambda t, q, u: object.subsystem2.A_IK(
        t, q[nq1:], object.frame_ID2
    ) @ object.subsystem2.K_Omega(t, q[nq1:], u[nu1:], object.frame_ID2)
    object.Omega2_q2 = lambda t, q, u: np.einsum(
        "ijk,j->ik",
        object.subsystem2.A_IK_q(t, q[nq1:], frame_ID=object.frame_ID2),
        object.subsystem2.K_Omega(t, q[nq1:], u[nu1:], object.frame_ID2),
    ) + object.subsystem2.A_IK(
        t, q[nq1:], frame_ID=object.frame_ID2
    ) @ object.subsystem2.K_Omega_q(
        t, q[nq1:], u[nu1:], object.frame_ID2
    )

    object.Psi2 = lambda t, q, u, u_dot: object.subsystem2.A_IK(
        t, q[nq1:], object.frame_ID2
    ) @ object.subsystem2.K_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], object.frame_ID2)
    object.Psi2_q2 = lambda t, q, u, u_dot: np.einsum(
        "ijk,j->ik",
        object.subsystem2.A_IK_q(t, q[nq1:], frame_ID=object.frame_ID2),
        object.subsystem2.K_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], object.frame_ID2),
    ) + object.subsystem2.A_IK(
        t, q[nq1:], frame_ID=object.frame_ID2
    ) @ object.subsystem2.K_Psi_q(
        t, q[nq1:], u[nu1:], u_dot[nu1:], object.frame_ID2
    )
    object.Psi2_u2 = lambda t, q, u, u_dot: object.subsystem2.A_IK(
        t, q[nq1:], frame_ID=object.frame_ID2
    ) @ object.subsystem2.K_Psi_u(t, q[nq1:], u[nu1:], u_dot[nu1:], object.frame_ID2)

    object.J_R2 = lambda t, q: object.subsystem2.A_IK(
        t, q[nq1:], object.frame_ID2
    ) @ object.subsystem2.K_J_R(t, q[nq1:], object.frame_ID2)
    object.J_R2_q2 = lambda t, q: np.einsum(
        "ijk,jl->ilk",
        object.subsystem2.A_IK_q(t, q[nq1:], object.frame_ID2),
        object.subsystem2.K_J_R(t, q[nq1:], object.frame_ID2),
    ) + np.einsum(
        "ij,jkl->ikl",
        object.subsystem2.A_IK(t, q[nq1:], object.frame_ID2),
        object.subsystem2.K_J_R_q(t, q[nq1:], object.frame_ID2),
    )
