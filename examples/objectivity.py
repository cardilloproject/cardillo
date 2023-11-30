import numpy as np
import matplotlib.pyplot as plt

from cardillo.math.rotations import (
    Exp_SO3,
    Log_SO3,
    Exp_SE3,
    Log_SE3,
    SE3,
    SE3inv,
    Exp_SO3_quat,
    Log_SO3_quat,
    quatprod,
    # quat_conjugate,
)


# # TODO: Add linear interpolation of dual quaternions: https://en.wikipedia.org/wiki/Dual_quaternion
# no quat_conjugate available
# def interpolate_SE3_dual_quaternion(s, r_OP0, A_IK0, r_OP1, A_IK1):
#     # compute quaternion from transformation matrix
#     q_r0 = Log_SO3_quat(A_IK0)
#     q_r1 = Log_SO3_quat(A_IK1)

#     # build pure translational quaternions
#     t0 = np.array([0, *r_OP0])
#     t1 = np.array([0, *r_OP1])

#     # build translation part of dual quaternion
#     q_t0 = 0.5 * quatprod(t0, q_r0)
#     q_t1 = 0.5 * quatprod(t1, q_r1)

#     # stack dual quaternions as tuples of R^8
#     Q0 = np.hstack((q_r0, q_t0))
#     Q1 = np.hstack((q_r1, q_t1))

#     # lineare interpolate dual quaternion
#     Q = (1 - s) * Q0 + s * Q1

#     # def dualquat_conjugate(Q):
#     #     Q_r = Q[:4]
#     #     Q_t = Q[4:]
#     #     return np.hstack((quat_conjugate(Q_r), quat_conjugate(Q_t)))

#     # def dualquat_prod(Q0, Q1):
#     #     Q_r0 = Q0[:4]
#     #     Q_r1 = Q1[:4]
#     #     Q_t0 = Q0[4:]
#     #     Q_t1 = Q1[4:]
#     #     Q_r = quatprod(Q_r0, Q_r1)
#     #     Q_t = quatprod(Q_r0, Q_t1) + quatprod(Q_t0, Q_r1)
#     #     return np.hstack((Q_r, Q_t))

#     def dual_div(z0, z1):
#         r0, d0 = z0
#         r1, d1 = z1
#         r = r0 * r1 / (r1 * r1)
#         d = (r1 * d0 - r0 * d1) / (r1 * r1)
#         return np.array([r, d])

#     def dualquat_div_dual(Q, z):
#         z0 = np.array([Q[0], Q[4]])
#         zi = np.array([Q[1], Q[5]])
#         zj = np.array([Q[2], Q[6]])
#         zk = np.array([Q[3], Q[7]])

#         z0 = dual_div(z0, z)
#         zi = dual_div(zi, z)
#         zj = dual_div(zj, z)
#         zk = dual_div(zk, z)

#         Q_r = np.array([z0[0], zi[0], zj[0], zk[0]])
#         Q_t = np.array([z0[1], zi[1], zj[1], zk[1]])

#         return np.hstack((Q_r, Q_t))

#     norm_Q_r = np.linalg.norm(Q[:4])
#     Q_z2 = np.array([norm_Q_r, (Q[:4] @ Q[4:]) / norm_Q_r])
#     # print(f"Q_z2: {Q_z2}")
#     Q = dualquat_div_dual(Q, Q_z2)

#     # norm_Q_r = np.linalg.norm(Q[:4])
#     # Q_z2 = np.array([
#     #     norm_Q_r,
#     #     (Q[:4] @ Q[4:]) / norm_Q_r
#     # ])
#     # print(f"Q_z2: {Q_z2}")

#     # extract rotation and translation parts
#     q_r = Q[:4]
#     q_t = Q[4:]
#     t = 2 * quatprod(q_t, quat_conjugate(q_r))

#     # compute corresponding transformation matrix
#     A_IK = Exp_SO3_quat(q_r)
#     # K_r_OP = t[1:]
#     # r_OP = A_IK @ K_r_OP
#     r_OP = t[1:]

#     return r_OP, A_IK


def interpolate_R3_x_R3(s, r_OP0, A_IK0, r_OP1, A_IK1):
    r_OP = (1.0 - s) * r_OP0 + s * r_OP1
    psi0 = Log_SO3(A_IK0)
    psi1 = Log_SO3(A_IK1)
    psi01 = (1.0 - s) * psi0 + s * psi1
    A_IK = Exp_SO3(psi01)
    return r_OP, A_IK


def interpolate_R3_x_R4(s, r_OP0, A_IK0, r_OP1, A_IK1):
    r_OP = (1.0 - s) * r_OP0 + s * r_OP1
    psi0 = Log_SO3_quat(A_IK0)
    psi1 = Log_SO3_quat(A_IK1)

    # # LERP
    # psi01 = (1.0 - s) * psi0 + s * psi1

    # NLERP
    psi01 = (1.0 - s) * psi0 + s * psi1
    psi01 = psi01 / np.sqrt(psi01 @ psi01)

    # # # SLERP
    # angle = np.arccos(psi0 @ psi1)
    # sa = np.sin(angle)
    # psi01 = (
    #     np.sin((1 - s) *  angle) * psi0
    #     + np.sin(s * angle) * psi1
    # ) / sa

    A_IK = Exp_SO3_quat(psi01, normalize=False)
    return r_OP, A_IK


def interpolate_R3_x_R9(s, r_OP0, A_IK0, r_OP1, A_IK1):
    r_OP = (1.0 - s) * r_OP0 + s * r_OP1
    A_IK = (1.0 - s) * A_IK0 + s * A_IK1
    return r_OP, A_IK


def interpolate_R3_x_SO3(s, r_OP0, A_IK0, r_OP1, A_IK1):
    r_OP = (1.0 - s) * r_OP0 + s * r_OP1
    psi01 = Log_SO3(A_IK0.T @ A_IK1)
    A_IK = A_IK0 @ Exp_SO3(s * psi01)
    return r_OP, A_IK


def interpolate_SE3(s, r_OP0, A_IK0, r_OP1, A_IK1):
    H_IK0 = SE3(A_IK0, r_OP0)
    H_IK1 = SE3(A_IK1, r_OP1)
    h_01 = Log_SE3(SE3inv(H_IK0) @ H_IK1)
    H_IK = H_IK0 @ Exp_SE3(s * h_01)
    r_OP = H_IK[:3, 3]
    A_IK = H_IK[:3, :3]
    return r_OP, A_IK


def plot(ax, r_OPs, A_IKs):
    ax.plot(*r_OPs.T, "-k")
    ax.quiver(*r_OPs.T, *A_IKs[:, :, 0].T, length=0.1, color="red")
    ax.quiver(*r_OPs.T, *A_IKs[:, :, 1].T, length=0.1, color="green")
    ax.quiver(*r_OPs.T, *A_IKs[:, :, 2].T, length=0.1, color="blue")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim3d(left=-1, right=1)
    ax.set_ylim3d(bottom=-1, top=1)
    ax.set_zlim3d(bottom=-1, top=1)


def comp(s, H_IK0, H_IK1, H_JI, interpolate):
    r_OP0 = H_IK0[:3, 3]
    A_IK0 = H_IK0[:3, :3]

    r_OP1 = H_IK1[:3, 3]
    A_IK1 = H_IK1[:3, :3]

    r_OP, A_IK = interpolate(s, r_OP0, A_IK0, r_OP1, A_IK1)
    H_IK = SE3(A_IK, r_OP)
    H_JK = H_JI @ H_IK

    H_JK0 = H_JI @ H_IK0
    J_r_OP0 = H_JK0[:3, 3]
    A_JK0 = H_JK0[:3, :3]

    H_JK1 = H_JI @ H_IK1
    J_r_OP1 = H_JK1[:3, 3]
    A_JK1 = H_JK1[:3, :3]

    J_r_OP, A_JK = interpolate(s, J_r_OP0, A_JK0, J_r_OP1, A_JK1)
    H_JK_bar = SE3(A_JK, J_r_OP)

    h_K_K_bar = Log_SE3(SE3inv(H_JK) @ H_JK_bar)

    return H_JK, H_JK_bar, h_K_K_bar


if __name__ == "__main__":
    # first node
    r_OP0 = np.zeros(3, dtype=float)
    # psi0 = np.zeros(3, dtype=float)
    psi0 = np.random.rand(3)
    A_IK0 = Exp_SO3(psi0)
    H_IK0 = SE3(A_IK0, r_OP0)

    # second node
    r_OP1 = np.array([1, 1, 1])
    # psi1 = np.array([0, 0, np.pi/2], dtype=float)
    # psi1 = np.array([0, np.pi/2, np.pi/2], dtype=float)
    # A_IK1 = Exp_SO3(psi1)
    # A_IK1 = Exp_SO3(np.array([0, 0, np.pi / 2])) @ Exp_SO3(np.array([0, -np.pi / 2, 0]))
    A_IK1 = Exp_SO3(np.random.rand(3))
    # A_IK1 = Exp_SO3(np.array([0, 0, np.pi / 2]))
    H_IK1 = SE3(A_IK1, r_OP1)

    # # h_K0K1 = np.array([1.5, 0, 0, 0, 0, 0], dtype=float)
    # # h_K0K1 = np.array([1.0, 0.5, 0, 0, 0, 0], dtype=float)
    # # h_K0K1 = np.array([1.0, 0.5, 0.5, 0, 0, 0], dtype=float)
    # # h_K0K1 = np.array([1.0, 0.0, 0, np.pi, 0, 0], dtype=float)
    # # h_K0K1 = np.array([1.0, 0.0, 0, 0, np.pi, 0], dtype=float)
    # h_K0K1 = np.array([1.0, 0.0, 0, 0, 0, np.pi * 0.9], dtype=float)
    # H_IK1 = H_IK0 @ Exp_SE3(h_K0K1)
    # r_OP1 = H_IK1[:3, 3]
    # A_IK1 = H_IK1[:3, :3]

    # random se(3) displacement
    # h_rigid = np.random.rand(6)
    h_rigid = np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1], dtype=float)
    # h_rigid = np.array([0.1, 0.2, 0.3, 0.1, np.pi/2, 0.1], dtype=float)
    H_JI = Exp_SE3(h_rigid)

    # H_JK0 = H_JI @ H_IK0
    # J_r_JP0 = H_JK0[:3, 3]
    # A_JK0 = H_JK0[:3, :3]

    # H_JK1 = H_JI @ H_IK1
    # J_r_JP1 = H_JK1[:3, 3]
    # A_JK1 = H_JK1[:3, :3]

    # interpoalte value in between both nodes
    # s = 0.5
    s = np.random.rand(1)[0]

    np.set_printoptions(5, suppress=True)

    H_R3_x_R3, H_bar_R3_x_R3, h_K_K_bar_R3_x_R3 = comp(
        s, H_IK0, H_IK1, H_JI, interpolate_R3_x_R3
    )
    # print(f"H_R3_x_R3:\n{H_R3_x_R3}")
    # print(f"H_bar_R3_x_R3:\n{H_bar_R3_x_R3}")
    print(f"h_K_K_bar_R3_x_R3:\n{h_K_K_bar_R3_x_R3}")
    print(f"psi_R3_x_R3: {np.linalg.norm(h_K_K_bar_R3_x_R3[3:])}")
    print(f"psi_R3_x_R3 [°]: {np.linalg.norm(h_K_K_bar_R3_x_R3[3:]) * 180 / np.pi}")

    H_R3_R4, H_bar_R3_R4, h_K_K_bar_R3_R4 = comp(
        s, H_IK0, H_IK1, H_JI, interpolate_R3_x_R4
    )
    print(f"h_K_K_bar_R3_R4:\n{h_K_K_bar_R3_R4}")

    H_R3_R9, H_bar_R3_R9, h_K_K_bar_R3_R9 = comp(
        s, H_IK0, H_IK1, H_JI, interpolate_R3_x_R9
    )
    # print(f"H_R3_R9:\n{H_R3_R9}")
    # print(f"H_bar_R3_R9:\n{H_bar_R3_R9}")
    print(f"h_K_K_bar_R3_R9:\n{h_K_K_bar_R3_R9}")

    H_R3_SO3, H_bar_R3_SO3, h_K_K_bar_R3_SO3 = comp(
        s, H_IK0, H_IK1, H_JI, interpolate_R3_x_SO3
    )
    print(f"h_K_K_bar_R3_SO3:\n{h_K_K_bar_R3_SO3}")

    H_SE3, H_bar_SE3, h_K_K_bar_SE3 = comp(s, H_IK0, H_IK1, H_JI, interpolate_SE3)
    print(f"h_K_K_bar_SE3:\n{h_K_K_bar_SE3}")

    # H_SE3_quaternion, H_bar_SE3_quaternion, h_K_K_bar_SE3_quaternion = comp(
    #     s, H_IK0, H_IK1, H_JI, interpolate_SE3_dual_quaternion
    # )
    # print(f"h_K_K_bar_SE3_quaternion:\n{h_K_K_bar_SE3_quaternion}")

    # H_R3_x_R3, H_bar_R3_x_R3 = comp(s, H_IK0, H_IK1, H_JI, interpolate_R3_x_R3)
    # print(f"H_R3_x_R3:\n{H_R3_x_R3}")
    # print(f"H_R3_x_R3:\n{H_bar_R3_x_R3}")

    # r_OP_R3_x_R3, A_IK_OP_R3_x_R3 = interpolate_R3_x_R3(s, r_OP0, A_IK0, r_OP1, A_IK1)
    # H_IK_R3_x_R3 = SE3(A_IK_OP_R3_x_R3, r_OP_R3_x_R3)
    # H_JK_R3_x_R3 = H_JI @ H_IK_R3_x_R3
    # # J_r_JP_R3_x_R3 = H_JK_R3_x_R3[:3, 3]
    # # A_JK_R3_x_R3 = H_JK_R3_x_R3[:3, :3]

    # r_OP_R3_x_R9, A_IK_OP_R3_x_R9 = interpolate_R3_x_R9(s, r_OP0, A_IK0, r_OP1, A_IK1)
    # H_IK_R3_x_R9 = SE3(A_IK_OP_R3_x_R9, r_OP_R3_x_R9)
    # H_JK_R3_x_R9 = H_JI @ H_IK_R3_x_R9
    # # J_r_JP_R3_x_R9 = H_JK_R3_x_R9[:3, 3]
    # # A_JK_R3_x_R9 = H_JK_R3_x_R9[:3, :3]

    # r_OP_R3_x_SO3, A_IK_OP_R3_x_SO3 = interpolate_R3_x_SO3(s, r_OP0, A_IK0, r_OP1, A_IK1)
    # H_IK_R3_x_SO3 = SE3(A_IK_OP_R3_x_SO3, r_OP_R3_x_SO3)
    # H_JK_R3_x_SO3 = H_JI @ H_IK_R3_x_SO3
    # # J_r_JP_R3_x_SO3 = H_JK_R3_x_SO3[:3, 3]
    # # A_JK_R3_x_SO3 = H_JK_R3_x_SO3[:3, :3]

    # r_OP_SE3, A_IK_OP_SE3 = interpolate_SE3(s, r_OP0, A_IK0, r_OP1, A_IK1)
    # H_IK_SE3 = SE3(A_IK_OP_SE3, r_OP_SE3)
    # H_JK_SE3 = H_JI @ H_IK_SE3
    # # J_r_JP_SE3 = H_JK_SE3[:3, 3]
    # # A_JK_SE3 = H_JK_SE3[:3, :3]

    if True:
        num = 10
        s = np.linspace(0, 1, num=num)

        r_OP_R3_x_R3 = np.zeros((num, 3))
        A_IK_R3_x_R3 = np.zeros((num, 3, 3))
        r_OP_R3_x_R9 = np.zeros((num, 3))
        A_IK_R3_x_R9 = np.zeros((num, 3, 3))
        r_OP_R3_x_SO3 = np.zeros((num, 3))
        A_IK_R3_x_SO3 = np.zeros((num, 3, 3))
        r_OP_SE3 = np.zeros((num, 3))
        A_IK_SE3 = np.zeros((num, 3, 3))
        for i, si in enumerate(s):
            r_OP_R3_x_R3[i], A_IK_R3_x_R3[i] = interpolate_R3_x_R3(
                si, r_OP0, A_IK0, r_OP1, A_IK1
            )
            r_OP_R3_x_R9[i], A_IK_R3_x_R9[i] = interpolate_R3_x_R9(
                si, r_OP0, A_IK0, r_OP1, A_IK1
            )
            r_OP_R3_x_SO3[i], A_IK_R3_x_SO3[i] = interpolate_R3_x_SO3(
                si, r_OP0, A_IK0, r_OP1, A_IK1
            )
            r_OP_SE3[i], A_IK_SE3[i] = interpolate_SE3(si, r_OP0, A_IK0, r_OP1, A_IK1)

        fig = plt.figure()

        ax = fig.add_subplot(2, 2, 1, projection="3d")
        ax.set_title("R3 x R3")
        plot(ax, r_OP_R3_x_R9, A_IK_R3_x_R9)

        ax = fig.add_subplot(2, 2, 2, projection="3d")
        ax.set_title("R3 x R9")
        plot(ax, r_OP_R3_x_R9, A_IK_R3_x_R9)

        ax = fig.add_subplot(2, 2, 3, projection="3d")
        ax.set_title("R3 x SO(3)")
        plot(ax, r_OP_R3_x_SO3, A_IK_R3_x_SO3)

        ax = fig.add_subplot(2, 2, 4, projection="3d")
        ax.set_title("SE(3)")
        plot(ax, r_OP_SE3, A_IK_SE3)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_title("SE(3)")
        plot(ax, r_OP_SE3, A_IK_SE3)

        plt.show()
