import numpy as np
import matplotlib.pyplot as plt

from cardillo.beams.SE3 import Exp_SO3, Log_SO3, Exp_SE3, Log_SE3, SE3, SE3inv


def interpolate_R3_x_R3(s, r_OP0, A_IK0, r_OP1, A_IK1):
    r_OP = (1.0 - s) * r_OP0 + s * r_OP1
    psi0 = Log_SO3(A_IK0)
    psi1 = Log_SO3(A_IK1)
    psi01 = (1.0 - s) * psi0 + s * psi1
    A_IK = Exp_SO3(psi01)
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
    psi0 = np.zeros(3, dtype=float)
    A_IK0 = Exp_SO3(psi0)
    H_IK0 = SE3(A_IK0, r_OP0)

    # second node
    r_OP1 = np.array([1, 1, 1])
    # psi1 = np.array([0, 0, np.pi/2], dtype=float)
    # psi1 = np.array([0, np.pi/2, np.pi/2], dtype=float)
    # A_IK1 = Exp_SO3(psi1)
    A_IK1 = Exp_SO3(np.array([0, 0, np.pi / 2])) @ Exp_SO3(np.array([0, -np.pi / 2, 0]))
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
    s = 0.5

    np.set_printoptions(5, suppress=True)

    H_R3_x_R3, H_bar_R3_x_R3, h_K_K_bar_R3_x_R3 = comp(
        s, H_IK0, H_IK1, H_JI, interpolate_R3_x_R3
    )
    # print(f"H_R3_x_R3:\n{H_R3_x_R3}")
    # print(f"H_bar_R3_x_R3:\n{H_bar_R3_x_R3}")
    print(f"h_K_K_bar_R3_x_R3:\n{h_K_K_bar_R3_x_R3}")
    print(f"psi_R3_x_R3: {np.linalg.norm(h_K_K_bar_R3_x_R3[3:])}")
    print(f"psi_R3_x_R3 [°]: {np.linalg.norm(h_K_K_bar_R3_x_R3[3:]) * 180 / np.pi}")

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
