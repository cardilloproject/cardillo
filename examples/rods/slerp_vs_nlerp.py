import numpy as np
import matplotlib.pyplot as plt

from cardillo.math.rotations import (
    Exp_SO3,
    Log_SO3,
    Log_SE3,
    SE3,
    SE3inv,
    Exp_SO3_quat,
    Log_SO3_quat,
    T_SO3_quat,
)

def interpolate_nlerp(s, r_OP0, A_IK0, r_OP1, A_IK1):
    r_OP = (1.0 - s) * r_OP0 + s * r_OP1
    psi0 = Log_SO3_quat(A_IK0)
    psi1 = Log_SO3_quat(A_IK1)

    # NLERP
    psi01 = (1.0 - s) * psi0 + s * psi1
    A_IK = Exp_SO3_quat(psi01, normalize=True)

    K_kappa_IK = T_SO3_quat(psi01, normalize=False) @ (psi1 - psi0)
    K_kappa_IK_normalized = T_SO3_quat(psi01, normalize=True) @ (psi1 - psi0)
    return r_OP, A_IK, K_kappa_IK, K_kappa_IK_normalized

def interpolate_slerp(s, r_OP0, A_IK0, r_OP1, A_IK1):
    r_OP = (1.0 - s) * r_OP0 + s * r_OP1
    psi0 = Log_SO3_quat(A_IK0)
    psi1 = Log_SO3_quat(A_IK1)

    # SLERP
    angle = np.arccos(psi0 @ psi1)
    sa = np.sin(angle)
    psi01 = (
        np.sin((1 - s) *  angle) * psi0
        + np.sin(s * angle) * psi1
    ) / sa

    A_IK = Exp_SO3_quat(psi01, normalize=False)

    K_kappa_IK = Log_SO3(A_IK0.T @ A_IK1)

    return r_OP, A_IK, K_kappa_IK

def plot(ax, r_OPs, A_IKs, centerline_color="-k"):
    ax.plot(*r_OPs.T, centerline_color)
    ax.quiver(*r_OPs.T, *A_IKs[:, :, 0].T, length=0.1, color="red")
    ax.quiver(*r_OPs.T, *A_IKs[:, :, 1].T, length=0.1, color="green")
    ax.quiver(*r_OPs.T, *A_IKs[:, :, 2].T, length=0.1, color="blue")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim3d(left=-1, right=1)
    ax.set_ylim3d(bottom=-1, top=1)
    ax.set_zlim3d(bottom=-1, top=1)


def comp(s, H_IK0, H_IK1, interpolate1, interpolate2):
    r_OP0 = H_IK0[:3, 3]
    A_IK0 = H_IK0[:3, :3]

    r_OP1 = H_IK1[:3, 3]
    A_IK1 = H_IK1[:3, :3]

    r_OP_i1, A_IK_i1, K_kappa_IK_i1 = interpolate1(s, r_OP0, A_IK0, r_OP1, A_IK1)
    H_IK_i1 = SE3(A_IK_i1, r_OP_i1)

    print(f"\nevaluation at s: {s}")
    print(f"---------------------")
    print(f"K_kappa_nlerp: {K_kappa_IK_i1}")

    r_OP_i2, A_IK_i2, K_kappa_IK_i2 = interpolate2(s, r_OP0, A_IK0, r_OP1, A_IK1)
    H_IK_i2 = SE3(A_IK_i2, r_OP_i2)
    print(f"K_kappa_slerp: {K_kappa_IK_i2}")

    h_K_i1_K_i2 = Log_SE3(SE3inv(H_IK_i1) @ H_IK_i2)

    Delta_kappa = K_kappa_IK_i1 - K_kappa_IK_i2

    return h_K_i1_K_i2, r_OP_i1, A_IK_i1, K_kappa_IK_i1, r_OP_i2, A_IK_i2, K_kappa_IK_i2


if __name__ == "__main__":
    # first node
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = Exp_SO3(np.zeros(3, dtype=float))
    # A_IK0 = Exp_SO3(np.random.rand(3))
    H_IK0 = SE3(A_IK0, r_OP0)

    # second node
    r_OP1 = np.array([1, 1, 1])
    A_IK1 = Exp_SO3(np.array([0, 0, np.pi / 2])) @ Exp_SO3(np.array([0, -np.pi / 8, 0]))
    # A_IK1 = Exp_SO3(np.random.rand(3))
    H_IK1 = SE3(A_IK1, r_OP1)

    # interpoalte value in between both nodes
    num = 11
    s = np.linspace(0, 1, num)
    np.set_printoptions(10, suppress=True)

    r_OP_nlerp = np.zeros((num, 3))
    A_IK_nlerp = np.zeros((num, 3, 3))
    kappa_nlerp = np.zeros((3, num))
    kappa_nlerp_normalized = np.zeros((3, num))

    r_OP_slerp = np.zeros((num, 3))
    A_IK_slerp = np.zeros((num, 3, 3))
    kappa_slerp = np.zeros((3, num))

    for (i, si) in enumerate(s):
        print(f"\nevaluation at s: {si}")
        print(f"---------------------")
        
        r_OP_nlerp[i], A_IK_nlerp[i], kappa_nlerp[:, i], kappa_nlerp_normalized[:, i] = interpolate_nlerp(si, r_OP0, A_IK0, r_OP1, A_IK1)
        H_IK_nlerp = SE3(A_IK_nlerp[i], r_OP_nlerp[i])

        print(f"kappa_nlerp: {kappa_nlerp[:, i]}")
        print(f"kappa_nlerp_normalized: {kappa_nlerp_normalized[:, i]}")

        r_OP_slerp[i], A_IK_slerp[i], kappa_slerp[:, i] = interpolate_slerp(si, r_OP0, A_IK0, r_OP1, A_IK1)
        H_IK_slerp = SE3(A_IK_slerp[i], r_OP_slerp[i])
        print(f"kappa_slerp: {kappa_slerp[:, i]}")
        print(f"kappa_nlerp - kappa_slerp: {kappa_nlerp[:, i]- kappa_slerp[:, i]}")

        h_K_nlerp_K_slerp = Log_SE3(SE3inv(H_IK_nlerp) @ H_IK_slerp)

        print(f"h_nlerp_slerp:\n{h_K_nlerp_K_slerp}")
        # print(f"psi_nlerp_slerp: {np.linalg.norm(h_K_nlerp_K_slerp[3:])}")
        print(f"psi_nlerp_slerp [°]: {np.linalg.norm(h_K_nlerp_K_slerp[3:]) * 180 / np.pi}")

    print(f"\nrelation kappa_nlerp to kappa_slerp: {kappa_nlerp[: ,-1] / kappa_slerp[: ,-1]}")

    fig, ax = plt.subplots(1, 3)

    ax[0].plot(s, kappa_nlerp[0, :], s, kappa_nlerp[1, :], s, kappa_nlerp[2, :])
    ax[0].set_title("nlerp w/o normalization in T_SO3_quat")
    ax[0].set_ylim([-0.5, 1.8])
    ax[0].grid()
    ax[1].plot(s, kappa_slerp[0, :], s, kappa_slerp[1, :], s, kappa_slerp[2, :])
    ax[1].set_title("slerp")
    ax[1].set_ylim([-0.5, 1.8])
    ax[1].grid()
    ax[2].plot(s, kappa_nlerp_normalized[0, :], s, kappa_nlerp_normalized[1, :], s, kappa_nlerp_normalized[2, :])
    ax[2].set_title("nlerp with normalization in T_SO3_quat")
    ax[2].set_ylim([-0.5, 1.8])
    ax[2].grid()

    # ax[1].plot(s, kappa_i2[0, :], s, kappa_i2[1, :], s, kappa_i2[2, :])


    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("curve")
    plot(ax, r_OP_nlerp, A_IK_nlerp)
    plot(ax, r_OP_slerp, A_IK_slerp, centerline_color=":g")

    plt.show()
