import numpy as np
from cardillo.beams.spatial.SE3 import Exp_SO3

from cardillo.beams.spatial.material_models import Simo1986, ShearStiffQuadratic
from cardillo.beams.spatial.cross_section import CircularCrossSection
from cardillo.beams.spatial import DirectorAxisAngle, Kirchhoff, KirchhoffSingularity
from cardillo.beams import animate_beam

from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import (
    RigidConnection,
    SphericalJoint,
)

from cardillo.forces import Force, K_Moment, DistributedForce1DBeam
from cardillo.model import System
from cardillo.solver import Newton, ScipyIVP

from cardillo.math import e1, e2, e3, ax2skew, inv3D, cross3, A_IK_basic
from math import pi

import matplotlib.pyplot as plt

use_Kirchhoff = True
# use_Kirchhoff = False


def statics():
    # nelements = 5
    # basis_r = "Lagrange"
    # basis_psi = "Lagrange"
    # polynomial_degree_r = 1
    # polynomial_degree_psi = 1

    # nelements = 2
    # basis_r = "Lagrange"
    # basis_psi = "Lagrange"
    # polynomial_degree_r = 2
    # polynomial_degree_psi = 2

    # nelements = 2
    # basis_r = "Lagrange"
    # basis_psi = "Lagrange"
    # polynomial_degree_r = 3
    # polynomial_degree_psi = 3

    # nelements = 10
    # basis_r = "B-spline"
    # basis_psi = "B-spline"
    # polynomial_degree_r = 3
    # polynomial_degree_psi = 3

    # nelements = 3
    # basis_r = "Hermite"
    # basis_psi = "Lagrange"
    # polynomial_degree_r = 3
    # polynomial_degree_psi = 3

    # test for Kirchhoff beam
    polynomial_degree_r = 3
    polynomial_degree_psi = 1
    nelements = 4

    # beam parameters
    L = 10
    EA = GA = 1.0e2
    GJ = EI = 1.0e2

    # Note: This is never used in statics!
    line_density = 1.0
    radius = 1.0
    cross_section = CircularCrossSection(line_density, radius)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    # # build quadratic material model
    # Ei = np.array([EA, GA, GA], dtype=float)
    # Fi = np.array([GJ, EI, EI], dtype=float)
    # material_model = Simo1986(Ei, Fi)

    # Q = DirectorAxisAngle.straight_configuration(
    #     polynomial_degree_r,
    #     polynomial_degree_psi,
    #     basis_r,
    #     basis_psi,
    #     nelements,
    #     L,
    #     r_OP=r_OP0,
    #     A_IK=A_IK0,
    # )
    # beam = DirectorAxisAngle(
    #     material_model,
    #     A_rho0,
    #     K_S_rho0,
    #     K_I_rho0,
    #     polynomial_degree_r,
    #     polynomial_degree_psi,
    #     nelements,
    #     Q,
    #     basis_r=basis_r,
    #     basis_psi=basis_psi,
    # )

    # build quadratic material model
    Ei = np.array([EA, GA, GA], dtype=float)
    Fi = np.array([GJ, EI, EI], dtype=float)
    if use_Kirchhoff:
        material_model = ShearStiffQuadratic(EA, Fi)
    else:
        material_model = Simo1986(Ei, Fi)

    nquadrature = int(max(polynomial_degree_r, polynomial_degree_psi)) + 1

    Q = Kirchhoff.straight_configuration(nelements, L)
    beam = Kirchhoff(
        material_model,
        A_rho0,
        K_I_rho0,
        nquadrature,
        nelements,
        Q,
        use_Kirchhoff=use_Kirchhoff,
    )

    # Q = KirchhoffSingularity.straight_configuration(nelements, L)
    # beam = KirchhoffSingularity(
    #     material_model,
    #     A_rho0,
    #     K_I_rho0,
    #     nquadrature,
    #     nelements,
    #     Q,
    #     use_Kirchhoff=use_Kirchhoff,
    # )

    # # junctions
    # n = 1
    # # t_star = 0.25
    # t_star = 0.1

    # def A_IK0(t):
    #     # phi = (
    #     #     n * np.heaviside(t - t_star, 1.0) * (t - t_star) / (1.0 - t_star) * 2.0 * pi
    #     # )
    #     # phi = t * n * 2.0 * pi
    #     phi = t * pi
    #     # return A_IK_basic(phi).x()
    #     # return A_IK_basic(phi).y()
    #     return A_IK_basic(phi).z()
    #     # return A_IK_basic(phi).z() @ A_IK_basic(phi).x()

    r_OP0 = np.random.rand(3)
    frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, frame_ID2=(0,))

    # moment at right end
    Fi = material_model.Fi
    # M = lambda t: t * 2 * np.pi * (Fi[0] * e1 + Fi[2] * e3) / L * 0.45
    # M = lambda t: t * 2 * np.pi * Fi[0] * e1 / L * 0.45
    M = lambda t: t * 2 * np.pi * Fi[1] * e2 / L * 0.45
    # M = lambda t: t * 2 * np.pi * Fi[2] * e3 / L  * 0.45
    moment = K_Moment(M, beam, (1,))

    # assemble the model
    model = System()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    model.add(moment)
    model.assemble()

    n_load_steps = 10

    solver = Newton(
        model,
        n_load_steps=n_load_steps,
        max_iter=15,
        atol=1.0e-6,
    )
    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    animate_beam(t, q, [beam], L, show=True)
    # animate_beam(t, [q[-1]], [beam], L, show=True)


def dynamics():
    nelements = 10
    polynomial_degree = 1

    # beam parameters
    L = 1
    EA = GA = 1.0e3
    GJ = EI = 1.0e2

    # build quadratic material model
    Ei = np.array([EA, GA, GA], dtype=float)
    Fi = np.array([GJ, EI, EI], dtype=float)
    material_model = Simo1986(Ei, Fi)

    # Note: This is never used in statics!
    line_density = 1.0
    radius = 1.0
    cross_section = CircularCrossSection(line_density, radius)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    Q = DirectorAxisAngle.straight_configuration(
        polynomial_degree,
        nelements,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )

    beam = DirectorAxisAngle(
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        nelements,
        Q,
    )

    # junctions
    A_IK0 = lambda t: Exp_SO3(t * 0.5 * np.pi * np.sin(2 * np.pi * t) * e3)
    frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, frame_ID2=(0,))

    # # moment at right end
    # Fi = material_model.Fi
    # M = lambda t: t**2 * 2 * np.pi * (Fi[0] * e1 + Fi[2] * e3) / L * 0.1
    # moment = Moment(M, beam, (1,))

    # F = -1.0e0 * e3
    # force = Force(F, beam, (1,))

    # assemble the model
    model = System()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    # model.add(moment)
    # model.add(force)
    model.assemble()

    t1 = 1
    dt = 1.0e-2
    # method = "RK45"
    method = "RK23"
    rtol = 1.0e-3
    atol = 1.0e-3

    solver = ScipyIVP(model, t1, dt, method=method, rtol=rtol, atol=atol)
    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    animate_beam(t, q, [beam], L, show=True)


def HeavyTop():
    class RigidHeavyTopODE:
        def __init__(self, m, r, l, g):
            self.m = m
            self.r = r
            self.l = l
            self.g = g
            self.K_r_PS = np.array([l / 2, 0, 0], dtype=float)
            self.A = (1.0 / 2.0) * m * r**2
            self.B = (1.0 / 4.0) * m * r**2 + (1.0 / 12.0) * m * l**2
            self.K_Theta_S = np.diag([self.A, self.B, self.B])
            self.K_Theta_P = (
                self.K_Theta_S + m * ax2skew(self.K_r_PS) @ ax2skew(self.K_r_PS).T
            )
            self.K_Theta_P_inv = inv3D(self.K_Theta_P)

        def A_IK(self, t, q):
            alpha, beta, gamma = q
            # z, y, x
            # A_IB = A_IK_basic(alpha).z()
            # A_BC = A_IK_basic(beta).y()
            # A_CK = A_IK_basic(gamma).x()
            # return A_IB @ A_BC @ A_CK
            sa, ca = np.sin(alpha), np.cos(alpha)
            sb, cb = np.sin(beta), np.cos(beta)
            sg, cg = np.sin(gamma), np.cos(gamma)
            # fmt: off
            return np.array([
                [ca * cb, ca * sb  *sg - sa * cg, ca * sb * cg + sa * sg],
                [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
                [    -sb,                cb * sg,                cb * cg],
            ])
            # fmt: on

        def r_OP(self, t, q, K_r_SP=np.zeros(3, dtype=float)):
            A_IK = self.A_IK(t, q)
            r_OS = A_IK @ self.K_r_PS
            return r_OS + A_IK @ K_r_SP

        def __call__(self, t, x):
            dx = np.zeros(6, dtype=float)
            alpha, beta, gamma = x[:3]
            K_Omega = x[3:]

            m = self.m
            l = self.l
            g = self.g
            A = self.A
            B = self.B

            sb, cb = np.sin(beta), np.cos(beta)
            sg, cg = np.sin(gamma), np.cos(gamma)

            # z, y, x
            # fmt: off
            Q = np.array([
                [    -sb, 0.0, 1.0], 
                [cb * sg,  cg, 0.0], 
                [cb * cg, -sg, 0.0]
            ], dtype=float)
            # fmt: on

            f_gyr = cross3(K_Omega, self.K_Theta_P @ K_Omega)
            K_J_S = -ax2skew(self.K_r_PS)
            I_J_S = self.A_IK(t, x[:3]) @ K_J_S
            f_pot = I_J_S.T @ (-self.m * self.g * e3)
            h = f_pot - f_gyr

            dx[:3] = inv3D(Q) @ x[3:]
            dx[3:] = self.K_Theta_P_inv @ h

            return dx

    polynomial_degree = 1
    nelements = 1

    ######################
    # nice locking results
    ######################
    g = 9.81
    l = 0.5
    r = 0.1
    omega_x = 2 * pi * 50
    E_stiff = 210e6  # steel (stiff beam)
    E_soft = E_stiff * 1.0e-3  # soft beam
    rho = 8000  # steel [kg/m^3]

    # tip volume and mass
    V = l * pi * r**2
    m = rho * V
    print(f"total mass: {m}")
    cross_section = CircularCrossSection(rho, r)
    A_rho0 = rho * cross_section.area
    K_S_rho0 = rho * cross_section.first_moment
    K_I_rho0 = rho * cross_section.second_moment

    # initial angular velocity and orientation
    # A = 0.5 * m * r**2
    # omega_pr = m * g * (0.5 * l) / (A * omega_x)
    omega_pr = g * l / (r**2 * omega_x)
    K_omega_IK0 = omega_x * e1 + omega_pr * e3  # perfect precession motion
    A_IK0 = np.eye(3, dtype=float)
    from scipy.spatial.transform import Rotation

    angles0 = Rotation.from_matrix(A_IK0).as_euler("zyx")

    # starting point
    r_OP0 = np.zeros(3, dtype=float)

    ###################
    # solver parameters
    ###################
    t0 = 0.0
    t1 = 2.0 * pi / omega_pr
    # t1 *= 0.5
    t1 *= 0.25
    # t1 *= 0.125
    # t1 *= 0.075
    # t1 *= 0.01

    # nt = np.ceil(t1 / 1.0e-3)
    # dt = t1 * 1.0e-2
    dt = t1 * 1.0e-3
    rtol = 1.0e-8
    atol = 1.0e-8

    t_eval = np.arange(t0, t1 + dt, dt)

    def solve(E):
        nu = 1.0 / 3.0
        G = E / (2.0 * (1.0 + nu))

        A = cross_section.area
        Ip, I2, I3 = np.diag(cross_section.second_moment)
        Ei = np.array([E * A, G * A, G * A])
        Fi = np.array([G * Ip, E * I2, E * I3])
        material_model = Simo1986(Ei, Fi)

        q0, u0 = DirectorAxisAngle.initial_configuration(
            polynomial_degree,
            nelements,
            l,
            r_OP0=r_OP0,
            A_IK0=A_IK0,
            v_P0=np.zeros(3, dtype=float),
            K_omega_IK0=K_omega_IK0,
        )
        beam = DirectorAxisAngle(
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            polynomial_degree,
            nelements,
            q0,
            u0=u0,
        )

        # junction
        r_OB0 = np.zeros(3, dtype=float)
        frame = Frame(r_OP=r_OB0, A_IK=A_IK0)
        joint = SphericalJoint(frame, beam, r_OB0, frame_ID2=(0,))

        # gravity beam
        vg = np.array(-cross_section.area * cross_section.density * g * e3, dtype=float)
        f_g_beam = DistributedForce1DBeam(lambda t, xi: vg, beam)

        # assemble the model
        model = System()
        model.add(beam)
        model.add(frame)
        model.add(joint)
        model.add(f_g_beam)
        model.assemble()

        # TODO: Use RK45!!!
        # sol = ScipyIVP(model, t1, dt, method="RK23", rtol=rtol, atol=atol).solve()
        sol = ScipyIVP(model, t1, dt, method="RK45", rtol=rtol, atol=atol).solve()

        return beam, sol

    # compute reference solution using Euler top equations
    heavy_top = RigidHeavyTopODE(m, r, l, g)
    x0 = np.concatenate((angles0, K_omega_IK0))
    from scipy.integrate import solve_ivp

    ref = solve_ivp(
        heavy_top,
        (t_eval[0], t_eval[-1]),
        x0,
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    t_ref = ref.t
    angles_ref = ref.y[:3].T
    r_OP_ref = np.array(
        [
            heavy_top.r_OP(ti, qi, K_r_SP=np.array([l / 2, 0, 0]))
            for (ti, qi) in zip(t_ref, angles_ref)
        ]
    )

    # # debug rigid top solution
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot(*r_OP_ref.T)
    # plt.show()
    # exit()

    # sove for beam solutions
    beam_stiff, sol_stiff = solve(E_stiff)
    beam_soft, sol_soft = solve(E_soft)

    q_stiff = sol_stiff.q
    nt = len(q_stiff)
    t = sol_stiff.t[:nt]
    q_soft = sol_soft.q
    nt = len(q_soft)
    t = sol_soft.t[:nt]

    ############################
    # Visualize tip displacement
    ############################
    elDOF = beam_stiff.qDOF[beam_stiff.elDOF[-1]]
    r_OP_stiff = np.array(
        [beam_stiff.r_OP(ti, qi[elDOF], (1,)) for (ti, qi) in zip(t, q_stiff)]
    )
    elDOF = beam_soft.qDOF[beam_soft.elDOF[-1]]
    r_OP_soft = np.array(
        [beam_soft.r_OP(ti, qi[elDOF], (1,)) for (ti, qi) in zip(t, q_soft)]
    )

    fig = plt.figure(figsize=(10, 8))

    # 3D tracjectory of tip displacement
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_title("3D tip trajectory")
    ax.plot(*r_OP_ref.T, "-k", label="rigid body")
    ax.plot(*r_OP_stiff.T, "--r", label="stiff beam")
    ax.plot(*r_OP_soft.T, "--b", label="soft beam")
    ax.set_xlabel("x [-]")
    ax.set_ylabel("y [-]")
    ax.set_zlabel("z [-]")
    ax.grid()
    ax.legend()

    # tip displacement
    ax = fig.add_subplot(1, 2, 2)

    ax.set_title("tip displacement (components)")

    ax.plot(t, r_OP_ref[:, 0], "-k", label="x_ref")
    ax.plot(t, r_OP_ref[:, 1], "-k", label="y_ref")
    ax.plot(t, r_OP_ref[:, 2], "-k", label="z_ref")
    ax.plot(t, r_OP_stiff[:, 0], "--r", label="x stiff")
    ax.plot(t, r_OP_stiff[:, 1], "--r", label="y stiff")
    ax.plot(t, r_OP_stiff[:, 2], "--r", label="z stiff")
    ax.plot(t, r_OP_soft[:, 0], "--b", label="x soft")
    ax.plot(t, r_OP_soft[:, 1], "--b", label="y soft")
    ax.plot(t, r_OP_soft[:, 2], "--b", label="z soft")
    ax.set_xlabel("t")
    ax.set_ylabel("x, y")
    ax.grid()
    ax.legend()

    # ax2 = ax.twinx()
    # ax2.plot(t, r_OP_ref[:, 2], "-k", label="z_ref")
    # ax2.plot(t, r_OP_stiff[:, 2], "--r", label="z stiff")
    # ax2.plot(t, r_OP_soft[:, 2], "--b", label="z soft")
    # ax2.set_ylabel("z")
    # ax2.grid()
    # ax2.legend()

    fig.tight_layout()

    plt.show()

    animate_beam(t, q_soft, [beam_soft], scale=l)


if __name__ == "__main__":
    statics()
    # dynamics()
    # HeavyTop()
