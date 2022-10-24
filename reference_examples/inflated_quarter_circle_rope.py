from cardillo.discrete import Frame
from cardillo.constraints import Linear_guidance_xyz, SphericalJoint
from cardillo.beams import (
    Rope,
    animate_rope,
)

from cardillo.beams.rope import QuadraticMaterial
from cardillo import System
from cardillo.solver import (
    Newton,
    ScipyIVP,
)
from cardillo.math import pi, e3, rodriguez, approx_fprime, norm

import numpy as np
import matplotlib.pyplot as plt


class Inflated(Rope):
    def __init__(self, *args, **kwargs):
        super().__init__(*args[1:], **kwargs)
        self.pressure = args[0]

    def area(self, q):
        a = np.zeros(1, dtype=q.dtype)[0]
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]

            for i in range(self.nquadrature):
                # extract reference state variables
                qwi = self.qw[el, i]

                # interpolate tangent vector
                r = np.zeros(3, dtype=qe.dtype)
                r_xi = np.zeros(3, dtype=qe.dtype)
                for node in range(self.nnodes_element):
                    r += self.N[el, i, node] * qe[self.nodalDOF_element[node]]
                    r_xi += self.N_xi[el, i, node] * qe[self.nodalDOF_element[node]]

                # counterclockwise rotated tangent vector
                r_xi_perp = np.array([-r_xi[1], r_xi[0], 0.0], dtype=qe.dtype)

                # integrate area
                a += 0.5 * r @ r_xi_perp * qwi
        return a

    def h(self, t, q, u):
        f = super().h(t, q, u)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f[elDOF] += self.f_npot_el(t, q[elDOF], u[elDOF], el)
        return f

    def f_npot_el(self, t, qe, ue, el):
        pressure = self.pressure(t)

        f_el = np.zeros(self.nu_element, dtype=qe.dtype)
        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]

            # interpolate tangent vector
            r_xi = np.zeros(3, dtype=qe.dtype)
            for node in range(self.nnodes_element):
                r_xi += self.N_xi[el, i, node] * qe[self.nodalDOF_element[node]]

            # counterclockwise rotated tangent vector
            r_xi_perp = np.array([-r_xi[1], r_xi[0], 0.0], dtype=qe.dtype)

            # assemble
            for node in range(self.nnodes_element):
                f_el[self.nodalDOF_element[node]] += (
                    self.N[el, i, node] * pressure * r_xi_perp * qwi
                )
        return f_el

    def h_q(self, t, q, u, coo):
        super().h_q(t, q, u, coo)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f_npot_q_el = self.f_npot_q_el(t, q[elDOF], u[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(f_npot_q_el, (self.uDOF[elDOF], self.qDOF[elDOF]))

    def f_npot_q_el(self, t, qe, ue, el):
        f_npot_q_el_num = approx_fprime(
            qe, lambda qe: self.f_npot_el(t, qe, ue, el), eps=1.0e-10, method="cs"
        )
        return f_npot_q_el_num


case = "statics"
# case = "dynamics"

if __name__ == "__main__":

    # rope parameters
    R = 1  # radius inflated circle
    A_rho0 = 1.0e0  # reference density per unit length
    pressure_level = 2.0e2  # max internal pressure

    # material law rope
    k_e = 1.0e3  # stiffness
    material_law = QuadraticMaterial(k_e)  # quadratic potential

    ############################################################################
    #                   rope + boundary constraints
    ############################################################################

    # discretization properties
    nelements = 10
    polynomial_degree = 1
    basis = "Lagrange"
    # polynomial_degree = 3
    # basis = "B-spline"

    # initial configuration: quarter circle
    q0 = Rope.quarter_circle_configuration(
        basis,
        polynomial_degree,
        nelements,
        R,
    )

    # initial nodal positions
    r0 = q0.copy().reshape(3, -1, order="C")

    # reference configuration: corresponds to initial configuration
    Q = q0

    if case == "statics":
        # internal pressure function
        pressure = lambda t: t * pressure_level

        # Manipulate initial configuration in order to overcome singular initial
        # configuration. Do not change first and last node, otherwise constraints
        # are violated!

        # random manipulation
        # eps = 1.0e-6
        # r0 = Q.copy().reshape(3, -1, order="C")
        # nn = r0.shape[1]
        # for i in range(1, nn - 1):
        #     r0[:2, i] += eps * 0.5 * (2.0 * np.random.rand(2) - 1)
        # q0 = r0.reshape(-1, order="C")

        # radial manipulation (more robust for linear Lagrange elements)
        eps = 1.0e-6
        nn = r0.shape[1]
        for i in range(1, nn - 1):
            radius = norm(r0[:2, i])
            r0[0, i] += eps * r0[0, i] / radius
            r0[1, i] += eps * r0[1, i] / radius
        q0 = r0.reshape(-1, order="C")
    elif case == "dynamics":
        # internal pressure function
        pressure = lambda t: pressure_level

    # construct inflated rope
    rope = Inflated(
        pressure,
        material_law,
        A_rho0,
        polynomial_degree,
        nelements,
        Q,
        q0=q0,
        basis=basis,
    )

    # left joint
    r_OP0 = r0[:, 0]
    A_IK0 = rodriguez(pi / 2 * e3)
    frame0 = Frame(r_OP=r_OP0, A_IK=A_IK0)
    # joint0 = SphericalJoint(frame0, rope, r_OP0, frame_ID2=(0,))
    joint0 = Linear_guidance_xyz(frame0, rope, r_OP0, A_IK0, frame_ID2=(0,))

    # left joint
    r_OP1 = r0[:, -1]
    A_IK1 = np.eye(3, dtype=float)
    frame1 = Frame(r_OP=r_OP1, A_IK=A_IK1)
    # joint1 = SphericalJoint(frame1, rope, r_OP1, frame_ID2=(1,))
    joint1 = Linear_guidance_xyz(frame1, rope, r_OP1, A_IK1, frame_ID2=(1,))

    ############################################################################
    #                   model
    ############################################################################
    model = System()
    model.add(rope)
    model.add(frame0)
    model.add(joint0)
    model.add(frame1)
    model.add(joint1)
    model.assemble()

    # # show initial configuration
    # animate_rope([0], [q0], [rope], R, show=True)

    ############################################################################
    #                   solver
    ############################################################################
    if case == "statics":
        atol = 1.0e-8
        n_load_steps = 10
        max_iter = 20
        solver = Newton(
            model,
            n_load_steps=n_load_steps,
            max_iter=max_iter,
            atol=atol,
        )
    elif case == "dynamics":
        atol = 1.0e-8
        rtol = 1.0e-6
        t1 = 1
        dt = 1.0e-2
        method = "RK45"
        solver = ScipyIVP(
            model,
            t1,
            dt,
            method=method,
            rtol=rtol,
            atol=atol,
        )

    sol = solver.solve()
    q = sol.q
    t = sol.t

    ############################################################################
    #                   compute analytical reference solutions
    ############################################################################

    # radius in final configuration
    r = rope.r_OP(1, q[-1][rope.local_qDOF_P((1,))], (1,))[0]

    # initial vs. current area
    a0 = rope.area(q[0])
    a = rope.area(q[-1])
    a0_analytic = np.pi * R**2 / 4
    a_analytic = np.pi * r**2 / 4
    print(f"A: {a0}")
    print(f"a: {a}")
    print(f"A analytic: {a0_analytic}")
    print(f"a analytic: {a_analytic}")

    # ratio of rope initial and deformed length
    l = pi * r / 2
    L = pi * R / 2
    print(f"l / L: {l / L}")

    # analytical stretch
    la_analytic = pressure(1) * r / k_e + 1
    print(f"analytical stretch: {la_analytic}")
    # stretch of the final configuration
    n = 100
    xis = np.linspace(0, 1, num=n)
    la = rope.stretch(q[-1], n=n)

    fig0, ax0 = plt.subplots()
    ax0.set_ylabel(r"stretch $\lambda$")
    ax0.set_xlabel(r"beam coordinate $\xi$")
    ax0.plot(xis, la, "-r", label="FEM solution")
    ax0.plot(
        xis, np.ones_like(xis) * la_analytic, "--b", label="analytic static solution"
    )
    ax0.set_ylim(0, 2)
    ax0.grid()
    ax0.legend()

    scale_plane = R * la_analytic * 1.2
    scale_z = 0.5

    fig1, ax1, anim1 = animate_rope(t, q, [rope], scale_plane, show=False)

    #     fig, ax, anim = animate_beam(t, q, [rod], scale, show=False)

    # plane with x-direction as normal
    Y_x = np.linspace(0, scale_plane, num=2)
    Z_x = np.linspace(-scale_z, scale_z, num=2)
    Y_x, Z_x = np.meshgrid(Y_x, Z_x)
    X_x = np.zeros_like(Y_x)
    ax1.plot_surface(X_x, Y_x, Z_x, alpha=0.5)

    # plane with y-direction as normal
    X_y = np.linspace(0, scale_plane, num=2)
    Z_y = np.linspace(-scale_z, scale_z, num=2)
    X_y, Z_y = np.meshgrid(X_y, Z_y)
    Y_y = np.zeros_like(X_y)
    ax1.plot_surface(X_y, Y_y, Z_y, alpha=0.5)

    # plane with z-direction as normal
    X_z = np.linspace(0, scale_plane, num=2)
    Y_z = np.linspace(0, scale_plane, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.5)

    plt.show()
