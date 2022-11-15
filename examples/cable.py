from cardillo.math import e3
from cardillo.discrete import Frame
from cardillo.constraints import SphericalJoint
from cardillo.beams import (
    animate_beam,
    Cable,
)
from cardillo.beams.cable import QuadraticMaterial
from cardillo.forces import DistributedForce1DBeam
from cardillo import System
from cardillo.solver import RadauIIa, EulerBackward

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # number of elements
    nelements = 3

    # used polynomial degree
    polynomial_degree = 3
    basis = "Hermite"

    # beam parameters found in Section 5.1 Ibrahimbegovic1997
    L = np.pi
    k_e = 1.0e4
    k_b = 1.0e2

    # build quadratic material model
    material_model = QuadraticMaterial(k_e, k_b)

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    q0 = Cable.straight_configuration(
        basis,
        polynomial_degree,
        nelements,
        L,
        r_OP0,
        A_IK0,
    )
    u0 = np.zeros_like(q0)

    # Note: This is never used in statics!
    A_rho0 = 1.0

    beam = Cable(
        material_model,
        A_rho0,
        polynomial_degree,
        nelements,
        Q=q0,
        q0=q0,
        u0=u0,
        basis=basis,
    )

    # junctions
    frame_left = Frame()
    r_OP1 = np.array([L, 0, 0])
    frame_right = Frame(r_OP=r_OP1)

    # left and right joint
    joint_left = SphericalJoint(frame_left, beam, r_OB=r_OP0, frame_ID2=(0,))
    joint_right = SphericalJoint(frame_right, beam, r_OB=r_OP1, frame_ID2=(1,))

    # line distributed body force
    l = lambda t, xi: -e3 * 2e1
    line_force = DistributedForce1DBeam(l, beam)

    # assemble the system
    system = System()
    system.add(beam)
    system.add(frame_left)
    system.add(joint_left)
    system.add(frame_right)
    system.add(joint_right)
    system.add(line_force)
    system.assemble()

    t1 = 1
    dt = 1.0e-2
    rtol = 1.0e-2
    atol = 1.0e-2
    dae_index = "GGL"
    # sol = RadauIIa(system, t1, dt, rtol, atol, dae_index).solve()
    sol = EulerBackward(system, t1, dt).solve()

    t = sol.t
    q = sol.q

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam], L, show=True)
