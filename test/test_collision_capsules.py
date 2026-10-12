"""Capsules and the generic pair kernels of the CollisionWorld (Plan D, phase 3)."""

import numpy as np
import pytest
from scipy.optimize import minimize

from cardillo import System
from cardillo.contacts.collision import (
    CapsuleShape,
    CollisionWorld,
    PlaneShape,
    SphereShape,
)
from cardillo.contacts.collision.world import INACTIVE_GAP
from cardillo.discrete import Frame, PointMass, RigidBody
from cardillo.forces import Force
from cardillo.math import A_IB_basic, Exp_SO3
from cardillo.solver import DualStormerVerlet, Moreau, SolverOptions

MU = 0.3


def _random_rotation(rng):
    return Exp_SO3(rng.standard_normal(3))


def _static_world(shapes_data, exclude_same=True):
    """World of static shapes on frames. shapes_data: list of (kind, params, r, A)."""
    system = System()
    shapes = []
    for kind, params, r, A in shapes_data:
        frame = Frame(r_OP=np.asarray(r, dtype=float), A_IB=np.asarray(A, dtype=float))
        system.add(frame)
        if kind == "sphere":
            shapes.append(SphereShape(frame, params[0]))
        elif kind == "capsule":
            shapes.append(CapsuleShape(frame, params[0], params[1]))
        else:
            shapes.append(PlaneShape(frame))
    world = CollisionWorld(shapes, margin=1e3)
    system.add(world)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))
    return system, world


def _geometry(system, world):
    # all slots are candidates (large margin); gaps and contact point offsets
    geom = world._geometry(0.0, np.zeros(0))
    R, _ = world._positions(0.0, np.zeros(0))
    return geom, R


def _segment(r, A, h):
    e = A[:, 2]
    return r - h * e, r + h * e


def _brute_force_segments(p1, q1, p2, q2):
    """Closest points of two segments by bounded minimization of a convex function."""

    def dist2(x):
        s, t = x
        d = (p1 + s * (q1 - p1)) - (p2 + t * (q2 - p2))
        return d @ d

    best = None
    for start in [(0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]:
        res = minimize(
            dist2,
            start,
            bounds=[(0, 1), (0, 1)],
            method="L-BFGS-B",
            options=dict(ftol=1e-16, gtol=1e-12),
        )
        if best is None or res.fun < best.fun:
            best = res
    s, t = best.x
    return np.sqrt(best.fun), p1 + s * (q1 - p1), p2 + t * (q2 - p2)


def test_capsule_capsule_gap_and_witness_points():
    rng = np.random.default_rng(0)
    checked_unique = 0
    for _ in range(150):
        r1, r2 = rng.uniform(-0.3, 0.3, 3), rng.uniform(-0.3, 0.3, 3)
        A1, A2 = _random_rotation(rng), _random_rotation(rng)
        h1, h2 = rng.uniform(0.05, 0.3, 2)
        rho1, rho2 = rng.uniform(0.01, 0.05, 2)
        system, world = _static_world(
            [("capsule", (rho1, h1), r1, A1), ("capsule", (rho2, h2), r2, A2)]
        )
        geom, R = _geometry(system, world)
        dist, Q1, Q2 = _brute_force_segments(
            *_segment(r1, A1, h1), *_segment(r2, A2, h2)
        )
        g = geom.g[0]
        assert abs(g - (dist - rho1 - rho2)) < 1e-9
        if dist > 1e-6:  # normal and contact points are only unique for distinct axes
            n = geom.n[0]
            assert np.allclose(n, (Q2 - Q1) / dist, atol=1e-6)
            P1, P2 = R[0] + geom.c1[0], R[1] + geom.c2[0]
            assert np.allclose(P1, Q1 + rho1 * n, atol=1e-6)
            assert np.allclose(P2, Q2 - rho2 * n, atol=1e-6)
            checked_unique += 1
        assert abs(np.linalg.norm(geom.n[0]) - 1) < 1e-12
    assert checked_unique > 100


def test_capsule_capsule_special_configurations():
    rho, h = 0.05, 0.2
    I3 = np.eye(3)
    Ax = A_IB_basic(np.pi / 2).y  # axis along +x
    # crossing axes (intersecting): normal is perpendicular to both, gap = -2 rho
    system, world = _static_world(
        [("capsule", (rho, h), [0, 0, 0], I3), ("capsule", (rho, h), [0, 0, 0], Ax)]
    )
    geom, _ = _geometry(system, world)
    assert np.isclose(geom.g[0], -2 * rho)
    assert np.allclose(np.abs(geom.n[0]), [0, 1, 0])  # cross(e_z, e_x) = +-e_y

    # parallel axes, side by side: contact point in the middle of the overlap
    system, world = _static_world(
        [("capsule", (rho, h), [0, 0, 0], I3), ("capsule", (rho, h), [0.3, 0, 0.1], I3)]
    )
    geom, R = _geometry(system, world)
    assert np.isclose(geom.g[0], 0.3 - 2 * rho)
    assert np.allclose(geom.n[0], [1, 0, 0])
    # overlap of the axes is z in [-0.1, 0.2]: middle z = 0.05 on both axes
    assert np.isclose((R[0] + geom.c1[0])[2], 0.05) and np.isclose(
        (R[1] + geom.c2[0])[2], 0.05
    )

    # end to end (collinear): closest points are the touching end points
    system, world = _static_world(
        [("capsule", (rho, h), [0, 0, 0], I3), ("capsule", (rho, h), [0, 0, 0.5], I3)]
    )
    geom, R = _geometry(system, world)
    assert np.isclose(geom.g[0], (0.5 - 2 * h) - 2 * rho)
    assert np.allclose(geom.n[0], [0, 0, 1])


def test_sphere_capsule_gap_and_witness_points():
    rng = np.random.default_rng(1)
    for _ in range(100):
        rc, xs = rng.uniform(-0.2, 0.2, 3), rng.uniform(-0.2, 0.2, 3)
        A = _random_rotation(rng)
        h = rng.uniform(0.05, 0.3)
        rho_s, rho_c = rng.uniform(0.01, 0.05, 2)
        system, world = _static_world(
            [("sphere", (rho_s,), xs, np.eye(3)), ("capsule", (rho_c, h), rc, A)]
        )
        geom, R = _geometry(system, world)
        # reference: projection of the center onto the axis
        e = A[:, 2]
        u = np.clip((xs - rc) @ e, -h, h)
        Q = rc + u * e
        dist = np.linalg.norm(Q - xs)
        slot = 0
        assert np.isclose(geom.g[slot], dist - rho_s - rho_c)
        n = (Q - xs) / dist
        assert np.allclose(geom.n[slot], n)
        assert np.allclose(R[0] + geom.c1[slot], xs + rho_s * n)
        assert np.allclose(R[1] + geom.c2[slot], Q - rho_c * n)


def test_plane_capsule_has_two_end_point_slots():
    rho, h = 0.05, 0.2
    A = A_IB_basic(0.4).x  # tilted capsule above the plane z = 0
    r = np.array([0.1, 0.0, 0.5])
    system, world = _static_world(
        [("plane", (), [0, 0, 0], np.eye(3)), ("capsule", (rho, h), r, A)]
    )
    assert world.nla_N == 2
    geom, R = _geometry(system, world)
    ends = [r - h * A[:, 2], r + h * A[:, 2]]
    assert np.allclose(geom.g[:2], [e[2] - rho for e in ends])
    assert np.allclose(geom.n[:2], [[0, 0, 1], [0, 0, 1]])
    for k in range(2):  # contact point on the capsule: end point shifted by -rho n
        assert np.allclose(R[1] + geom.c2[k], ends[k] - rho * np.array([0, 0, 1]))
        assert np.allclose(R[0] + geom.c1[k], [ends[k][0], ends[k][1], 0])


def _motion_scene(seed):
    """Rigid bodies and a point mass with all supported shapes, a floor frame and a
    moving board carrying a second plane. All pairs are candidates."""
    rng = np.random.default_rng(seed)

    def rigid(r):
        p = rng.standard_normal(4)
        return RigidBody(
            1.0,
            np.diag([0.1, 0.2, 0.3]),
            np.concatenate([r, p / np.linalg.norm(p)]),
            rng.standard_normal(6),
        )

    floor = Frame(A_IB=A_IB_basic(0.1).x @ A_IB_basic(0.05).y)
    board = rigid(np.array([0.0, 0.0, -0.4]))
    sphere = rigid(np.array([0.0, 0.0, 0.15]))
    caps1 = rigid(np.array([0.1, 0.05, 0.2]))
    caps2 = rigid(np.array([-0.05, 0.1, 0.1]))
    pm = PointMass(0.5, q0=np.array([0.1, -0.1, 0.05]), u0=rng.standard_normal(3))
    system = System()
    system.add(floor, board, sphere, caps1, caps2, pm)
    shapes = [
        PlaneShape(floor),
        PlaneShape(board, B_r_CS=np.array([0.0, 0.0, 0.1]), A_BS=A_IB_basic(0.3).y),
        SphereShape(sphere, 0.05),
        SphereShape(pm, 0.03),
        CapsuleShape(caps1, 0.04, 0.12),
        CapsuleShape(
            caps2, 0.03, 0.1, B_r_CS=np.array([0.01, 0.0, 0.02]), A_BS=A_IB_basic(0.5).x
        ),
    ]
    world = CollisionWorld(shapes, mu=MU, e_N=0.2, e_F=0.1, margin=1e3)
    system.add(world)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))
    return system, world


def _state(system, rng):
    q = system.q0 + 0.03 * rng.standard_normal(system.nq)
    u = rng.standard_normal(system.nu)
    # keep the quaternions close to unit length
    return q, u


def test_slot_layout_with_capsules():
    system, world = _motion_scene(0)
    # planes: 2, spheres: 2, capsules: 2
    # plane-sphere 4, plane-capsule 2 planes * 2 capsules * 2 end points = 8,
    # body pairs of 4 non-plane shapes: 6
    assert world.nla_N == 4 + 8 + 6
    assert world.nla_F == 2 * world.nla_N
    types = world._pair_type
    assert list(types) == [0, 1, 1, 1, 1, 2]  # SS, SC x4, CC in lexicographic order


@pytest.mark.parametrize("seed", range(3))
def test_normal_velocity_is_derivative_of_gap(seed):
    """g_N_dot equals the time derivative of g_N along the motion for every pair
    type (envelope theorem for moving contact points), and W^T u reproduces the
    velocities."""
    system, world = _motion_scene(seed)
    rng = np.random.default_rng(10 + seed)
    for _ in range(3):
        q, u = _state(system, rng)
        t = 0.0
        q_dot = system.q_dot(t, q, u)
        h = 1e-6
        dg = (
            system.g_N(
                t,
                q + h * q_dot,
            )
            - system.g_N(t, q - h * q_dot)
        ) / (2 * h)
        g_N_dot = system.g_N_dot(t, q, u)
        assert np.allclose(g_N_dot, dg, rtol=1e-5, atol=1e-6), np.abs(
            g_N_dot - dg
        ).max()
        W_N = system.W_N(t, q, format="csc").toarray()
        assert np.allclose(W_N.T @ u, g_N_dot, rtol=1e-12, atol=1e-12)
        W_F = system.W_F(t, q, format="csc").toarray()
        assert np.allclose(W_F.T @ u, system.gamma_F(t, q, u), rtol=1e-12, atol=1e-12)


def test_tangents_are_orthonormal_and_continuous():
    system, world = _motion_scene(0)
    q, u = _state(system, np.random.default_rng(3))
    geom = world._geometry(0.0, q)
    _, A = world._positions(0.0, q)
    slots = geom.candidates
    t12 = world._tangents(slots, geom, A)
    n = geom.n[slots]
    assert np.allclose(np.einsum("kmi,kmi->km", t12, t12), 1)
    assert np.allclose(np.einsum("ki,ki->k", t12[:, 0], t12[:, 1]), 0, atol=1e-14)
    assert np.allclose(np.einsum("kmi,ki->km", t12, n), 0, atol=1e-14)
    assert np.allclose(np.cross(t12[:, 0], t12[:, 1]), n, atol=1e-14)


def test_acceleration_level_finite_difference_matches_exact_formulas():
    """The central difference used for capsule slots reproduces the exact
    formulas of the plane-sphere slots."""
    system, world = _motion_scene(1)
    rng = np.random.default_rng(7)
    q, u = _state(system, rng)
    u_dot = rng.standard_normal(system.nu)
    t = 0.0
    g_ddot = system.g_N_ddot(t, q, u, u_dot)
    gamma_dot = system.gamma_F_dot(t, q, u, u_dot)
    # the world works on its own coordinate layout (qDOF, uDOF)
    qw, uw, udw = q[world.qDOF], u[world.uDOF], u_dot[world.uDOF]
    exact_slots = np.arange(world._nps_sphere)  # plane-sphere slots
    assert len(exact_slots) == 4
    fd_N = world._fd_time_derivative(
        lambda t_, q_, u_: world.g_N_dot(t_, q_, u_, I_N=exact_slots)[exact_slots],
        t,
        qw,
        uw,
        udw,
    )
    I_F = np.concatenate((2 * exact_slots, 2 * exact_slots + 1))
    fd_F = world._fd_time_derivative(
        lambda t_, q_, u_: world._gamma_F(t_, q_, u_, I_F=I_F)[I_F], t, qw, uw, udw
    )
    assert np.allclose(g_ddot[exact_slots], fd_N, rtol=1e-5, atol=1e-5)
    assert np.allclose(gamma_dot[I_F], fd_F, rtol=1e-5, atol=1e-5)

    # capsule slots use the central difference of the system level velocities
    qd = system.q_dot(t, q, u)
    h = 1e-5
    ref = (
        system.g_N_dot(t + h, q + h * qd, u + h * u_dot)
        - system.g_N_dot(t - h, q - h * qd, u - h * u_dot)
    ) / (2 * h)
    capsule_slots = np.arange(world._nps_sphere, world._nps)
    assert np.allclose(g_ddot[capsule_slots], ref[capsule_slots], rtol=1e-6, atol=1e-6)
    assert np.any(g_ddot[capsule_slots] != 0)
    # sphere-capsule and capsule-capsule slots as well
    pair_fd = world._nps + np.nonzero(world._pair_type > 0)[0]
    assert np.allclose(g_ddot[pair_fd], ref[pair_fd], rtol=1e-6, atol=1e-6)


def _pair_gaps_bruteforce(world, R, A):
    g, _, _, _ = world._pair_geometry(np.arange(world._npair), R, A)
    return g


def test_broad_phase_with_capsules():
    rng = np.random.default_rng(4)
    system = System()
    bodies, shapes = [], []
    for i in range(40):
        if i % 2 == 0:
            body = PointMass(1.0, q0=rng.uniform(0, 1, 3), u0=np.zeros(3))
            shapes.append(SphereShape(body, rng.uniform(0.01, 0.04)))
        else:
            p = rng.standard_normal(4)
            body = RigidBody(
                1.0,
                np.eye(3),
                np.concatenate([rng.uniform(0, 1, 3), p / np.linalg.norm(p)]),
                np.zeros(6),
            )
            shapes.append(
                CapsuleShape(body, rng.uniform(0.01, 0.03), rng.uniform(0.03, 0.1))
            )
        bodies.append(body)
    margin = 0.05
    world = CollisionWorld(shapes, margin=margin)
    system.add(*bodies)
    system.add(world)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))
    q = system.q0.copy()
    g = system.g_N(0.0, q)
    R, A = world._positions(0.0, q)
    exact = _pair_gaps_bruteforce(world, R, A)
    close = exact <= margin
    assert np.allclose(g[close], exact[close])  # all close pairs are candidates
    skipped = g >= INACTIVE_GAP
    assert np.all(exact[skipped] > margin)  # skipped pairs are separated
    assert np.count_nonzero(~skipped) < len(g) / 2

    # moves below margin / 4: no rebuild, skipped pairs keep a gap above margin / 2
    snapshot = world._snapshot.copy()
    for _ in range(10):
        q_new = q.copy()
        pos = np.array([b.qDOF[:3] for b in bodies])
        d = rng.standard_normal((len(bodies), 3))
        d *= 0.999 * margin / 4 / np.linalg.norm(d, axis=1, keepdims=True)
        q_new[pos] = q[pos] + d
        # additionally rotate the capsules arbitrarily (does not move the centers)
        for b in bodies[1::2]:
            p = rng.standard_normal(4)
            q_new[b.qDOF[3:]] = p / np.linalg.norm(p)
        g = system.g_N(0.0, q_new)
        assert np.array_equal(world._snapshot, snapshot)  # not rebuilt
        R, A = world._positions(0.0, q_new)
        exact = _pair_gaps_bruteforce(world, R, A)
        skipped = g >= INACTIVE_GAP
        assert np.all(exact[skipped] > margin / 2)
        assert np.allclose(g[~skipped], exact[~skipped])


def _capsule_body(r, A, mass, rho, h):
    # solid capsule: use the inertia of a cylinder of the same length as estimate
    Theta = np.diag([mass * (3 * rho**2 + (2 * h) ** 2) / 12] * 2 + [mass * rho**2 / 2])
    return RigidBody(mass, Theta, RigidBody.pose2q(r, A), np.zeros(6))


# Moreau enforces the contact on velocity level, the penetration at impact
# (about v * dt / 2) is not corrected afterwards. This is the same as for
# Sphere2Plane. The scenarios start close to the surface, so the remaining
# penetration is a few 1e-5; the tolerance leaves some margin.
RESTING_TOL = 2e-4


def _simulate(system, t1=0.4, dt=1e-3, Solver=Moreau):
    return Solver(
        system,
        t1,
        dt,
        options=SolverOptions(fixed_point_max_iter=int(1e4), prox_scaling=0.5),
    ).solve()


def test_capsule_rests_flat_on_plane_with_two_contact_points():
    rho, h, mass, g = 0.05, 0.2, 1.0, 10.0
    system = System()
    floor = Frame(name="floor")
    # horizontal capsule (axis along x) slightly above the plane z = 0
    A = A_IB_basic(np.pi / 2).y
    body = _capsule_body(np.array([0.0, 0.0, rho + 0.001]), A, mass, rho, h)
    system.add(floor, body, Force(np.array([0, 0, -mass * g]), body))
    system.add(CollisionWorld([PlaneShape(floor), CapsuleShape(body, rho, h)], mu=MU))
    system.assemble()
    sol = _simulate(system, t1=0.4)
    z = sol.q[:, 2]
    assert abs(z[-1] - rho) < RESTING_TOL  # resting on the plane
    # the axis stays horizontal and the capsule does not slide or tumble
    assert np.abs(sol.q[-1, :2]).max() < 1e-6
    # the normal forces of the two end points are equal and carry the weight
    # (Moreau returns forces la_N, not percussions)
    F_end = sol.la_N[-100:].mean(axis=0)
    assert np.isclose(F_end.sum(), mass * g, rtol=1e-3)
    assert np.isclose(F_end[0], F_end[1], rtol=1e-3)


def test_sphere_rests_on_capsule():
    rho_s, rho_c, h = 0.05, 0.04, 0.3
    system = System()
    frame = Frame(
        r_OP=np.array([0.0, 0.0, 0.0]), A_IB=A_IB_basic(np.pi / 2).y
    )  # axis x
    ball = RigidBody(
        1.0,
        np.eye(3) * 1e-3,
        RigidBody.pose2q(np.array([0.05, 0.0, rho_c + rho_s + 0.001]), np.eye(3)),
        np.zeros(6),
    )
    system.add(frame, ball, Force(np.array([0, 0, -10.0]), ball))
    system.add(
        CollisionWorld([CapsuleShape(frame, rho_c, h), SphereShape(ball, rho_s)], mu=MU)
    )
    system.assemble()
    sol = _simulate(system, t1=0.5)
    # the ball rests on top of the capsule axis (x within the axis length)
    assert np.isclose(sol.q[-1, 2], rho_c + rho_s, atol=RESTING_TOL)
    assert abs(sol.q[-1, 0] - 0.05) < 1e-3


def test_capsule_on_capsule_cross():
    """A capsule falls on a perpendicular, fixed capsule with its center above the
    crossing point and rests at the sum of the radii."""
    rho, h = 0.04, 0.25
    system = System()
    lower = Frame(r_OP=np.zeros(3), A_IB=A_IB_basic(np.pi / 2).y)  # axis x
    A_up = A_IB_basic(np.pi / 2).x  # axis along -y
    upper = _capsule_body(np.array([0.0, 0.0, 2 * rho + 0.001]), A_up, 1.0, rho, h)
    system.add(lower, upper, Force(np.array([0, 0, -10.0]), upper))
    system.add(
        CollisionWorld(
            [CapsuleShape(lower, rho, h), CapsuleShape(upper, rho, h)],
            mu=MU,
        )
    )
    system.assemble()
    sol = _simulate(system, t1=0.3)
    assert np.isclose(sol.q[-1, 2], 2 * rho, atol=RESTING_TOL)
    assert np.abs(sol.q[-1, :2]).max() < 1e-4


def test_moreau_and_dual_stormer_verlet_agree_for_capsules():
    def build():
        rho, h = 0.05, 0.2
        system = System()
        floor = Frame(name="floor")
        A = A_IB_basic(np.pi / 2).y @ A_IB_basic(0.4).x
        body = _capsule_body(np.array([0.0, 0.0, 0.3]), A, 1.0, rho, h)
        body.u0 = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 1.0])
        system.add(floor, body, Force(np.array([0, 0, -10.0]), body))
        system.add(
            CollisionWorld([PlaneShape(floor), CapsuleShape(body, rho, h)], mu=MU)
        )
        system.assemble(
            options=SolverOptions(compute_consistent_initial_conditions=False)
        )
        return system

    a = _simulate(build(), t1=0.5, dt=5e-4, Solver=Moreau)
    b = _simulate(build(), t1=0.5, dt=5e-4, Solver=DualStormerVerlet)
    # same physics: the capsule lands on the plane at the same time and place
    assert np.allclose(a.q[-1, :3], b.q[-1, :3], atol=5e-3)
