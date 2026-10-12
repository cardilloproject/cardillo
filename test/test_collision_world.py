"""CollisionWorld (generic contacts, Plan D phase 2) compared with the
per-pair contact classes Sphere2Sphere and Sphere2Plane."""

import numpy as np
import pytest

from cardillo import System
from cardillo.contacts import Sphere2Plane, Sphere2Sphere
from cardillo.contacts.collision import CollisionWorld, PlaneShape, SphereShape
from cardillo.contacts.collision.world import INACTIVE_GAP
from cardillo.discrete import Frame, PointMass, RigidBody
from cardillo.forces import Force
from cardillo.math import A_IB_basic, Spurrier
from cardillo.solver import BackwardEuler, DualStormerVerlet, Moreau, SolverOptions

MU = 0.3


def _bodies(rng):
    """Floor frame, a rigid body carrying a second plane, two rigid spheres and
    a point mass."""
    floor = Frame(A_IB=A_IB_basic(0.1).x @ A_IB_basic(0.05).y, name="floor")

    def rigid(r):
        p = rng.standard_normal(4)
        return RigidBody(
            1.0,
            np.diag([0.1, 0.2, 0.3]),
            np.concatenate([r, p / np.linalg.norm(p)]),
            rng.standard_normal(6),
        )

    board = rigid(np.array([0.0, 0.0, -0.5]))
    s1 = rigid(np.array([0.0, 0.0, 0.2]))
    s2 = rigid(np.array([0.15, 0.05, 0.25]))
    pm = PointMass(0.5, q0=np.array([-0.1, 0.1, 0.3]), u0=rng.standard_normal(3))
    return floor, board, s1, s2, pm


def _pair_system(world, seed=0):
    rng = np.random.default_rng(seed)
    floor, board, s1, s2, pm = _bodies(rng)
    system = System()
    system.add(floor, board, s1, s2, pm)
    A_board = A_IB_basic(0.3).z
    B_r_board = np.array([0.0, 0.0, 0.1])
    offset_s2 = np.array([0.01, 0.0, -0.02])
    radii = {id(s1): 0.05, id(s2): 0.04, id(pm): 0.03}
    spheres = [s1, s2, pm]
    if world:
        shapes = [PlaneShape(floor), PlaneShape(board, B_r_CS=B_r_board, A_BS=A_board)]
        shapes += [
            SphereShape(s1, 0.05),
            SphereShape(s2, 0.04, B_r_CS=offset_s2),
            SphereShape(pm, 0.03),
        ]
        # Sphere2Sphere does not support offsets: the reference only has the
        # sphere pair (s1, pm), so the two pairs with s2 are excluded
        contr = CollisionWorld(
            shapes,
            mu=MU,
            e_N=0.2,
            e_F=0.1,
            margin=1e3,
            exclude=[(shapes[2], shapes[3]), (shapes[3], shapes[4])],
        )
        system.add(contr)
    else:
        offsets = {id(s1): np.zeros(3), id(s2): offset_s2, id(pm): np.zeros(3)}
        for s in spheres:  # slot order of CollisionWorld: per sphere all planes
            system.add(
                Sphere2Plane(
                    floor, s, MU, radii[id(s)], B_r_CP2=offsets[id(s)], e_N=0.2, e_F=0.1
                )
            )
            system.add(
                Sphere2Plane(
                    board,
                    s,
                    MU,
                    radii[id(s)],
                    B_r_CP1=B_r_board,
                    A_B1P=A_board,
                    B_r_CP2=offsets[id(s)],
                    e_N=0.2,
                    e_F=0.1,
                )
            )
        # Sphere2Sphere has no offsets, use the spheres without offset only
        system.add(Sphere2Sphere(s1, pm, 0.05, 0.03, MU, e_N=0.2, e_F=0.1))
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))
    return system


def test_contact_quantities_match_pair_classes():
    sys_pair = _pair_system(world=False)
    sys_world = _pair_system(world=True)
    assert sys_world.nla_N == sys_pair.nla_N == 7
    assert sys_world.nq == sys_pair.nq and sys_world.nu == sys_pair.nu

    rng = np.random.default_rng(1)
    for trial in range(5):
        q = sys_pair.q0 + 0.02 * rng.standard_normal(sys_pair.nq)
        u = rng.standard_normal(sys_pair.nu)
        u_dot = rng.standard_normal(sys_pair.nu)
        t = 0.1 * trial
        for name, args in [
            ("g_N", (t, q)),
            ("g_N_dot", (t, q, u)),
            ("gamma_F", (t, q, u)),
            ("g_N_ddot", (t, q, u, u_dot)),
            ("gamma_F_dot", (t, q, u, u_dot)),
        ]:
            a = getattr(sys_world, name)(*args)
            b = getattr(sys_pair, name)(*args)
            assert np.allclose(a, b, rtol=1e-12, atol=1e-13), name
        for name in ("W_N", "W_F"):
            a = getattr(sys_world, name)(t, q, format="csc").toarray()
            b = getattr(sys_pair, name)(t, q, format="csc").toarray()
            assert np.allclose(a, b, rtol=1e-12, atol=1e-13), name
        # active subsets
        I_N = np.array([1, 6])
        a = sys_world.W_N(t, q, format="csc", I_N=I_N)[:, I_N].toarray()
        b = sys_pair.W_N(t, q, format="csc")[:, I_N].toarray()
        assert np.allclose(a, b, rtol=1e-12, atol=1e-13)
        # transport of the reference bases
        sys_world.step_callback(t, q.copy(), u.copy())
        sys_pair.step_callback(t, q.copy(), u.copy())


def _random_sphere_scene(n, rng, margin):
    system = System()
    bodies = []
    for _ in range(n):
        body = PointMass(1.0, q0=rng.uniform(0, 1, 3), u0=np.zeros(3))
        bodies.append(body)
    system.add(*bodies)
    radii = rng.uniform(0.01, 0.05, n)
    world = CollisionWorld(
        [SphereShape(b, r) for b, r in zip(bodies, radii)], margin=margin
    )
    system.add(world)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))
    return system, world, radii


def _all_pair_gaps(q, radii):
    X = q.reshape(-1, 3)
    i, j = np.triu_indices(len(X), 1)
    return np.linalg.norm(X[j] - X[i], axis=1) - radii[i] - radii[j]


def test_broad_phase_finds_all_close_pairs():
    rng = np.random.default_rng(2)
    system, world, radii = _random_sphere_scene(80, rng, margin=0.02)
    q = system.q0.copy()
    exact = _all_pair_gaps(q, radii)
    g = system.g_N(0.0, q)
    close = exact <= world.margin
    assert np.allclose(g[close], exact[close])  # all close pairs are evaluated
    assert np.all(g[~close] >= exact[~close] - 1e-12)  # others never too small
    assert np.count_nonzero(g < INACTIVE_GAP) < len(g) / 4  # and most are skipped

    # move all spheres by more than margin / 4 without a step callback: the
    # broad phase must be rebuilt before gaps are evaluated
    q_moved = q + rng.uniform(-0.05, 0.05, q.shape)
    exact = _all_pair_gaps(q_moved, radii)
    g = system.g_N(0.0, q_moved)
    close = exact <= world.margin
    assert np.allclose(g[close], exact[close])
    assert np.all(g[exact <= 0] <= 0)  # no penetration is missed


def _small_scene(world):
    """Two rigid spheres and a point mass colliding with each other and a plane."""
    rng = np.random.default_rng(3)
    system = System()
    floor = Frame(A_IB=A_IB_basic(0.2).x, name="floor")
    r = 0.05

    def rigid(r_OC, v):
        return RigidBody(
            1.0,
            np.eye(3) * 1e-3,
            RigidBody.pose2q(np.array(r_OC), np.eye(3)),
            np.concatenate([v, rng.standard_normal(3)]),
        )

    a = rigid([0.0, 0.0, 0.06], [1.0, 0.0, -0.5])
    b = rigid([0.14, 0.04, 0.06], [0.0, 0.0, 0.0])
    c = PointMass(0.5, q0=np.array([0.05, -0.12, 0.07]), u0=np.array([0.0, 1.0, 0.0]))
    system.add(floor, a, b, c)
    for body in (a, b, c):
        system.add(Force(np.array([0, 0, -10.0]) * body.mass, body))
    if world:
        shapes = [PlaneShape(floor)] + [SphereShape(x, r) for x in (a, b, c)]
        system.add(CollisionWorld(shapes, mu=MU))
    else:
        for x in (a, b, c):
            system.add(Sphere2Plane(floor, x, MU, r))
        for x, y in [(a, b), (a, c), (b, c)]:
            system.add(Sphere2Sphere(x, y, r, r, MU))
    system.assemble()
    return system


@pytest.mark.parametrize("Solver", [Moreau, DualStormerVerlet])
def test_solvers_match_pair_classes(Solver):
    def solve(world):
        return Solver(
            _small_scene(world),
            0.1,
            1e-3,
            options=SolverOptions(fixed_point_max_iter=int(1e4), prox_scaling=0.5),
        ).solve()

    sol, ref = solve(True), solve(False)
    P_N = getattr(sol, "la_N", None)
    P_N = sol.P_N if P_N is None or Solver is DualStormerVerlet else P_N
    assert np.abs(P_N).max() > 0  # contacts are active
    for name in ("q", "u"):
        assert np.allclose(
            getattr(sol, name), getattr(ref, name), rtol=1e-9, atol=1e-12
        )


def test_unsupported_solvers_raise():
    with pytest.raises(NotImplementedError):
        BackwardEuler(_small_scene(True), 0.01, 1e-3).solve()


def test_slots_and_filters():
    system = System()
    a, b, c = (
        PointMass(1.0, q0=np.array([k, 0.0, 0.0]), u0=np.zeros(3)) for k in range(3)
    )
    system.add(a, b, c)
    floor = system.origin
    shapes = [
        PlaneShape(floor),
        SphereShape(a, 0.1),
        SphereShape(a, 0.1, B_r_CS=np.array([0.0, 0.0, 0.2])),  # same body
        SphereShape(b, 0.1),
        SphereShape(c, 0.1),
    ]
    world = CollisionWorld(shapes, mu=0.0, exclude=[(shapes[3], shapes[4])])
    # 4 plane-sphere pairs, sphere pairs: (1,3), (1,4), (2,3), (2,4); (1,2) same
    # body, (3,4) excluded
    assert world.nla_N == 4 + 4
    assert not hasattr(world, "gamma_F")  # no friction for mu = 0
    system.add(world)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))
    assert system.nla_F == 0


def test_broad_phase_bound_for_skipped_pairs():
    margin = 0.04
    # two spheres with a gap just above the margin, a third far away
    radius = 0.05
    gap0 = 1.001 * margin
    x = np.array([[0.0, 0, 0], [2 * radius + gap0, 0, 0], [1.0, 1.0, 1.0]])
    system = System()
    bodies = [PointMass(1.0, q0=xi, u0=np.zeros(3)) for xi in x]
    system.add(*bodies)
    world = CollisionWorld([SphereShape(b, radius) for b in bodies], margin=margin)
    system.add(world)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))
    pair01 = 0  # slot of the sphere pair (0, 1)
    # the gap is above the margin, the pair is skipped
    assert system.g_N(0.0, system.q0)[pair01] == INACTIVE_GAP

    # worst case: both spheres move towards each other by just below margin / 4,
    # which does not trigger a rebuild; the pair stays skipped and its true gap is
    # still above margin / 2
    q = system.q0.copy()
    step = 0.999 * margin / 4
    q[0] += step
    q[3] -= step
    snapshot = world._snapshot.copy()
    assert system.g_N(0.0, q)[pair01] == INACTIVE_GAP
    assert np.array_equal(world._snapshot, snapshot)  # no rebuild
    true_gap = np.linalg.norm(q[3:6] - q[0:3]) - 2 * radius
    assert true_gap > margin / 2
    assert np.isclose(true_gap, gap0 - 2 * step)

    # moving both spheres by more than margin / 4 triggers a rebuild, the pair
    # becomes a candidate and its gap is evaluated exactly
    q = system.q0.copy()
    step = 1.01 * margin / 4
    q[0] += step
    q[3] -= step
    g = system.g_N(0.0, q)
    assert np.isclose(g[pair01], gap0 - 2 * step)

    # random scene: for random displacements just below margin / 4 no rebuild
    # happens and the gaps of all skipped pairs stay above margin / 2
    rng = np.random.default_rng(5)
    system, world, radii = _random_sphere_scene(60, rng, margin=margin)
    q = system.q0.copy()
    g = system.g_N(0.0, q)
    snapshot = world._snapshot.copy()
    for _ in range(20):
        direction = rng.standard_normal((len(radii), 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        q = (snapshot + 0.999 * margin / 4 * direction).ravel()  # no rebuild
        g = system.g_N(0.0, q)
        assert np.array_equal(world._snapshot, snapshot)  # list was not rebuilt
        exact = _all_pair_gaps(q, radii)
        skipped = g >= INACTIVE_GAP
        assert np.all(exact[skipped] > margin / 2)
        assert np.allclose(g[~skipped], exact[~skipped])
