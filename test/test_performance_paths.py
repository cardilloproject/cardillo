"""Tests for the optimized code paths. Each of them is compared with the
straightforward implementation it replaces."""

import numpy as np
import pytest
from scipy.sparse import coo_array

from cardillo import System
from cardillo.contacts import Sphere2Sphere
from cardillo.discrete import PointMass, RigidBody, Sphere
from cardillo.forces import Force
from cardillo.math.prox import Sphere as Sphere_prox
from cardillo.math import Spurrier
from cardillo.solver import (
    BackwardEuler,
    DualStormerVerlet,
    Moreau,
    Rattle,
    SolverOptions,
)
from cardillo.utility.cache import cached_last, fast_key
from cardillo.utility.coo_matrix import CooMatrix


def test_coo_matrix_matches_dense():
    rng = np.random.default_rng(0)
    n = 20
    ref = np.zeros((n, n))
    coo = CooMatrix((n, n))
    for _ in range(40):
        rows = rng.choice(n, 3, replace=False)
        cols = rng.choice(n, 4, replace=False)
        dense = rng.random((3, 4))
        coo[rows, cols] = dense
        ref[np.ix_(rows, cols)] += dense

        sparse = rng.random((3, 4)) * (rng.random((3, 4)) > 0.5)
        coo[rows, cols] = coo_array(sparse)
        ref[np.ix_(rows, cols)] += sparse

    nested = CooMatrix((3, 4))
    nested[np.arange(3), np.arange(4)] = np.ones((3, 4))
    coo[[1, 2, 3], [4, 5, 6, 7]] = nested
    ref[np.ix_([1, 2, 3], [4, 5, 6, 7])] += 1
    coo[0:2, 0:1] = np.array([[5.0], [6.0]])
    ref[0:2, 0:1] += np.array([[5.0], [6.0]])
    coo[3, 4] = 2.0
    ref[3, 4] += 2.0
    coo[5, 5] = None  # None is ignored

    for fmt in ("coo", "csr", "csc"):
        assert np.allclose(coo.asformat(fmt).toarray(), ref)
    # stored values are copied, later changes of the assigned array have no effect
    block = np.ones((2, 2))
    c = CooMatrix((2, 2))
    c[[0, 1], [0, 1]] = block
    block[:] = 7
    assert np.allclose(c.toarray(), 1)


def test_coo_matrix_empty_and_shapes():
    for shape in [(3, 0), (0, 3), (0, 0), (4, 5)]:
        for fmt in ("coo", "csr", "csc"):
            A = CooMatrix(shape).asformat(fmt)
            assert A.shape == shape and A.nnz == 0
    # shapes that are no tuple of ints are still accepted
    assert CooMatrix([2, 3]).shape == (2, 3)
    with pytest.raises(Exception):
        CooMatrix((-1, 2))


def test_coo_matrix_property_setters():
    coo = CooMatrix((3, 3))
    coo.data = np.array([1.0, 2.0, 3.0])
    coo.row = np.array([0, 1, 2])
    coo.col = np.array([2, 1, 0])
    assert np.allclose(coo.toarray(), np.fliplr(np.diag([1.0, 2.0, 3.0])))
    coo[0, 0] = 4.0  # chunks are appended after assigning properties
    assert coo.toarray()[0, 0] == 4.0


def test_fast_key_and_cached_last():
    q = np.array([1.0, 2.0, 3.0])
    assert fast_key(0.5, q) == fast_key(0.5, q.copy())
    assert fast_key(0.5, q) != fast_key(0.5, q + 1e-15)
    assert fast_key(0.5, q) != fast_key(0.6, q)
    assert fast_key(0.5, [1.0, 2.0, 3.0]) == fast_key(0.5, q)
    # non-contiguous arrays are keyed by their values
    assert fast_key(np.arange(6.0)[::2]) == fast_key(np.array([0.0, 2.0, 4.0]))

    class A:
        calls = 0

        @cached_last(key=lambda self, q: fast_key(q))
        def f(self, q):
            self.calls += 1
            return q * 2

    a, b = A(), A()
    r = a.f(q)
    assert a.f(q.copy()) is r and a.calls == 1  # hit
    a.f(q + 1)
    assert a.calls == 2  # miss
    b.f(q)
    assert b.calls == 1  # caches are per instance


def _rigid_body():
    q0 = np.concatenate([[0.1, 0.2, 0.3], Spurrier(np.eye(3))])
    rng = np.random.default_rng(1)
    p = rng.random(4) + 0.5
    q = np.concatenate([rng.random(3), p / np.linalg.norm(p)])
    return RigidBody(1.0, np.eye(3), q0, np.zeros(6)), q, rng.random(6)


def test_rigid_body_center_of_mass_fast_paths():
    body, q, u = _rigid_body()
    zero = np.zeros(3)
    offset = np.array([0.1, -0.2, 0.3])
    for B_r_CP in [zero, [0.0, 0.0, 0.0]]:
        assert np.array_equal(body.r_OP(0, q), body.r_OP(0, q, B_r_CP=B_r_CP))
        assert np.array_equal(body.v_P(0, q, u), body.v_P(0, q, u, B_r_CP=B_r_CP))
        assert np.array_equal(body.J_P(0, q), body.J_P(0, q, B_r_CP=B_r_CP))
    # reference for non-zero offsets
    A = body.A_IB(0, q)
    assert np.allclose(body.r_OP(0, q, B_r_CP=offset), q[:3] + A @ offset)
    J = body.J_P(0, q, B_r_CP=offset)
    assert np.allclose(J[:, :3], np.eye(3))
    assert not np.allclose(J[:, 3:], 0)
    # the cached results of a different offset must not leak
    assert np.array_equal(body.r_OP(0, q), q[:3])


def _sphere_system(radius=0.05):
    """Two rigid bodies and a point mass that collide with each other."""
    system = System()

    def rigid(r_OC, v):
        return Sphere(RigidBody)(
            radius=radius,
            density=1,
            subdivisions=1,
            q0=RigidBody.pose2q(np.array(r_OC), np.eye(3)),
            u0=np.concatenate([v, [0.0, 0.0, 30.0]]),
        )

    a = rigid([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    b = rigid([0.14, 0.04, 0.0], [0.0, 0.0, 0.0])
    c = Sphere(PointMass)(
        radius=radius,
        mass=0.5,
        q0=np.array([0.05, -0.1, 0.0]),
        u0=np.array([0.0, 1.0, 0.0]),
    )
    system.add(a, b, c)
    for i, j in [(a, b), (a, c), (b, c)]:
        system.add(Sphere2Sphere(i, j, radius, radius, 0.3, e_N=0, e_F=0))
    system.assemble()
    return system


def _solve(system, t1=0.1, dt=1e-3):
    solver = Moreau(
        system,
        t1,
        dt,
        options=SolverOptions(fixed_point_max_iter=int(1e4), prox_scaling=0.5),
    )
    return solver.solve()


def _assert_same_solution(a, b):
    for name in ("q", "u", "la_N", "la_F"):
        assert np.allclose(
            getattr(a, name), getattr(b, name), rtol=1e-9, atol=1e-11
        ), name


def test_sphere2sphere_batch_is_used_and_exact(monkeypatch):
    sol = _solve(_sphere_system())
    assert np.abs(sol.la_N).max() > 0 and np.abs(sol.la_F).max() > 0  # contacts active
    system = _sphere_system()
    assert len(system._System__batches) == 1
    assert len(system._System__g_N_single) == 0

    monkeypatch.setattr(Sphere2Sphere, "batch_class", None)
    system_ref = _sphere_system()
    assert len(system_ref._System__batches) == 0
    _assert_same_solution(sol, _solve(system_ref))

    q = system.q0
    assert np.allclose(
        system.g_N(0.0, q), system_ref.g_N(0.0, q), rtol=1e-14, atol=1e-14
    )


def test_active_set_subsets_equal_full_evaluation():
    system = _sphere_system()
    sol = _solve(system, t1=0.05)
    t, q, u = sol.t[-1], sol.q[-1], sol.u[-1]
    I_N = np.array([0, 2])
    I_F = np.array([0, 1, 4, 5])
    assert np.allclose(
        system.W_N(t, q, "csc", I_N=I_N)[:, I_N].toarray(),
        system.W_N(t, q, "csc")[:, I_N].toarray(),
    )
    assert np.allclose(
        system.W_F(t, q, "csc", I_F=I_F)[:, I_F].toarray(),
        system.W_F(t, q, "csc")[:, I_F].toarray(),
    )
    assert np.allclose(
        system.g_N_dot(t, q, u, I_N=I_N)[I_N], system.g_N_dot(t, q, u)[I_N]
    )
    assert np.allclose(
        system.gamma_F(t, q, u, I_F=I_F)[I_F], system.gamma_F(t, q, u)[I_F]
    )
    # entries outside of the active set are zero
    assert np.all(system.g_N_dot(t, q, u, I_N=I_N)[[1]] == 0)


def test_vectorized_friction_prox_matches_loop(monkeypatch):
    sol = _solve(_sphere_system())
    assert np.abs(sol.la_F).max() > 0

    def setup_loop_only(self):
        self._friction_rest = self.global_active_friction_laws
        self._friction_iF = None

    monkeypatch.setattr(Moreau, "_setup_friction_prox", setup_loop_only)
    _assert_same_solution(sol, _solve(_sphere_system()))


def _disable_active_set_evaluation(monkeypatch):
    """Make System ignore all active-set arguments, i.e., evaluate everything."""
    for name, kw in [
        ("W_N", "I_N"),
        ("W_F", "I_F"),
        ("g_N_dot", "I_N"),
        ("gamma_F", "I_F"),
        ("Wla_N_q", "I_N"),
        ("Wla_F_q", "I_F"),
        ("xi_N", "I_N"),
        ("xi_F", "I_N"),
    ]:
        original = getattr(System, name)

        def wrapper(self, *args, __f=original, __kw=kw, **kwargs):
            kwargs.pop(__kw, None)
            return __f(self, *args, **kwargs)

        monkeypatch.setattr(System, name, wrapper)
    original = System.gamma_F_contributions
    monkeypatch.setattr(
        System, "gamma_F_contributions", lambda self, I_N=None: original(self, None)
    )


@pytest.mark.parametrize("Solver", [BackwardEuler, Rattle, DualStormerVerlet])
def test_active_set_evaluation_in_solvers(Solver, monkeypatch):
    def solve():
        solver = Solver(
            _sphere_system(),
            0.06,
            2e-3,
            options=SolverOptions(fixed_point_max_iter=int(1e4), prox_scaling=0.5),
        )
        return solver.solve()

    sol = solve()
    P_N = sol.P_N if hasattr(sol, "P_N") else sol.la_N
    P_F = sol.P_F if hasattr(sol, "P_F") else sol.la_F
    assert np.abs(P_N).max() > 0 and np.abs(P_F).max() > 0  # contacts active

    _disable_active_set_evaluation(monkeypatch)
    ref = solve()
    for name in ("q", "u", "P_N", "P_F"):
        assert np.allclose(
            getattr(sol, name), getattr(ref, name), rtol=1e-9, atol=1e-11
        ), name


def test_sphere2sphere_cache_follows_reference_basis():
    system = _sphere_system()
    contr = system._System__batches[0].contrs[0]
    q = system.q0[contr.qDOF]
    first = contr.t1t2(0.0, q)
    # same position but a rotated reference basis, e.g., after a step callback
    n = contr.n(0.0, q)
    basis = contr.reference_contact_basis
    contr.reference_contact_basis = np.vstack(
        (basis[:, 1], -basis[:, 0], n)
    ).T  # t1, t2 rotated by 90 degrees
    second = contr.t1t2(0.0, q)
    assert not np.allclose(first[0], second[0])


def test_coo_matrix_add_entries():
    coo = CooMatrix((3, 4))
    coo.add_entries([0, 2, 2], [1, 3, 3], [1.0, 2.0, 0.5])  # duplicates are summed
    coo[np.array([1]), np.array([0])] = np.array([[4.0]])
    ref = np.zeros((3, 4))
    ref[0, 1], ref[2, 3], ref[1, 0] = 1.0, 2.5, 4.0
    for fmt in ("coo", "csr", "csc"):
        assert np.allclose(coo.asformat(fmt).toarray(), ref)
    data = np.array([1.0])
    coo.add_entries([0], [0], data)
    data[0] = 99.0  # entries are copied
    assert coo.toarray()[0, 0] == 1.0
    with pytest.raises(AssertionError):
        coo.add_entries([0, 1], [0], [1.0, 2.0])


class _MultiFloor:
    """Test double: a point mass in contact with several floors at different
    heights, i.e., one contribution with several normal contacts."""

    supports_active_set = True

    def __init__(self, subsystem, heights, name="multi_floor"):
        self.subsystem = subsystem
        self.heights = np.asarray(heights, dtype=float)
        self.nla_N = len(heights)
        self.e_N = np.zeros(self.nla_N)
        self.name = name
        self.received = []

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    def _active(self, I_N):
        self.received.append(None if I_N is None else np.array(I_N))
        return np.arange(self.nla_N) if I_N is None else I_N

    def g_N(self, t, q):
        return q[2] - self.heights

    def g_N_q(self, t, q):
        g_N_q = np.zeros((self.nla_N, 3))
        g_N_q[:, 2] = 1
        return g_N_q

    def W_N(self, t, q, I_N=None):
        W_N = np.zeros((3, self.nla_N))
        W_N[2, self._active(I_N)] = 1
        return W_N

    def g_N_dot(self, t, q, u, I_N=None):
        g_N_dot = np.zeros(self.nla_N)
        g_N_dot[self._active(I_N)] = u[2]
        return g_N_dot

    def g_N_ddot(self, t, q, u, u_dot):
        return np.full(self.nla_N, u_dot[2])

    def Wla_N_q(self, t, q, la_N, I_N=None):
        return np.zeros((3, 3))


def _multi_floor_system(supports_active_set):
    system = System()
    falling = PointMass(1.0, q0=np.array([0.0, 0.0, 0.05]), u0=np.zeros(3))
    resting = PointMass(1.0, q0=np.array([1.0, 0.0, 0.5]), u0=np.zeros(3))
    system.add(falling, resting)
    for body in (falling, resting):
        system.add(Force(np.array([0.0, 0.0, -10.0]), body))
    # a contribution without active set support first, so that the global and
    # local contact indices of the second one differ
    plain = _MultiFloor(resting, [0.0, -1.0], name="plain")
    plain.supports_active_set = False
    floors = _MultiFloor(falling, [0.03, -1.0, 0.0], name="floors")
    floors.supports_active_set = supports_active_set
    system.add(plain, floors)
    system.assemble()
    return system, plain, floors


def test_local_active_sets_are_passed_to_contributions():
    system, plain, floors = _multi_floor_system(True)
    t, q = system.t0, system.q0
    I_N = floors.la_NDOF[[0, 2]]  # global indices of the first and third floor
    plain.received.clear()
    floors.received.clear()
    W_N = system.W_N(t, q, format="csc", I_N=I_N).toarray()
    assert np.array_equal(floors.received[-1], [0, 2])  # local indices
    assert len(plain.received) == 0  # contribution without active contacts
    W_N_full = system.W_N(t, q, format="csc").toarray()
    assert np.allclose(W_N[:, I_N], W_N_full[:, I_N])
    assert np.all(W_N[:, floors.la_NDOF[1]] == 0)  # inactive contact is skipped
    # a mixed active set reaches both contributions; plain gets no kwargs
    plain.received.clear()
    system.g_N_dot(t, q, system.u0, I_N=np.array([plain.la_NDOF[0], I_N[0]]))
    assert plain.received[-1] is None
    assert np.array_equal(floors.received[-1], [0])


def test_local_active_sets_in_solvers():
    def solve(supports):
        system, _, floors = _multi_floor_system(supports)
        sol = Moreau(system, 0.15, 1e-3).solve()
        return sol, floors

    sol, floors = solve(True)
    assert np.abs(sol.la_N).max() > 0  # the falling mass hits the 0.03 floor
    assert any(r is not None for r in floors.received)
    ref, _ = solve(False)
    for name in ("q", "u", "la_N"):
        assert np.allclose(getattr(sol, name), getattr(ref, name), rtol=1e-12), name


def test_friction_prox_vectorized_matches_per_law():
    class SphereSubclass(Sphere_prox):
        """Same law, but not recognized by the vectorized path."""

    system = _sphere_system()
    system_ref = _sphere_system()
    for contr in system_ref.get_contribution_list("gamma_F"):
        contr.friction_laws = [
            (i_N, i_F, SphereSubclass(force.r))
            for i_N, i_F, force in contr.friction_laws
        ]
    system_ref.assemble()
    rng = np.random.default_rng(0)
    for _ in range(20):
        arg = rng.standard_normal(system.nla_F)
        P_N = rng.standard_normal(system.nla_N)  # includes non-positive values
        a = system.friction_prox(arg, P_N, 0.01)
        b = system_ref.friction_prox(arg, P_N, 0.01)
        assert np.allclose(a, b, rtol=1e-14, atol=0)
    # friction indices: no normal force, no friction (Sphere laws only)
    assert len(system.friction_indices(np.array([], dtype=int))) == 0
    assert len(system_ref.friction_indices(np.array([], dtype=int))) == system.nla_F
