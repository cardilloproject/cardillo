"""Tests for the optimized code paths. Each of them is compared with the
straightforward implementation it replaces."""

import numpy as np
import pytest
from scipy.sparse import coo_array

from cardillo import System
from cardillo.contacts import Sphere2Sphere
from cardillo.discrete import PointMass, RigidBody, Sphere
from cardillo.math import Spurrier
from cardillo.solver import Moreau, SolverOptions
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
