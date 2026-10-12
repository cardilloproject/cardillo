from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from cardillo.math import ax2skew, cross3, norm
from cardillo.math.prox import Sphere
from cardillo.utility.cache import fast_key
from cardillo.utility.coo_matrix import CooMatrix

# gap reported for pairs that are not candidates of the broad phase
INACTIVE_GAP = 1.0e6

# distance below which the normal between two axes/points is considered undefined
_EPS_DIST = 1.0e-12

# slot types of body-body pairs
_SS, _SC, _CC = 0, 1, 2

# step size of the central difference used for acceleration-level quantities
_FD_STEP = 1.0e-5


def _triangular_index(i, j, n):
    """Index of the pair (i, j), i < j, in the lexicographic list of all pairs
    of n objects."""
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def _dot(a, b):
    return np.einsum("ki,ki->k", a, b)


def _unit(d, dist, fallback):
    """d / dist, with the (unit) fallback direction where dist is zero."""
    small = dist < _EPS_DIST
    safe = np.where(small, 1.0, dist)
    n = d / safe[:, None]
    if np.any(small):
        n[small] = fallback[small]
    return n


def _perpendicular(e):
    """Some unit vector perpendicular to each (unit) row of e."""
    axis = np.argmin(np.abs(e), axis=1)
    helper = np.zeros_like(e)
    helper[np.arange(len(e)), axis] = 1.0
    v = np.cross(e, helper)
    return v / np.linalg.norm(v, axis=1)[:, None]


class CollisionWorld:
    """All contacts between a set of shapes as one contribution.

    Every pair of shapes that may collide gets fixed contact slots with one
    normal contact and, if mu > 0, two friction forces (Coulomb). A broad phase
    selects candidate pairs; all other pairs report a large gap and are inactive.
    Quantities that involve velocities or force directions are only evaluated for
    the slots requested by the solver (``supports_active_set``).

    Supported pairs: plane-sphere, plane-capsule, sphere-sphere, sphere-capsule
    and capsule-capsule, see THEORY.md. A capsule on a plane has two slots, one
    for each end point of its axis. Second derivatives (required by
    BackwardEuler, Rattle and the static solvers) are not implemented yet; use
    Moreau or DualStormerVerlet.

    Slot order: plane-sphere pairs (for each sphere, all planes), plane-capsule
    pairs (for each capsule, all planes, for each plane both end points), then all
    other pairs in lexicographic order of the non-plane shapes (spheres first,
    then capsules, each in the order of the list of shapes).

    Parameters
    ----------
    shapes : list of SphereShape, CapsuleShape or PlaneShape
    mu : float
        Friction coefficient of all pairs.
    e_N, e_F : float
        Restitution coefficients in normal and tangential direction.
    margin : float
        Distance below which pairs of non-plane shapes become candidates of the
        broad phase (measured between bounding spheres). The broad phase is
        rebuilt as soon as a shape moved more than margin / 4, so that every pair
        outside the candidates keeps a gap above margin / 2. The margin therefore
        only affects efficiency, not the result. Default: half the largest
        bounding radius (or 1e-2 for points only).
    exclude : iterable of pairs of shapes
        Pairs that never collide. Shapes on the same subsystem never collide.
    name : str
    """

    supports_active_set = True

    def __init__(
        self,
        shapes,
        mu=0.0,
        e_N=None,
        e_F=None,
        margin=None,
        exclude=(),
        name="collision_world",
    ):
        self.shapes = list(shapes)
        self.name = name
        for shape in self.shapes:
            if shape.kind not in ("sphere", "plane", "capsule"):
                raise NotImplementedError(f"shape kind '{shape.kind}' not supported")
            if shape.xi is not None:
                raise NotImplementedError("shapes on parametrized subsystems (xi)")

        kinds = [s.kind for s in self.shapes]
        plane_ids = [k for k, kind in enumerate(kinds) if kind == "plane"]
        sphere_ids = [k for k, kind in enumerate(kinds) if kind == "sphere"]
        capsule_ids = [k for k, kind in enumerate(kinds) if kind == "capsule"]
        S = len(self.shapes)
        self._radius = np.zeros(S)
        self._half_length = np.zeros(S)
        for k in sphere_ids + capsule_ids:
            self._radius[k] = self.shapes[k].radius
        for k in capsule_ids:
            self._half_length[k] = self.shapes[k].half_length

        # non-plane shapes in canonical order: spheres first, then capsules
        self._bodies = np.array(sphere_ids + capsule_ids, dtype=int)
        self._bounding_radius = (
            self._radius[self._bodies] + self._half_length[self._bodies]
        )

        excluded = {frozenset((id(a), id(b))) for a, b in exclude}

        def allowed(a, b):
            sa, sb = self.shapes[a], self.shapes[b]
            return (
                sa.subsystem is not sb.subsystem
                and frozenset((id(sa), id(sb))) not in excluded
            )

        # plane slots: a sphere center (or a capsule end point) against a plane
        ps_p, ps_s, ps_o, ps_rho = [], [], [], []
        for s in sphere_ids:
            for p in plane_ids:
                if allowed(p, s):
                    ps_p.append(p)
                    ps_s.append(s)
                    ps_o.append(np.zeros(3))
                    ps_rho.append(self._radius[s])
        self._nps_sphere = len(ps_p)
        for c in capsule_ids:
            for p in plane_ids:
                if allowed(p, c):
                    for sign in (-1.0, 1.0):
                        ps_p.append(p)
                        ps_s.append(c)
                        ps_o.append(np.array([0.0, 0.0, sign * self._half_length[c]]))
                        ps_rho.append(self._radius[c])
        self._ps_p = np.array(ps_p, dtype=int)
        self._ps_s = np.array(ps_s, dtype=int)
        self._ps_o = np.array(ps_o, dtype=float).reshape(-1, 3)
        self._ps_rho = np.array(ps_rho, dtype=float)
        self._nps = len(ps_p)

        # pair slots of non-plane shapes, lookup from pairs of body indices
        nb = len(self._bodies)
        self._pair_lookup = np.full(nb * (nb - 1) // 2, -1, dtype=int)
        pair_i, pair_j = [], []
        for i in range(nb):
            for j in range(i + 1, nb):
                if allowed(self._bodies[i], self._bodies[j]):
                    self._pair_lookup[_triangular_index(i, j, nb)] = len(pair_i)
                    pair_i.append(i)
                    pair_j.append(j)
        pair_i = np.array(pair_i, dtype=int)
        pair_j = np.array(pair_j, dtype=int)
        self._pair_a = self._bodies[pair_i]
        self._pair_b = self._bodies[pair_j]
        kind_code = np.array([{"sphere": 0, "capsule": 1}.get(k, -1) for k in kinds])
        self._pair_type = (
            kind_code[self._pair_a] + kind_code[self._pair_b]
        )  # SS, SC, CC
        self._npair = len(pair_i)

        # shape of both sides of every slot
        self._slot_i1 = np.concatenate((self._ps_p, self._pair_a)).astype(int)
        self._slot_i2 = np.concatenate((self._ps_s, self._pair_b)).astype(int)

        if margin is None:
            b_max = self._bounding_radius.max() if nb > 0 else 0.0
            margin = 0.5 * b_max if b_max > 0 else 1e-2
        self.margin = float(margin)

        # contact slots
        self.nla_N = self._nps + self._npair
        self.e_N = np.zeros(self.nla_N) if e_N is None else e_N * np.ones(self.nla_N)
        if mu > 0:
            self.nla_F = 2 * self.nla_N
            self.e_F = (
                np.zeros(self.nla_F) if e_F is None else e_F * np.ones(self.nla_F)
            )
            law = Sphere(mu)
            self.friction_laws = [
                ([k], [2 * k, 2 * k + 1], law) for k in range(self.nla_N)
            ]
            self.gamma_F = self._gamma_F
            self.W_F = self._W_F
            self.gamma_F_dot = self._gamma_F_dot
            self.gamma_F_q = self._not_implemented
            self.Wla_F_q = self._not_implemented

        self._position_key = None
        self._velocity_key = None
        self._jacobian_key = None
        self._geometry_key = None
        self._snapshot = None
        self._pair_candidates = np.empty(0, dtype=int)

    #######################################################################
    # setup
    #######################################################################
    def assembler_callback(self):
        # generalized coordinates of all involved subsystems (each only once)
        subsystems = []
        for shape in self.shapes:
            if not any(shape.subsystem is s for s in subsystems):
                subsystems.append(shape.subsystem)
        q_parts, u_parts, q0_parts = [], [], []
        slices = {}
        nq = nu = 0
        for sub in subsystems:
            local_q = sub.local_qDOF_P(None)
            local_u = sub.local_uDOF_P(None)
            q_parts.append(sub.qDOF[local_q] if len(local_q) else local_q)
            u_parts.append(sub.uDOF[local_u] if len(local_u) else local_u)
            q0_parts.append(np.asarray(sub.q0)[local_q] if len(local_q) else [])
            slices[id(sub)] = (
                slice(nq, nq + len(local_q)),
                slice(nu, nu + len(local_u)),
            )
            nq += len(local_q)
            nu += len(local_u)
        self.qDOF = np.concatenate(q_parts).astype(int)
        self.uDOF = np.concatenate(u_parts).astype(int)
        self._nq = nq
        self._nu = nu
        self._subsystems = [(sub, *slices[id(sub)]) for sub in subsystems]

        self._q_slice, self._u_slice, self._u_rows, self._rotational = [], [], [], []
        for shape in self.shapes:
            q_slice, u_slice = slices[id(shape.subsystem)]
            self._q_slice.append(q_slice)
            self._u_slice.append(u_slice)
            self._u_rows.append(np.arange(u_slice.start, u_slice.stop))
            sub = shape.subsystem
            self._rotational.append(
                all(hasattr(sub, f) for f in ("A_IB", "B_Omega", "B_J_R", "B_Psi"))
            )

        # reference contact bases of the pair slots, columns t1, t2, n
        q0 = np.concatenate([np.asarray(p, dtype=float) for p in q0_parts])
        R, A = self._positions(self.t0, q0)
        self._basis = np.zeros((self._npair, 3, 3))
        if self._npair > 0:
            n = self._all_pair_normals(R, A)
            vs = np.cross(n[:, None, :], np.eye(3)[None, :, :])
            norm_vs = np.linalg.norm(vs, axis=-1)
            axis = np.argmax(norm_vs, axis=1)
            k = np.arange(self._npair)
            t1 = vs[k, axis] / norm_vs[k, axis][:, None]
            w = np.cross(n, t1)
            t2 = w / np.sqrt(_dot(w, w))[:, None]
            self._basis = np.stack((t1, t2, n), axis=2)

        self._update_candidates(R, force=True)

    #######################################################################
    # kinematics of the shapes (one evaluation per state)
    #######################################################################
    def _positions(self, t, q):
        key = fast_key(t, q)
        if key != self._position_key:
            S = len(self.shapes)
            R = np.empty((S, 3))
            A = np.empty((S, 3, 3))
            for k, shape in enumerate(self.shapes):
                sub = shape.subsystem
                qs = q[self._q_slice[k]]
                R[k] = sub.r_OP(t, qs, shape.xi, shape.B_r_CS)
                if self._rotational[k]:
                    A[k] = sub.A_IB(t, qs, shape.xi) @ shape.A_BS
                else:
                    A[k] = shape.A_BS
            self._position_key, self._R, self._A = key, R, A
        return self._R, self._A

    def _velocities(self, t, q, u):
        key = fast_key(t, q, u)
        if key != self._velocity_key:
            S = len(self.shapes)
            V = np.empty((S, 3))
            Omega = np.zeros((S, 3))
            for k, shape in enumerate(self.shapes):
                sub = shape.subsystem
                qs, us = q[self._q_slice[k]], u[self._u_slice[k]]
                V[k] = sub.v_P(t, qs, us, shape.xi, shape.B_r_CS)
                if self._rotational[k]:
                    Omega[k] = sub.A_IB(t, qs, shape.xi) @ sub.B_Omega(
                        t, qs, us, shape.xi
                    )
            self._velocity_key, self._V, self._Omega = key, V, Omega
        return self._V, self._Omega

    def _jacobians(self, t, q):
        key = fast_key(t, q)
        if key != self._jacobian_key:
            J_S, J_R = [], []
            for k, shape in enumerate(self.shapes):
                sub = shape.subsystem
                qs = q[self._q_slice[k]]
                J_S.append(sub.J_P(t, qs, shape.xi, shape.B_r_CS))
                if self._rotational[k]:
                    J_R.append(sub.A_IB(t, qs, shape.xi) @ sub.B_J_R(t, qs, shape.xi))
                else:
                    J_R.append(np.zeros((3, len(self._u_rows[k]))))
            self._jacobian_key, self._J_S, self._J_R = key, J_S, J_R
        return self._J_S, self._J_R

    def _accelerations(self, t, q, u, u_dot):
        S = len(self.shapes)
        a = np.empty((S, 3))
        Psi = np.zeros((S, 3))
        for k, shape in enumerate(self.shapes):
            sub = shape.subsystem
            qs = q[self._q_slice[k]]
            us, uds = u[self._u_slice[k]], u_dot[self._u_slice[k]]
            a[k] = sub.a_P(t, qs, us, uds, shape.xi, shape.B_r_CS)
            if self._rotational[k]:
                Psi[k] = sub.A_IB(t, qs, shape.xi) @ sub.B_Psi(t, qs, us, uds, shape.xi)
        return a, Psi

    #######################################################################
    # geometry of the contact slots
    #######################################################################
    def _ps_geometry(self, R, A):
        """Plane slots: sphere center (or capsule end point) against a plane.
        Returns gap, normal and the offsets c1, c2 from the shape frame origins to
        the contact points on the plane and on the sphere."""
        p, s = self._ps_p, self._ps_s
        n = A[p, :, 2]
        offset = np.einsum("kij,kj->ki", A[s], self._ps_o)
        r_12 = R[s] + offset - R[p]
        n_r12 = _dot(n, r_12)
        g = n_r12 - self._ps_rho
        c1 = r_12 - n * n_r12[:, None]
        c2 = offset - self._ps_rho[:, None] * n
        return g, n, c1, c2

    def _pair_geometry(self, k, R, A, normals_only=False):
        """Gap, normal and contact point offsets c1, c2 (from the origins of the
        shape frames) of the pair slots k. The normal points from shape a to
        shape b. For the segment based pairs the contact points are the closest
        points of the axes shifted by the radii along the normal. With
        ``normals_only`` only the normals are valid (cheaper)."""
        a, b, typ = self._pair_a[k], self._pair_b[k], self._pair_type[k]
        ra, rb = self._radius[a], self._radius[b]
        g = np.empty(len(k))
        n = np.empty((len(k), 3))
        c1 = np.empty((len(k), 3))
        c2 = np.empty((len(k), 3))

        m = typ == _SS
        if np.any(m):
            d = R[b[m]] - R[a[m]]
            dist = np.sqrt(_dot(d, d))
            n[m] = d / dist[:, None]
            if not normals_only:
                g[m] = dist - ra[m] - rb[m]
                c1[m] = ra[m, None] * n[m]
                c2[m] = -rb[m, None] * n[m]

        m = typ == _SC  # sphere a, capsule b
        if np.any(m):
            x = R[a[m]]
            e = A[b[m], :, 2]
            h = self._half_length[b[m]]
            u = np.clip(_dot(x - R[b[m]], e), -h, h)
            along = u[:, None] * e  # closest point on the axis relative to the capsule
            d = R[b[m]] + along - x
            dist = np.sqrt(_dot(d, d))
            nn = _unit(d, dist, _perpendicular(e))
            n[m] = nn
            if not normals_only:
                g[m] = dist - ra[m] - rb[m]
                c1[m] = ra[m, None] * nn
                c2[m] = along - rb[m, None] * nn

        m = typ == _CC
        if np.any(m):
            e1, e2 = A[a[m], :, 2], A[b[m], :, 2]
            h1, h2 = self._half_length[a[m]], self._half_length[b[m]]
            s_, t_ = self._segment_parameters(R[a[m]], e1, h1, R[b[m]], e2, h2)
            along1 = (s_ * 2 * h1 - h1)[:, None] * e1  # point on axis 1 relative to R_a
            along2 = (t_ * 2 * h2 - h2)[:, None] * e2
            d = R[b[m]] + along2 - R[a[m]] - along1
            dist = np.sqrt(_dot(d, d))
            cross = np.cross(e1, e2)
            cross_norm = np.sqrt(_dot(cross, cross))
            fallback = np.where(
                (cross_norm > 1e-8)[:, None],
                cross / np.where(cross_norm > 1e-8, cross_norm, 1.0)[:, None],
                _perpendicular(e1),
            )
            nn = _unit(d, dist, fallback)
            n[m] = nn
            if not normals_only:
                g[m] = dist - ra[m] - rb[m]
                c1[m] = along1 + ra[m, None] * nn
                c2[m] = along2 - rb[m, None] * nn
        return g, n, c1, c2

    @staticmethod
    def _segment_parameters(r1, e1, h1, r2, e2, h2):
        """Parameters s, t in [0, 1] of the closest points of two segments with
        centers r_i, unit directions e_i and half lengths h_i, see Ericson,
        Real-Time Collision Detection, Section 5.1.9. For (almost) parallel
        segments the middle of the overlap is used."""
        p1 = r1 - h1[:, None] * e1
        p2 = r2 - h2[:, None] * e2
        d1 = 2 * h1[:, None] * e1
        d2 = 2 * h2[:, None] * e2
        r = p1 - p2
        a = _dot(d1, d1)
        e = _dot(d2, d2)
        f = _dot(d2, r)
        c = _dot(d1, r)
        b = _dot(d1, d2)
        denom = a * e - b * b
        parallel = denom <= 1e-12 * a * e
        safe_denom = np.where(parallel, 1.0, denom)
        s = np.clip((b * f - c * e) / safe_denom, 0.0, 1.0)
        # parallel segments: middle of the overlap of the projections on segment 1
        s_a = np.clip(_dot(d1, p2 - p1) / a, 0.0, 1.0)  # projection of p2
        s_b = np.clip(_dot(d1, p2 + d2 - p1) / a, 0.0, 1.0)  # projection of p2 + d2
        s = np.where(parallel, 0.5 * (s_a + s_b), s)
        t = (b * s + f) / e
        t_low, t_high = t < 0, t > 1
        s = np.where(t_low, np.clip(-c / a, 0.0, 1.0), s)
        s = np.where(t_high, np.clip((b - c) / a, 0.0, 1.0), s)
        t = np.clip(t, 0.0, 1.0)
        return s, t

    def _all_pair_normals(self, R, A):
        _, n, _, _ = self._pair_geometry(
            np.arange(self._npair), R, A, normals_only=True
        )
        return n

    #######################################################################
    # broad phase
    #######################################################################
    def _update_candidates(self, R, force=False):
        """Candidate pair slots: pairs of non-plane shapes whose bounding spheres
        had a gap of at most the margin at the last rebuild. Rebuilt if a shape
        moved by more than margin / 4. A pair outside the candidates had a gap
        above margin and the gap decreases by at most the sum of the displacements
        of the two shapes, i.e., by at most margin / 2: its gap stays above
        margin / 2."""
        Rs = R[self._bodies]
        if not force and self._snapshot is not None:
            d = Rs - self._snapshot
            if len(d) == 0 or 4 * np.sqrt(_dot(d, d).max()) <= self.margin:
                return
        self._snapshot = Rs.copy()
        self._geometry_key = None
        candidate = np.zeros(self._npair, dtype=bool)
        n = len(Rs)
        if self._npair > 0:
            b = self._bounding_radius
            pairs = cKDTree(Rs).query_pairs(
                r=2 * b.max() + self.margin, output_type="ndarray"
            )
            if len(pairs) > 0:
                i, j = pairs[:, 0], pairs[:, 1]
                slots = self._pair_lookup[_triangular_index(i, j, n)]
                d = Rs[j] - Rs[i]
                close = np.sqrt(_dot(d, d)) - b[i] - b[j] <= self.margin
                keep = close & (slots >= 0)
                candidate[slots[keep]] = True
        self._pair_candidates = np.nonzero(candidate)[0]

    def _geometry(self, t, q):
        """Gaps, normals and contact point offsets of all candidate slots at the
        state (t, q), computed once per state."""
        key = fast_key(t, q)
        if key == self._geometry_key:
            return self._geom
        R, A = self._positions(t, q)
        self._update_candidates(R)  # resets the cached geometry if rebuilt
        N = self.nla_N
        g = np.full(N, INACTIVE_GAP)
        n = np.zeros((N, 3))
        c1 = np.zeros((N, 3))
        c2 = np.zeros((N, 3))
        if self._nps > 0:
            g[: self._nps], n[: self._nps], c1[: self._nps], c2[: self._nps] = (
                self._ps_geometry(R, A)
            )
        k = self._pair_candidates
        if len(k) > 0:
            slots = self._nps + k
            g[slots], n[slots], c1[slots], c2[slots] = self._pair_geometry(k, R, A)
        candidates = np.concatenate((np.arange(self._nps), self._nps + k))
        is_candidate = np.zeros(N, dtype=bool)
        is_candidate[candidates] = True
        self._geom = SimpleNamespace(
            g=g, n=n, c1=c1, c2=c2, candidates=candidates, is_candidate=is_candidate
        )
        self._geometry_key = key
        return self._geom

    def _normal_slots(self, t, q, I_N):
        geom = self._geometry(t, q)
        if I_N is None:
            return geom.candidates
        I_N = np.asarray(I_N, dtype=int)
        return I_N[geom.is_candidate[I_N]]

    def _friction_slots(self, t, q, I_F):
        geom = self._geometry(t, q)
        if I_F is None:
            return geom.candidates
        slots = np.unique(np.asarray(I_F, dtype=int) // 2)
        return slots[geom.is_candidate[slots]]

    def _tangents(self, slots, geom, A):
        """Tangent directions t1, t2 of the slots, shape (K, 2, 3). Plane slots use
        the first two axes of the plane frame, all other slots the transported
        reference basis, see THEORY.md."""
        t12 = np.empty((len(slots), 2, 3))
        is_plane = slots < self._nps
        if np.any(is_plane):
            p = self._ps_p[slots[is_plane]]
            t12[is_plane, 0] = A[p, :, 0]
            t12[is_plane, 1] = A[p, :, 1]
        pair = ~is_plane
        if np.any(pair):
            k = slots[pair] - self._nps
            n = geom.n[slots[pair]]
            # always use t2 of the reference basis as second direction
            v = np.cross(self._basis[k, :, 1], n)
            t1 = v / np.sqrt(_dot(v, v))[:, None]
            w = np.cross(n, t1)
            t12[pair, 0] = t1
            t12[pair, 1] = w / np.sqrt(_dot(w, w))[:, None]
        return t12

    #######################################################################
    # normal contact
    #######################################################################
    def g_N(self, t, q):
        return self._geometry(t, q).g.copy()

    def _relative_velocity(self, slots, geom, V, Omega):
        """Relative velocity of the contact points (material points of the two
        bodies) of the slots."""
        i1, i2 = self._slot_i1[slots], self._slot_i2[slots]
        v_P1 = V[i1] + np.cross(Omega[i1], geom.c1[slots])
        v_P2 = V[i2] + np.cross(Omega[i2], geom.c2[slots])
        return v_P2 - v_P1

    def g_N_dot(self, t, q, u, I_N=None):
        geom = self._geometry(t, q)
        V, Omega = self._velocities(t, q, u)
        slots = self._normal_slots(t, q, I_N)
        g_N_dot = np.zeros(self.nla_N)
        g_N_dot[slots] = _dot(
            geom.n[slots], self._relative_velocity(slots, geom, V, Omega)
        )
        return g_N_dot

    def _force_directions(self, slots, geom, J_S, J_R, directions):
        """Rows, values of the generalized force directions
        W = [-J_P1^T d; J_P2^T d] for each slot and each direction d in
        directions[slot] (shape (K, m, 3))."""
        rows, data = [], []
        for idx, slot in enumerate(slots):
            i1, i2 = self._slot_i1[slot], self._slot_i2[slot]
            J_P1 = J_S[i1] - np.cross(geom.c1[slot], J_R[i1], axis=0)
            J_P2 = J_S[i2] - np.cross(geom.c2[slot], J_R[i2], axis=0)
            rows.append(np.concatenate((self._u_rows[i1], self._u_rows[i2])))
            data.append(
                [np.concatenate((-d @ J_P1, d @ J_P2)) for d in directions[idx]]
            )
        return rows, data

    def W_N(self, t, q, I_N=None):
        geom = self._geometry(t, q)
        J_S, J_R = self._jacobians(t, q)
        slots = self._normal_slots(t, q, I_N)
        coo = CooMatrix((self._nu, self.nla_N))
        rows, data = self._force_directions(slots, geom, J_S, J_R, geom.n[slots, None])
        if len(slots) > 0:
            coo.add_entries(
                np.concatenate(rows),
                np.concatenate([np.full(len(r), s) for r, s in zip(rows, slots)]),
                np.concatenate([d[0] for d in data]),
            )
        return coo

    #######################################################################
    # friction
    #######################################################################
    def _gamma_F(self, t, q, u, I_F=None):
        geom = self._geometry(t, q)
        _, A = self._positions(t, q)
        V, Omega = self._velocities(t, q, u)
        slots = self._friction_slots(t, q, I_F)
        gamma_F = np.zeros(self.nla_F)
        if len(slots) > 0:
            t12 = self._tangents(slots, geom, A)
            gamma = np.einsum(
                "kmi,ki->km", t12, self._relative_velocity(slots, geom, V, Omega)
            )
            gamma_F[2 * slots] = gamma[:, 0]
            gamma_F[2 * slots + 1] = gamma[:, 1]
        return gamma_F

    def _W_F(self, t, q, I_F=None):
        geom = self._geometry(t, q)
        _, A = self._positions(t, q)
        J_S, J_R = self._jacobians(t, q)
        slots = self._friction_slots(t, q, I_F)
        coo = CooMatrix((self._nu, self.nla_F))
        if len(slots) > 0:
            t12 = self._tangents(slots, geom, A)
            rows, data = self._force_directions(slots, geom, J_S, J_R, t12)
            coo.add_entries(
                np.concatenate([np.tile(r, 2) for r in rows]),
                np.concatenate(
                    [np.repeat([2 * s, 2 * s + 1], len(r)) for r, s in zip(rows, slots)]
                ),
                np.concatenate([np.concatenate(d) for d in data]),
            )
        return coo

    #######################################################################
    # acceleration level
    #######################################################################
    # Plane-sphere and sphere-sphere slots use the formulas of the per-pair
    # classes Sphere2Plane and Sphere2Sphere. All other slots (end points of
    # capsules, segment based pairs) differentiate the velocity level quantities
    # along the motion with a central difference.
    def _exact_slots(self, slots):
        is_ps_sphere = slots < self._nps_sphere
        is_ss = np.zeros(len(slots), dtype=bool)
        pair = slots >= self._nps
        is_ss[pair] = self._pair_type[slots[pair] - self._nps] == _SS
        return slots[is_ps_sphere | is_ss], slots[~(is_ps_sphere | is_ss)]

    def _gamma_dot_exact(self, slot, geom, R, A, V, Omega, a, Psi):
        """d/dt (t_i . v_P1P2) and d/dt gamma_N of a plane-sphere (all three
        components, for plane tangents) or sphere-sphere slot as in the per-pair
        classes. Returns (gamma_N_dot, gamma_F_dot[2])."""
        i1, i2 = self._slot_i1[slot], self._slot_i2[slot]
        n = geom.n[slot]
        if slot < self._nps:
            A_J1 = A[i1]
            r_J1J2 = R[i2] - R[i1]
            r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
            r_J2C2 = -self._ps_rho[slot] * n
            v_J1, v_J2, Omega1, Omega2 = V[i1], V[i2], Omega[i1], Omega[i2]
            v_C1 = v_J1 + cross3(Omega1, r_J1C1)
            v_C2 = v_J2 + cross3(Omega2, r_J2C2)
            n_dot = cross3(Omega1, n)
            v_J1J2 = v_J2 - v_J1
            r_J1C1_dot = (
                v_J1J2 - n_dot * (n @ r_J1J2) - n * (n_dot @ r_J1J2) - n * (n @ v_J1J2)
            )
            r_J2C2_dot = -self._ps_rho[slot] * n_dot
            v_C1_dot = a[i1] + cross3(Psi[i1], r_J1C1) + cross3(Omega1, r_J1C1_dot)
            v_C2_dot = a[i2] + cross3(Psi[i2], r_J2C2) + cross3(Omega2, r_J2C2_dot)
            gamma_dot = A_J1.T @ (v_C2_dot - v_C1_dot - cross3(Omega1, v_C2 - v_C1))
            return gamma_dot[2], gamma_dot[:2]
        k = slot - self._nps
        t1, t2 = self._tangents(np.array([slot]), geom, A)[0]
        a_P1 = a[i1] + cross3(Psi[i1], self._radius[i1] * n)
        a_P2 = a[i2] + cross3(Psi[i2], -self._radius[i2] * n)
        a_P1P2 = a_P2 - a_P1
        return n @ (a[i2] - a[i1]), np.array([t1 @ a_P1P2, t2 @ a_P1P2])

    def _fd_time_derivative(self, fun, t, q, u, u_dot):
        """Central difference of fun(t, q, u) along the motion with
        dq/dt = q_dot(q, u) and du/dt = u_dot."""
        q_dot = np.zeros_like(q)
        for sub, q_slice, u_slice in self._subsystems:
            if q_slice.stop > q_slice.start:
                q_dot[q_slice] = sub.q_dot(t, q[q_slice], u[u_slice])
        h = _FD_STEP
        plus = fun(t + h, q + h * q_dot, u + h * u_dot)
        minus = fun(t - h, q - h * q_dot, u - h * u_dot)
        return (plus - minus) / (2 * h)

    def g_N_ddot(self, t, q, u, u_dot):
        geom = self._geometry(t, q)
        g_N_ddot = np.zeros(self.nla_N)
        exact, fd = self._exact_slots(geom.candidates)
        if len(exact) > 0:
            R, A = self._positions(t, q)
            V, Omega = self._velocities(t, q, u)
            a, Psi = self._accelerations(t, q, u, u_dot)
            for slot in exact:
                g_N_ddot[slot] = self._gamma_dot_exact(
                    slot, geom, R, A, V, Omega, a, Psi
                )[0]
        if len(fd) > 0:
            g_N_ddot[fd] = self._fd_time_derivative(
                lambda t_, q_, u_: self.g_N_dot(t_, q_, u_, I_N=fd)[fd], t, q, u, u_dot
            )
        return g_N_ddot

    def _gamma_F_dot(self, t, q, u, u_dot):
        geom = self._geometry(t, q)
        gamma_F_dot = np.zeros(self.nla_F)
        exact, fd = self._exact_slots(geom.candidates)
        if len(exact) > 0:
            R, A = self._positions(t, q)
            V, Omega = self._velocities(t, q, u)
            a, Psi = self._accelerations(t, q, u, u_dot)
            for slot in exact:
                gamma_F_dot[2 * slot : 2 * slot + 2] = self._gamma_dot_exact(
                    slot, geom, R, A, V, Omega, a, Psi
                )[1]
        if len(fd) > 0:
            I_F = np.concatenate((2 * fd, 2 * fd + 1))
            derivative = self._fd_time_derivative(
                lambda t_, q_, u_: self._gamma_F(t_, q_, u_, I_F=I_F)[I_F],
                t,
                q,
                u,
                u_dot,
            )
            gamma_F_dot[I_F] = derivative
        return gamma_F_dot

    #######################################################################
    # step callback: transport of the contact bases and broad phase
    #######################################################################
    def step_callback(self, t, q, u):
        R, A = self._positions(t, q)
        if self._npair > 0:
            # smallest rotation of the reference bases, see Crisfield1997 (16.107)
            n = self._all_pair_normals(R, A)
            B = self._basis
            b = np.einsum("kji,kj->ki", B, n)
            factor = (B[:, :, 2] + n) / (1 + b[:, 2])[:, None]
            self._basis = np.stack(
                (
                    B[:, :, 0] - b[:, 0, None] * factor,
                    B[:, :, 1] - b[:, 1, None] * factor,
                    n,
                ),
                axis=2,
            )
        self._update_candidates(R, force=True)
        return q, u

    #######################################################################
    # not yet implemented (Plan D, phase 5)
    #######################################################################
    def _not_implemented(self, *args, **kwargs):
        raise NotImplementedError(
            "CollisionWorld does not provide second derivatives yet; "
            "use Moreau or DualStormerVerlet."
        )

    g_N_q = _not_implemented
    Wla_N_q = _not_implemented
    g_N_dot_q = _not_implemented
    g_N_dot_u = _not_implemented
