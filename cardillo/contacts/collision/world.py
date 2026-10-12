import numpy as np
from scipy.spatial import cKDTree

from cardillo.math import ax2skew, cross3, norm
from cardillo.math.prox import Sphere
from cardillo.utility.cache import fast_key
from cardillo.utility.coo_matrix import CooMatrix

# gap reported for pairs that are not candidates of the broad phase
INACTIVE_GAP = 1.0e6


def _triangular_index(i, j, n):
    """Index of the pair (i, j), i < j, in the lexicographic list of all pairs
    of n objects."""
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


class CollisionWorld:
    """All contacts between a set of shapes as one contribution.

    Every pair of shapes that may collide gets a fixed contact slot with one
    normal contact and, if mu > 0, two friction forces (Coulomb). A broad phase
    selects candidate pairs; all other pairs report a large gap and are inactive.
    Quantities that involve velocities or force directions are only evaluated
    for the slots requested by the solver (``supports_active_set``).

    Supported pairs: sphere-sphere and plane-sphere. The
    formulas are the ones of `Sphere2Sphere` and `Sphere2Plane`. Second
    derivatives (required by BackwardEuler, Rattle and the static solvers) are
    not implemented yet; use Moreau or DualStormerVerlet.

    Slot order: all plane-sphere pairs (for each sphere, all planes), followed by
    all sphere-sphere pairs in lexicographic order of the spheres.

    Parameters
    ----------
    shapes : list of SphereShape or PlaneShape
    mu : float
        Friction coefficient of all pairs.
    e_N, e_F : float
        Restitution coefficients in normal and tangential direction.
    margin : float
        Distance below which sphere pairs become candidates of the broad phase.
        The broad phase is rebuilt as soon as a sphere moved more than margin / 4,
        so that every pair outside the candidates keeps a gap above margin / 2.
        The margin therefore only affects efficiency, not the result. Default: half the
        largest sphere radius (or 1e-2 for points only).
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
            if shape.kind not in ("sphere", "plane"):
                raise NotImplementedError(f"shape kind '{shape.kind}' not supported")
            if shape.xi is not None:
                raise NotImplementedError("shapes on parametrized subsystems (xi)")

        sphere_ids = [k for k, s in enumerate(self.shapes) if s.kind == "sphere"]
        plane_ids = [k for k, s in enumerate(self.shapes) if s.kind == "plane"]
        self._sphere_ids = np.array(sphere_ids, dtype=int)
        self._sphere_radius = np.array(
            [self.shapes[k].radius for k in sphere_ids], dtype=float
        )

        excluded = {frozenset((id(a), id(b))) for a, b in exclude}

        def allowed(a, b):
            sa, sb = self.shapes[a], self.shapes[b]
            return (
                sa.subsystem is not sb.subsystem
                and frozenset((id(sa), id(sb))) not in excluded
            )

        # plane-sphere slots
        ps_p, ps_s = [], []
        for s in sphere_ids:
            for p in plane_ids:
                if allowed(p, s):
                    ps_p.append(p)
                    ps_s.append(s)
        self._ps_p = np.array(ps_p, dtype=int)
        self._ps_s = np.array(ps_s, dtype=int)
        self._ps_r = np.array([self.shapes[s].radius for s in ps_s], dtype=float)
        self._nps = len(ps_p)

        # sphere-sphere slots, lookup from pairs of sphere indices to slots
        n = len(sphere_ids)
        self._ss_lookup = np.full(n * (n - 1) // 2, -1, dtype=int)
        ss_i, ss_j = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if allowed(sphere_ids[i], sphere_ids[j]):
                    self._ss_lookup[_triangular_index(i, j, n)] = len(ss_i)
                    ss_i.append(i)
                    ss_j.append(j)
        self._ss_a = self._sphere_ids[np.array(ss_i, dtype=int)]
        self._ss_b = self._sphere_ids[np.array(ss_j, dtype=int)]
        self._ss_r1 = self._sphere_radius[np.array(ss_i, dtype=int)]
        self._ss_r2 = self._sphere_radius[np.array(ss_j, dtype=int)]
        self._nss = len(ss_i)

        if margin is None:
            r_max = self._sphere_radius.max() if n > 0 else 0.0
            margin = 0.5 * r_max if r_max > 0 else 1e-2
        self.margin = float(margin)

        # contact slots
        self.nla_N = self._nps + self._nss
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
        self._snapshot = None

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
        self._nu = nu

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

        # reference contact bases of the sphere-sphere slots, see Sphere2Sphere
        q0 = np.concatenate([np.asarray(p, dtype=float) for p in q0_parts])
        R, _ = self._positions(self.t0, q0)
        self._basis = np.zeros((self._nss, 3, 3))
        if self._nss > 0:
            d = R[self._ss_b] - R[self._ss_a]
            n = d / np.sqrt(np.einsum("ki,ki->k", d, d))[:, None]
            vs = np.cross(n[:, None, :], np.eye(3)[None, :, :])
            norm_vs = np.linalg.norm(vs, axis=-1)
            axis = np.argmax(norm_vs, axis=1)
            k = np.arange(self._nss)
            t1 = vs[k, axis] / norm_vs[k, axis][:, None]
            w = np.cross(n, t1)
            t2 = w / np.sqrt(np.einsum("ki,ki->k", w, w))[:, None]
            self._basis = np.stack((t1, t2, n), axis=2)  # columns t1, t2, n

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
            qs, us, uds = (
                q[self._q_slice[k]],
                u[self._u_slice[k]],
                u_dot[self._u_slice[k]],
            )
            a[k] = sub.a_P(t, qs, us, uds, shape.xi, shape.B_r_CS)
            if self._rotational[k]:
                Psi[k] = sub.A_IB(t, qs, shape.xi) @ sub.B_Psi(t, qs, us, uds, shape.xi)
        return a, Psi

    #######################################################################
    # broad phase
    #######################################################################
    def _update_candidates(self, R, force=False):
        """Candidate sphere-sphere slots: pairs with a gap of at most the margin
        at the last rebuild. Rebuilt if a sphere moved by more than margin / 4.
        A pair outside the candidates had a gap above margin and the gap
        decreases by at most the sum of the displacements of its two spheres,
        i.e., by at most margin / 2: its gap stays above margin / 2."""
        Rs = R[self._sphere_ids]
        if not force and self._snapshot is not None:
            d = Rs - self._snapshot
            if (
                len(d) == 0
                or 4 * np.sqrt(np.einsum("ki,ki->k", d, d).max()) <= self.margin
            ):
                return
        self._snapshot = Rs.copy()
        candidate = np.zeros(self._nss, dtype=bool)
        n = len(Rs)
        if self._nss > 0:
            r = self._sphere_radius
            pairs = cKDTree(Rs).query_pairs(
                r=2 * r.max() + self.margin, output_type="ndarray"
            )
            if len(pairs) > 0:
                i, j = pairs[:, 0], pairs[:, 1]
                slots = self._ss_lookup[_triangular_index(i, j, n)]
                d = Rs[j] - Rs[i]
                close = (
                    np.sqrt(np.einsum("ki,ki->k", d, d)) - r[i] - r[j] <= self.margin
                )
                keep = close & (slots >= 0)
                candidate[slots[keep]] = True
        self._ss_candidates = np.nonzero(candidate)[0]

    def _candidate_slots(self, t, q):
        R, _ = self._positions(t, q)
        self._update_candidates(R)
        return np.concatenate((np.arange(self._nps), self._nps + self._ss_candidates))

    #######################################################################
    # per-slot geometry
    #######################################################################
    def _ss_normal(self, k, R):
        r_C1C2 = R[self._ss_b[k]] - R[self._ss_a[k]]
        return r_C1C2 / norm(r_C1C2)

    def _ss_tangents(self, k, n):
        # always use t2 of the reference basis as second direction
        t2_ref = self._basis[k, :, 1]
        v = cross3(t2_ref, n)
        t1 = v / norm(v)
        w = cross3(n, t1)
        t2 = w / norm(w)
        return t1, t2

    def _ps_geometry(self, k, R, A):
        p, s = self._ps_p[k], self._ps_s[k]
        A_IJ1 = A[p]
        n = A_IJ1[:, 2]
        r_J1J2 = R[s] - R[p]
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self._ps_r[k] * n
        return p, s, A_IJ1, n, r_J1J2, r_J1C1, r_J2C2

    def _ps_gamma(self, k, R, A, V, Omega):
        p, s, A_IJ1, n, r_J1J2, r_J1C1, r_J2C2 = self._ps_geometry(k, R, A)
        v_C1 = V[p] + cross3(Omega[p], r_J1C1)
        v_C2 = V[s] + cross3(Omega[s], r_J2C2)
        return A_IJ1.T @ (v_C2 - v_C1)

    def _ps_gamma_u(self, k, R, A, J_S, J_R):
        p, s, A_IJ1, n, r_J1J2, r_J1C1, r_J2C2 = self._ps_geometry(k, R, A)
        J_C1 = J_S[p] - np.cross(r_J1C1, J_R[p], axis=0)
        J_C2 = J_S[s] - np.cross(r_J2C2, J_R[s], axis=0)
        gamma_u = np.concatenate((-J_C1, J_C2), axis=1)
        rows = np.concatenate((self._u_rows[p], self._u_rows[s]))
        return A_IJ1.T @ gamma_u, rows

    def _ps_gamma_dot(self, k, R, A, V, Omega, a, Psi):
        p, s, A_IJ1, n, r_J1J2, r_J1C1, r_J2C2 = self._ps_geometry(k, R, A)
        v_J1, v_J2, Omega1, Omega2 = V[p], V[s], Omega[p], Omega[s]
        v_C1 = v_J1 + cross3(Omega1, r_J1C1)
        v_C2 = v_J2 + cross3(Omega2, r_J2C2)
        n_dot = cross3(Omega1, n)
        v_J1J2 = v_J2 - v_J1
        r_J1C1_dot = (
            v_J1J2 - n_dot * (n @ r_J1J2) - n * (n_dot @ r_J1J2) - n * (n @ v_J1J2)
        )
        r_J2C2_dot = -self._ps_r[k] * n_dot
        v_C1_dot = a[p] + cross3(Psi[p], r_J1C1) + cross3(Omega1, r_J1C1_dot)
        v_C2_dot = a[s] + cross3(Psi[s], r_J2C2) + cross3(Omega2, r_J2C2_dot)
        return A_IJ1.T @ (v_C2_dot - v_C1_dot - cross3(Omega1, v_C2 - v_C1))

    def _normal_slots(self, t, q, I_N):
        if I_N is None:
            return self._candidate_slots(t, q)
        return np.asarray(I_N, dtype=int)

    def _friction_slots(self, t, q, I_F):
        if I_F is None:
            return self._candidate_slots(t, q)
        return np.unique(np.asarray(I_F, dtype=int) // 2)

    #######################################################################
    # normal contact
    #######################################################################
    def g_N(self, t, q):
        R, A = self._positions(t, q)
        self._update_candidates(R)
        g_N = np.full(self.nla_N, INACTIVE_GAP)
        if self._nps > 0:
            n = A[self._ps_p, :, 2]
            d = R[self._ps_s] - R[self._ps_p]
            g_N[: self._nps] = np.einsum("ki,ki->k", n, d) - self._ps_r
        k = self._ss_candidates
        if len(k) > 0:
            d = R[self._ss_b[k]] - R[self._ss_a[k]]
            g_N[self._nps + k] = (
                np.sqrt(np.einsum("ki,ki->k", d, d)) - self._ss_r1[k] - self._ss_r2[k]
            )
        return g_N

    def g_N_dot(self, t, q, u, I_N=None):
        R, A = self._positions(t, q)
        V, Omega = self._velocities(t, q, u)
        g_N_dot = np.zeros(self.nla_N)
        for slot in self._normal_slots(t, q, I_N):
            if slot < self._nps:
                g_N_dot[slot] = self._ps_gamma(slot, R, A, V, Omega)[2]
            else:
                k = slot - self._nps
                n = self._ss_normal(k, R)
                g_N_dot[slot] = n @ (V[self._ss_b[k]] - V[self._ss_a[k]])
        return g_N_dot

    def W_N(self, t, q, I_N=None):
        R, A = self._positions(t, q)
        J_S, J_R = self._jacobians(t, q)
        rows, cols, data = [], [], []
        for slot in self._normal_slots(t, q, I_N):
            if slot < self._nps:
                gamma_u, r = self._ps_gamma_u(slot, R, A, J_S, J_R)
                rows.append(r)
                data.append(gamma_u[2])
            else:
                k = slot - self._nps
                a, b = self._ss_a[k], self._ss_b[k]
                n = self._ss_normal(k, R)
                rows.append(np.concatenate((self._u_rows[a], self._u_rows[b])))
                data.append(np.concatenate((-n @ J_S[a], n @ J_S[b])))
            cols.append(np.full(len(rows[-1]), slot))
        coo = CooMatrix((self._nu, self.nla_N))
        if rows:
            coo.add_entries(
                np.concatenate(rows), np.concatenate(cols), np.concatenate(data)
            )
        return coo

    def g_N_ddot(self, t, q, u, u_dot):
        R, A = self._positions(t, q)
        V, Omega = self._velocities(t, q, u)
        a, Psi = self._accelerations(t, q, u, u_dot)
        g_N_ddot = np.zeros(self.nla_N)
        for slot in self._candidate_slots(t, q):
            if slot < self._nps:
                g_N_ddot[slot] = self._ps_gamma_dot(slot, R, A, V, Omega, a, Psi)[2]
            else:
                k = slot - self._nps
                n = self._ss_normal(k, R)
                g_N_ddot[slot] = n @ (a[self._ss_b[k]] - a[self._ss_a[k]])
        return g_N_ddot

    #######################################################################
    # friction
    #######################################################################
    def _gamma_F(self, t, q, u, I_F=None):
        R, A = self._positions(t, q)
        V, Omega = self._velocities(t, q, u)
        gamma_F = np.zeros(self.nla_F)
        for slot in self._friction_slots(t, q, I_F):
            if slot < self._nps:
                gamma_F[2 * slot : 2 * slot + 2] = self._ps_gamma(slot, R, A, V, Omega)[
                    :2
                ]
            else:
                k = slot - self._nps
                a, b = self._ss_a[k], self._ss_b[k]
                n = self._ss_normal(k, R)
                t1, t2 = self._ss_tangents(k, n)
                v_P1 = V[a] + cross3(Omega[a], self._ss_r1[k] * n)
                v_P2 = V[b] + cross3(Omega[b], -self._ss_r2[k] * n)
                v_P1P2 = v_P2 - v_P1
                gamma_F[2 * slot] = t1 @ v_P1P2
                gamma_F[2 * slot + 1] = t2 @ v_P1P2
        return gamma_F

    def _W_F(self, t, q, I_F=None):
        R, A = self._positions(t, q)
        J_S, J_R = self._jacobians(t, q)
        rows, cols, data = [], [], []
        for slot in self._friction_slots(t, q, I_F):
            if slot < self._nps:
                gamma_u, r = self._ps_gamma_u(slot, R, A, J_S, J_R)
                for i in range(2):
                    rows.append(r)
                    cols.append(np.full(len(r), 2 * slot + i))
                    data.append(gamma_u[i])
            else:
                k = slot - self._nps
                a, b = self._ss_a[k], self._ss_b[k]
                n = self._ss_normal(k, R)
                t1, t2 = self._ss_tangents(k, n)
                J_P1 = J_S[a] - ax2skew(self._ss_r1[k] * n) @ J_R[a]
                J_P2 = J_S[b] - ax2skew(-self._ss_r2[k] * n) @ J_R[b]
                r = np.concatenate((self._u_rows[a], self._u_rows[b]))
                for i, t_i in enumerate((t1, t2)):
                    rows.append(r)
                    cols.append(np.full(len(r), 2 * slot + i))
                    data.append(np.concatenate((-t_i @ J_P1, t_i @ J_P2)))
        coo = CooMatrix((self._nu, self.nla_F))
        if rows:
            coo.add_entries(
                np.concatenate(rows), np.concatenate(cols), np.concatenate(data)
            )
        return coo

    def _gamma_F_dot(self, t, q, u, u_dot):
        R, A = self._positions(t, q)
        V, Omega = self._velocities(t, q, u)
        a, Psi = self._accelerations(t, q, u, u_dot)
        gamma_F_dot = np.zeros(self.nla_F)
        for slot in self._candidate_slots(t, q):
            if slot < self._nps:
                gamma_F_dot[2 * slot : 2 * slot + 2] = self._ps_gamma_dot(
                    slot, R, A, V, Omega, a, Psi
                )[:2]
            else:
                k = slot - self._nps
                i, j = self._ss_a[k], self._ss_b[k]
                n = self._ss_normal(k, R)
                t1, t2 = self._ss_tangents(k, n)
                a_P1 = a[i] + cross3(Psi[i], self._ss_r1[k] * n)
                a_P2 = a[j] + cross3(Psi[j], -self._ss_r2[k] * n)
                a_P1P2 = a_P2 - a_P1
                gamma_F_dot[2 * slot] = t1 @ a_P1P2
                gamma_F_dot[2 * slot + 1] = t2 @ a_P1P2
        return gamma_F_dot

    #######################################################################
    # step callback: transport of the contact bases and broad phase
    #######################################################################
    def step_callback(self, t, q, u):
        R, _ = self._positions(t, q)
        if self._nss > 0:
            # smallest rotation of the reference bases, see Crisfield1997 (16.107)
            d = R[self._ss_b] - R[self._ss_a]
            n = d / np.sqrt(np.einsum("ki,ki->k", d, d))[:, None]
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
