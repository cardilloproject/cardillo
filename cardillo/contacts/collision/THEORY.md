# Theory of the collision world

This document describes the mathematics behind `CollisionWorld`
(`world.py`) and the shapes in `shapes.py`: what is computed, why the formulas
are correct, and which assumptions the implementation makes. Section 11 maps
every formula to the code.

Supported pairs so far: sphere-sphere and plane-sphere. The formulas are the
ones of the per-pair classes `Sphere2Sphere` and `Sphere2Plane`; the collision
world reproduces their results bitwise (see `test/test_collision_world.py`).

## 1. Notation

| Symbol | Meaning |
|---|---|
| $q, u$ | generalized positions and velocities of the whole system |
| $r_i,\ A_i$ | position and orientation of the frame of shape $i$ (inertial basis) |
| $v_i,\ \Omega_i$ | velocity of the frame origin and angular velocity of shape $i$ |
| $J_i,\ J_{R,i}$ | Jacobians: $v_i = J_i u$, $\Omega_i = J_{R,i} u$ |
| $a_i,\ \Psi_i$ | acceleration of the frame origin and angular acceleration |
| $\rho_i$ | radius of sphere $i$ |
| $g_N$ | normal gap (signed distance between the surfaces) |
| $n$ | contact normal |
| $t_1, t_2$ | tangent directions, $(t_1, t_2, n)$ is a right-handed orthonormal basis |
| $\gamma_N=\dot g_N,\ \gamma_F$ | relative velocity of the contact points in normal / tangential direction |
| $\Lambda_N, \Lambda_F$ | normal and friction percussions (or forces times $\Delta t$) |
| $W_N, W_F$ | generalized force directions: $\gamma_N = W_N^T u$, $\gamma_F = W_F^T u$ |
| $\mu$ | Coulomb friction coefficient |

## 2. Shapes and frames

A shape is a geometric primitive with a frame $S$ that is rigidly attached to a
subsystem (rigid body, point mass, frame). With the body-fixed offset
${}_B r_{CS}$ and orientation $A_{BS}$ the frame of shape $i$ is

$$
r_i = r_{OP}(t, q, {}_B r_{CS}),
\qquad
A_i = A_{IB}(t, q)\, A_{BS},
$$

evaluated by the subsystem. The kinematic quantities of the frame origin follow
from the subsystem's point kinematics `v_P`, `J_P`, `a_P` with the same offset.
The angular quantities are

$$
\Omega_i = A_{IB}\,{}_B\Omega,
\qquad
J_{R,i} = A_{IB}\,{}_B J_R,
\qquad
\Psi_i = A_{IB}\,{}_B\Psi,
$$

and are zero for subsystems without orientation (point masses). The offset
${}_B r_{CS}$ only moves the frame origin; $A_{BS}$ rotates the shape frame, which
matters for planes (Section 3.2). Orientation offsets have no influence on
spheres.

Shape kinds:

* **Sphere**: center at $r_i$, radius $\rho_i \ge 0$ (a point for $\rho_i=0$).
* **Plane**: passes through $r_i$, its normal is the third column of $A_i$
  (the shape $e_z$-axis); the half-space opposite to the normal is solid. The
  first two columns are the friction directions.

## 3. Contact kinematics

Both pair types use the same structure: contact points $P_1$ on shape 1 and
$P_2$ on shape 2 that coincide when the gap is zero, the gap along the normal,
and the velocity of the *material* points of the two bodies that pass through
$P_1$ and $P_2$.

### 3.1 Sphere-sphere

Distance vector of the centers and normal (from sphere 1 to sphere 2):

$$
d = r_2 - r_1,
\qquad
g_N = \lVert d \rVert - \rho_1 - \rho_2,
\qquad
n = \frac{d}{\lVert d \rVert}.
$$

Contact points $P_1 = r_1 + \rho_1 n$, $P_2 = r_2 - \rho_2 n$. Their
material velocities are

$$
v_{P_1} = v_1 + \Omega_1 \times (\rho_1 n),
\qquad
v_{P_2} = v_2 + \Omega_2 \times (-\rho_2 n),
$$

and the relative velocity $v_{P_1P_2} = v_{P_2} - v_{P_1}$ gives

$$
\gamma_N = \dot g_N = n \cdot (v_2 - v_1),
\qquad
\gamma_F = \begin{pmatrix} t_1 \cdot v_{P_1P_2} \\ t_2 \cdot v_{P_1P_2} \end{pmatrix}.
$$

The normal velocity is the time derivative of the gap, because
$\tfrac{d}{dt}\lVert d\rVert = n\cdot\dot d$, and rotations do not change the
distance of the centers. In $\gamma_N$ the angular velocities drop out since
$n\cdot(\Omega\times n)=0$.

Generalized force directions follow from $v_i = J_i u$,
$\Omega_i = J_{R,i} u$ and $\Omega\times r = -\tilde r\,\Omega$, where
$\tilde r$ is the skew matrix of $r$:

$$
J_{P_1} = J_1 - \widetilde{\rho_1 n}\,J_{R,1},
\qquad
J_{P_2} = J_2 + \widetilde{\rho_2 n}\,J_{R,2},
$$

$$
W_N = \begin{pmatrix} -J_1^T n \\ J_2^T n \end{pmatrix},
\qquad
W_{F,k} = \begin{pmatrix} -J_{P_1}^T t_k \\ J_{P_2}^T t_k \end{pmatrix},
\quad k = 1, 2,
$$

each stacked over the $u$-coordinates of the two subsystems. The generalized force of a contact
force $\Lambda_N n + \Lambda_{F,1} t_1 + \Lambda_{F,2} t_2$ acting on body 2 and
the opposite force on body 1 is $W_N\Lambda_N + W_F\Lambda_F$.

### 3.2 Plane-sphere

Let $A_1 = (t_1, t_2, n)$ be the orientation of the plane frame (body 1), $r_1$
a point of the plane, $r_2$ the sphere center (body 2). With
$r_{12} = r_2 - r_1$:

$$
g_N = n\cdot r_{12} - \rho,
\qquad
n = A_1 e_z .
$$

Contact points: $C_1$ is the projection of the sphere center onto the plane,
$C_2$ the point of the sphere closest to the plane,

$$
r_{1C_1} = r_{12} - n\,(n\cdot r_{12}),
\qquad
r_{2C_2} = -\rho\, n .
$$

Material velocities of the points of the two bodies passing through them:

$$
v_{C_1} = v_1 + \Omega_1\times r_{1C_1},
\qquad
v_{C_2} = v_2 + \Omega_2\times r_{2C_2},
$$

$$
\gamma = A_1^T (v_{C_2} - v_{C_1}),
\qquad
\gamma_F = \gamma_{1:2},
\quad
\gamma_N = \gamma_3 .
$$

The tangent directions are the *fixed* columns $t_1, t_2$ of the plane frame, so
no transport is needed. $\gamma_N$ equals $\dot g_N$ also for a moving plane: the
derivative of $g_N$ contains $\dot n\cdot r_{12}$ with $\dot n=\Omega_1\times n$, and
$-n\cdot(\Omega_1\times r_{1C_1}) = \dot n\cdot r_{12}$ appears in $\gamma_N$;
the term $n\cdot(\Omega_2\times(-\rho n))$ is zero.

Force directions: with $J_{C_1} = J_1 - \widetilde{r_{1C_1}}\,J_{R,1}$ and
$J_{C_2} = J_2 - \widetilde{r_{2C_2}}\,J_{R,2}$,

$$
\gamma_u = A_1^T\,\bigl(-J_{C_1},\ J_{C_2}\bigr),
\qquad
W_N = \gamma_{u,3}^T,
\qquad
W_{F,k} = \gamma_{u,k}^T .
$$

## 4. Tangent frames of sphere-sphere contacts

For the plane the tangents are fixed. For two spheres no body-fixed tangent
exists, so every slot keeps a *reference basis* $B = (t_1^{ref}, t_2^{ref}, n^{ref})$
that is transported with the normal.

**Transport at the end of every time step.** Given the new normal $n$, rotate
$B$ by the smallest rotation $R$ that maps the old third column $a = n^{ref}$
onto $n$ (Crisfield 1996, (16.107)). With $c = a\cdot n$ the rotation is

$$
R x = x - \frac{x\cdot(a+n)}{1 + c}\,(a+n) + 2\,(x\cdot a)\,n,
$$

which satisfies $Ra = n$ and is orthogonal. Applied to the two tangent columns
(which are orthogonal to $a$) it reduces to

$$
b = B^T n,
\qquad
\text{factor} = \frac{n^{ref} + n}{1 + b_3},
\qquad
t_i^{new} = t_i^{ref} - b_i\cdot\text{factor},
\quad i = 1, 2,
\qquad
n^{new} = n .
$$

The formula is singular for $b_3=-1$ (the normal flipped by $180^\circ$ in one
step). Because the basis is updated every step, the rotation per step is small
for any resolved simulation.

**Tangents at an evaluation point.** Inside a step the tangents are rebuilt from
the *current* normal and the second reference tangent:

$$
t_1 = \frac{t_2^{ref}\times n}{\lVert t_2^{ref}\times n\rVert},
\qquad
t_2 = \frac{n\times t_1}{\lVert n\times t_1\rVert}.
$$

This gives an orthonormal right-handed basis $(t_1, t_2, n)$ at every $q$,
close to the reference basis (whose normal differs from $n$ only by the
motion in one step). For isotropic Coulomb friction the physical result does
not depend on the choice of the tangent basis; it only has to be continuous in
$q$ and change little within a step, which is what the transport ensures. The
construction degenerates for $t_2^{ref}\parallel n$, i.e., if the normal turns
by about $90^\circ$ relative to the previous step's normal within a single step;
as for the transport, this does not happen for resolved motions.

**Initial basis.** At assembly the normal is computed from the initial
configuration; the tangent $t_1$ is chosen as the normalized cross product of $n$
with the coordinate axis that makes the largest angle with it, and
$t_2 = n\times t_1$ (normalized).

## 5. Contact laws as used by the solvers

The collision world only provides the data. For completeness, the laws that the
solvers apply to these data (all slots are independent):

**Signorini condition** (unilateral contact) on position and velocity level,
with the Newton-type impact law

$$
0 \le g_N \perp \Lambda_N \ge 0,
\qquad
\xi_N = \gamma_N^+ + e_N\,\gamma_N^- ,
\qquad
0 \le \xi_N \perp \Lambda_N \ge 0 \text{ if } g_N \le 0 ,
$$

**Coulomb friction** with the force reservoir (the disc
$C(\Lambda_N) = \{\Lambda\in\mathbb R^2 : \lVert\Lambda\rVert \le \mu\Lambda_N\}$)

$$
-\xi_F \in N_{C(\Lambda_N)}(\Lambda_F),
\qquad
\xi_F = \gamma_F^+ + e_F\,\gamma_F^- ,
$$

where $e_N, e_F$ are the restitution coefficients of the slot (`e_N`, `e_F`).

In the proximal-point form that the solvers iterate, with relaxation
parameters $r_N, r_F>0$:

$$
\Lambda_N = -\operatorname{prox}_{\mathbb R^-}(r_N\xi_N - \Lambda_N),
\qquad
\Lambda_F = -\operatorname{prox}_{C(\Lambda_N)}(r_F\xi_F - \Lambda_F).
$$

The world exposes `friction_laws = [([k], [2k, 2k+1], Sphere(mu))]` for slot
$k$: normal force index, the two friction indices, and the force reservoir
`Sphere(mu)`, a disc of radius $\mu\Lambda_N$. Since the disc is symmetric,
$-\operatorname{prox}_C(x) = \operatorname{proj}_C(-x)$, and the iteration is
the usual projection $\Lambda_F = \operatorname{proj}_C(\Lambda_F - r_F\xi_F)$.

## 6. Active sets and exactness of the restricted evaluation

Evaluating velocity-level quantities and force directions is the expensive
part. They are only evaluated for slots the solver asks for. This is exact
because of two facts:

1. **Columns of inactive contacts are never used.** The solvers multiply
   $W_N$ and $W_F$ with percussions that are zero for inactive contacts, or
   slice them to the active columns (Moreau). Skipping the evaluation of the
   other columns changes nothing.
2. **Friction vanishes without normal force.** For a normal percussion
   $\Lambda_N \le 0$ the force reservoir is $C = \{0\}$ (radius
   $\max(0, \mu\Lambda_N) = 0$), so
   $\Lambda_F = -\operatorname{prox}_{\{0\}}(\cdot) = 0$ whatever $\xi_F$
   is. $\gamma_F$ and $W_F$ are therefore only needed for slots with a positive
   normal percussion. `System.friction_indices` computes these indices and
   `System.friction_prox` applies the vectorized projection.

The world declares `supports_active_set = True`; `System` then passes the slot
indices *local to the contribution* as `I_N` (for normal quantities) or `I_F`
(for friction quantities, in friction indices $2k, 2k+1$). Slots that are not
requested stay zero.

## 7. Broad phase

The number of slots grows quadratically with the number of spheres. Almost all
sphere pairs are far apart, so their gap is not needed exactly: it is only needed
to decide that they are inactive.

**Candidates.** A pair is a candidate if the gap of the sphere surfaces at the
last rebuild was at most `margin`. Candidates are found with a `cKDTree` on the
sphere centers: all pairs closer than $2\rho_{max} + \text{margin}$, filtered by
the exact gap. Gaps of candidate pairs are computed exactly. All other sphere
pairs report the constant gap `INACTIVE_GAP = 1e6`; plane-sphere pairs are always
evaluated.

**Rebuild.** The candidate list is rebuilt

* at assembly,
* in `step_callback` after every accepted time step, and
* before evaluating gaps whenever some sphere center has moved more than
  `margin / 2` since the last rebuild (solvers evaluate the midpoint and
  iterates, which can differ from the last accepted state).

**Guarantee.** After a rebuild the gap of a non-candidate pair satisfies
$g_0 > \text{margin}$. If every center moved by at most $\text{margin}/2$ since
then, the gap decreased by at most $|\Delta_i| + |\Delta_j| \le \text{margin}$,
hence $g > 0$: a non-candidate pair is still strictly separated, i.e.
inactive, and reporting a large gap does not change the active set. The
guarantee is strict positivity, not a quantitative lower bound: a pair whose
true gap is positive but smaller than the solver's closeness tolerance may be
treated as inactive. Choosing a larger margin makes this unlikely; the default
is half the largest radius.

The margin therefore only affects efficiency.

## 8. Slots and data layout

* One normal slot per allowed pair, `nla_N` in total. With friction
  ($\mu > 0$) there are two friction forces per slot, `nla_F = 2 nla_N`, in the
  order $(2k, 2k+1)$ for slot $k$.
* **Slot order**: first all plane-sphere pairs (for each sphere, all planes in
  the order of the shape list), then all sphere-sphere pairs in lexicographic
  order of the spheres. Pairs on the same subsystem and pairs listed in
  `exclude` get no slot.
* Slots are fixed after assembly, so warm starts and solution arrays have a
  constant size. The pair index of the sphere pair $(i, j)$, $i<j$, among $n$
  spheres is $i(2n-i-1)/2 + (j-i-1)$ before filtering; a lookup table maps it to
  the slot.
* **Coordinates**: `qDOF`/`uDOF` of the contribution are the concatenation of the
  coordinates of all involved subsystems (each once). Shapes keep slices into
  these arrays.
* **Sparse output**: `W_N`, `W_F` are returned as `CooMatrix` with local row
  indices (`u`-coordinates of the contribution) and global slot columns, built
  from flat arrays with `CooMatrix.add_entries`. Entries of the same coordinate
  from several contacts are summed on conversion.
* **Per-state cache**: positions, velocities and Jacobians of all shapes are
  evaluated once per state and reused by all quantities at the same
  $(t, q, u)$ (the key is the raw bytes of the arrays).

## 9. Acceleration-level quantities

`g_N_ddot` and `gamma_F_dot` (used only for the computation of consistent
initial conditions) are copied from the per-pair classes. For sphere-sphere
they are $n\cdot(a_2-a_1)$ and $t_k\cdot(a_{P_2}-a_{P_1})$ with
$a_{P_1} = a_1 + \Psi_1\times(\rho_1 n)$, $a_{P_2} = a_2 + \Psi_2\times(-\rho_2 n)$;
for plane-sphere they contain the derivative of the contact points and the
normal. These formulas are inherited unchanged from `Sphere2Sphere` and
`Sphere2Plane` and have not been re-derived or verified independently here; the
world reproduces them exactly.

## 10. What is not covered yet

* **Second derivatives** ($g_{N,q}$, $\partial(W_N\Lambda_N)/\partial q$, $\gamma_{F,q}$,
  $\partial(W_F\Lambda_F)/\partial q$) are required by the Newton-based solvers
  (BackwardEuler, Rattle, static solvers). The world raises
  `NotImplementedError` for them; use Moreau or DualStormerVerlet.
* **Other shapes** (capsules, boxes, ...) and contact manifolds with several
  points per pair are planned in `SPEED_UP_IDEAS.md` (Plan D, phases 3 and 4).
* **Parametrized subsystems** (rods with $\xi$) are not supported.

## 11. Code map

| Formula / concept | Code (`world.py`) |
|---|---|
| shape frame pose, velocity, Jacobian, acceleration | `_positions`, `_velocities`, `_jacobians`, `_accelerations` |
| sphere-sphere gap | `g_N` (vectorized over candidate slots) |
| sphere-sphere normal and normal velocity | `_ss_normal`, `g_N_dot` |
| sphere-sphere tangents | `_ss_tangents` |
| sphere-sphere $\gamma_F$, $W_F$ | `_gamma_F`, `_W_F` |
| plane-sphere geometry, $\gamma$, $\gamma_u$, $\dot\gamma$ | `_ps_geometry`, `_ps_gamma`, `_ps_gamma_u`, `_ps_gamma_dot` |
| $W_N$ | `W_N` |
| transport of the reference basis | `step_callback` |
| initial reference basis | `assembler_callback` |
| broad phase | `_update_candidates`, `_candidate_slots` |
| active-set slots | `_normal_slots`, `_friction_slots` |
| friction projection and active friction indices | `System.friction_prox`, `System.friction_indices` (`system.py`) |

## References

* M. A. Crisfield, *Non-linear Finite Element Analysis of Solids and Structures,
  Vol. 2: Advanced Topics*, Wiley, 1996, Section 16.13 (smallest rotation).
* V. Acary, B. Brogliato, *Numerical Methods for Nonsmooth Dynamical Systems*,
  Springer, 2008 (set-valued contact laws, proximal formulation).
* C. Studer, *Numerics of Unilateral Contacts and Friction*, Springer, 2009;
  see also the references in `cardillo/math/prox.py`.
