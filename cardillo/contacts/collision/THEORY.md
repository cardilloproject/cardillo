# Theory of the collision world

This document describes the mathematics behind `CollisionWorld`
(`world.py`) and the shapes in `shapes.py`: what is computed, why the formulas
are correct, and which assumptions the implementation makes. Section 12 maps
every formula to the code.

Supported pairs: plane-sphere, plane-capsule, sphere-sphere, sphere-capsule and
capsule-capsule. For sphere-sphere and plane-sphere the formulas are the ones of
the per-pair classes `Sphere2Sphere` and `Sphere2Plane`; the collision world
reproduces their results to rounding level (see `test/test_collision_world.py`).

## 1. Notation

| Symbol | Meaning |
|---|---|
| $q, u$ | generalized positions and velocities of the whole system |
| $r_i,\ A_i$ | position and orientation of the frame of shape $i$ (inertial basis) |
| $v_i,\ \Omega_i$ | velocity of the frame origin and angular velocity of shape $i$ |
| $J_i,\ J_{R,i}$ | Jacobians: $v_i = J_i u$, $\Omega_i = J_{R,i} u$ |
| $a_i,\ \Psi_i$ | acceleration of the frame origin and angular acceleration |
| $\rho_i,\ h_i$ | radius of a sphere or capsule, half length of a capsule axis |
| $e_i$ | unit axis of capsule $i$, the third column of $A_i$ |
| $g_N$ | normal gap (signed distance between the surfaces) |
| $n$ | contact normal, pointing from shape 1 to shape 2 |
| $P_1, P_2$ | contact (witness) points on shape 1 and 2 |
| $c_1, c_2$ | offsets from the frame origins to the contact points: $c_i = P_i - r_i$ |
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

and are zero for subsystems without orientation (point masses; their shapes
keep the orientation $A_{BS}$). The offset ${}_B r_{CS}$ only moves the frame
origin; $A_{BS}$ rotates the shape frame, which matters for planes and capsules.

Shape kinds:

* **Sphere**: center at $r_i$, radius $\rho_i \ge 0$ (a point for $\rho_i=0$).
* **Plane**: passes through $r_i$, its normal is the third column of $A_i$
  (the shape $e_z$-axis); the half-space opposite to the normal is solid. The
  first two columns are the friction directions.
* **Capsule**: all points within the distance $\rho_i$ of the axis segment
  $\{ r_i + s\, e_i : |s| \le h_i \}$ with $e_i = A_i e_z$ and $h_i > 0$.

## 3. Contact kinematics

### 3.1 Generic form: contact points with offsets

All pair kernels produce the same four quantities for a slot: the gap $g_N$, the
normal $n$ (from shape 1 to shape 2) and the offsets $c_1, c_2$ from the frame
origins of the two shapes to the contact points $P_1 = r_1 + c_1$ and
$P_2 = r_2 + c_2$. Everything that involves velocities then follows from
rigid-body kinematics of the *material points* of the two bodies that pass
through $P_1$ and $P_2$:

$$
v_{P_i} = v_i + \Omega_i \times c_i,
\qquad
J_{P_i} = J_i - \tilde c_i\,J_{R,i},
$$

where $\tilde c$ is the skew matrix of $c$ (using $\Omega\times c = -\tilde c\,\Omega$).
With $v_{P_1P_2} = v_{P_2} - v_{P_1}$:

$$
\gamma_N = n\cdot v_{P_1P_2},
\qquad
\gamma_{F,k} = t_k\cdot v_{P_1P_2},
\quad k = 1, 2,
$$

$$
W_N = \begin{pmatrix} -J_{P_1}^T n \\ J_{P_2}^T n \end{pmatrix},
\qquad
W_{F,k} = \begin{pmatrix} -J_{P_1}^T t_k \\ J_{P_2}^T t_k \end{pmatrix},
$$

each stacked over the $u$-coordinates of the two subsystems. The generalized force
of a contact force $\Lambda_N n + \Lambda_{F,1} t_1 + \Lambda_{F,2} t_2$ acting on
body 2 and the opposite force on body 1 is $W_N\Lambda_N + W_F\Lambda_F$.

**Why $\gamma_N$ is the derivative of the gap (moving contact points).** For
sphere and plane pairs the contact points move with the bodies only through the
normal. For capsules the closest points move *along the axes* while the bodies
move. The gap is a minimum over body-fixed parameters $s$ (positions along the
axes),

$$
g_N(q) = \min_{s \in K}\ \varphi(q, s),
\qquad
\varphi = \lVert Q_2(q, s_2) - Q_1(q, s_1)\rVert - \rho_1 - \rho_2 ,
$$

where $K$ is a box (the segment bounds) and $Q_i(q, s_i)$ are material points of the
bodies. The envelope theorem (Danskin) states that the derivative of the minimum
equals the partial derivative of $\varphi$ at the minimizer $s^\ast$ with $s$ held
fixed, also if some parameters sit at their bounds (the bounds do not depend on $q$).
For fixed $s^\ast$ the points $Q_i$ are material points and

$$
\dot g_N = n\cdot (v_{Q_2} - v_{Q_1}) = n\cdot (v_{P_2} - v_{P_1}),
$$

because $P_i = Q_i \pm \rho_i n$ differ from $Q_i$ by a vector along $n$, whose
rotational velocity is perpendicular to $n$. This proves $\gamma_N = \dot g_N$ for
all pairs below whenever the minimizer is unique; the test
`test_normal_velocity_is_derivative_of_gap` checks it numerically for every pair
type by central differences of the gap along the motion. The tangential velocity
$\gamma_F$ is the relative velocity of the material points that coincide with the
contact points at the current instant, which is the usual definition for sliding
and rolling contacts.

The unique minimizer requirement fails for parallel capsule axes (Section 3.6).

### 3.2 Sphere-sphere

Distance vector of the centers, gap and normal:

$$
d = r_2 - r_1,
\qquad
g_N = \lVert d \rVert - \rho_1 - \rho_2,
\qquad
n = \frac{d}{\lVert d \rVert},
$$

contact points $P_1 = r_1 + \rho_1 n$, $P_2 = r_2 - \rho_2 n$, i.e.

$$
c_1 = \rho_1 n,
\qquad
c_2 = -\rho_2 n .
$$

The normal velocity is $\tfrac{d}{dt}\lVert d\rVert = n\cdot(v_2 - v_1)$, in agreement
with the generic formula since $n\cdot(\Omega\times n) = 0$.

### 3.3 Plane-sphere

Let $A_1 = (t_1, t_2, n)$ be the orientation of the plane frame (body 1), $r_1$ a
point of the plane, $r_2$ the sphere center (body 2) and $r_{12} = r_2 - r_1$:

$$
n = A_1 e_z,
\qquad
g_N = n\cdot r_{12} - \rho .
$$

The contact point $P_1$ on the plane is the projection of the sphere center, $P_2$
the sphere point closest to the plane:

$$
c_1 = r_{12} - n\,(n\cdot r_{12}),
\qquad
c_2 = -\rho\, n .
$$

The tangent directions are the *fixed* columns $t_1, t_2$ of the plane frame. The
generic $\gamma_N$ equals $\dot g_N$ also for a moving plane: the derivative of $g_N$
contains $\dot n\cdot r_{12}$ with $\dot n = \Omega_1\times n$, and
$-n\cdot(\Omega_1\times c_1) = \dot n\cdot r_{12}$ appears in $\gamma_N$.

### 3.4 Plane-capsule: two end point slots

The distance of a point to the plane is affine along the capsule axis, so the
capsule is closest to the plane at one of its two end points (or along the whole
axis if it is parallel to the plane). A capsule on a plane therefore has **two
slots**, one for each axis end point $x_\pm = r + A\,o_\pm$ with
$o_\pm = (0, 0, \pm h)^T$ in the shape frame, each behaving like a sphere of radius
$\rho$ around $x_\pm$:

$$
r_{12} = x_\pm - r_{plane},
\qquad
g_N = n\cdot r_{12} - \rho,
\qquad
c_1 = r_{12} - n\,(n\cdot r_{12}),
\qquad
c_2 = A\,o_\pm - \rho\, n .
$$

For a capsule lying parallel to the plane both slots are active and any resultant
force acting on the line contact is represented by the two end point forces; for a
tilted capsule only the lower end touches. This is exact for the plane (no
approximation of the contact set).

### 3.5 Sphere-capsule

Sphere center $x = r_a$, capsule with center $r_b$, unit axis $e$ and half length $h$
(shape $a$ = sphere, shape $b$ = capsule). The closest axis point is the clamped
projection of the center,

$$
u^\ast = \operatorname{clip}\bigl((x - r_b)\cdot e,\ -h,\ h\bigr),
\qquad
Q = r_b + u^\ast e,
\qquad
d = Q - x,
\qquad
n = \frac{d}{\lVert d\rVert},
$$

$$
g_N = \lVert d\rVert - \rho_a - \rho_b,
\qquad
c_1 = \rho_a n,
\qquad
c_2 = u^\ast e - \rho_b n .
$$

If the center lies on the axis ($\lVert d\rVert < 10^{-12}$) the normal is undefined;
an arbitrary unit vector perpendicular to the axis is used.

### 3.6 Capsule-capsule

Segment 1: $Q_1(s) = p_1 + s\,d_1$ with $p_1 = r_a - h_a e_a$, $d_1 = 2h_a e_a$,
$s\in[0,1]$; segment 2 likewise with $t\in[0,1]$. The squared distance
$\lVert Q_2(t) - Q_1(s)\rVert^2$ is a convex quadratic in $(s, t)$ on the unit square.
Its minimizer is computed in closed form (Ericson, *Real-Time Collision
Detection*, Section 5.1.9): the unconstrained minimizer for $s$ clamped to $[0,1]$,
then $t = (b s + f)/e$ with the usual quadratic coefficients; if $t$ leaves
$[0, 1]$ it is clamped and $s$ recomputed and clamped again. With the closest
points $Q_1(s^\ast), Q_2(t^\ast)$:

$$
d = Q_2 - Q_1,
\qquad
n = \frac{d}{\lVert d\rVert},
\qquad
g_N = \lVert d\rVert - \rho_a - \rho_b,
$$

$$
c_1 = (Q_1 - r_a) + \rho_a n,
\qquad
c_2 = (Q_2 - r_b) - \rho_b n .
$$

**Degenerate cases.**

* *Parallel axes* (the determinant of the quadratic is below $10^{-12}$ of its scale):
  the minimizer is a whole interval. The middle of the overlap of the two segments
  (projected on segment 1) is used, which keeps symmetric configurations
  symmetric. The gap is correct; the contact is represented by this single point,
  so a line contact between parallel capsules is not represented by a force couple.
  A two point manifold for parallel pairs is future work (box contacts).
  At the parallel configuration the gap is not differentiable and the envelope
  argument of Section 3.1 only holds for the chosen witness point.
* *Intersecting axes* ($\lVert d\rVert < 10^{-12}$): the normal is undefined; the
  normalized cross product $e_a\times e_b$ is used (arbitrary sign), and an
  arbitrary perpendicular to $e_a$ if the axes are collinear.

## 4. Tangent frames

For plane pairs the tangents are the fixed columns $t_1, t_2$ of the plane frame, no
transport is needed. For all other pairs no body-fixed tangent exists, so every slot
keeps a *reference basis* $B = (t_1^{ref}, t_2^{ref}, n^{ref})$ that is transported
with the normal.

**Transport at the end of every time step.** Given the new normal $n$ (computed by
the pair kernel for *all* pair slots, candidate or not), rotate $B$ by the smallest
rotation $R$ that maps the old third column $a = n^{ref}$ onto $n$ (Crisfield 1996,
(16.107)). With $c = a\cdot n$ the rotation is

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
for any resolved simulation. Transporting all pair slots every step costs one pair
kernel evaluation per slot and step, i.e., $O(N^2)$ for fixed slots (see Section 9).

**Tangents at an evaluation point.** Inside a step the tangents are rebuilt from
the *current* normal and the second reference tangent:

$$
t_1 = \frac{t_2^{ref}\times n}{\lVert t_2^{ref}\times n\rVert},
\qquad
t_2 = \frac{n\times t_1}{\lVert n\times t_1\rVert}.
$$

This gives an orthonormal right-handed basis $(t_1, t_2, n)$ at every $q$,
close to the reference basis. For isotropic Coulomb friction the physical result does
not depend on the choice of the tangent basis; it only has to be continuous in
$q$ and change little within a step, which is what the transport ensures. The
construction degenerates for $t_2^{ref}\parallel n$, i.e., if the normal turns
by about $90^\circ$ relative to the previous step's normal within a single step;
as for the transport, this does not happen for resolved motions.

**Initial basis.** At assembly the normals are computed from the initial
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
requested stay zero. Requested slots that are not candidates of the broad phase are
ignored (they are inactive).

## 7. Broad phase

The number of slots grows quadratically with the number of shapes. Almost all
pairs are far apart, so their gap is not needed exactly: it is only needed to
decide that they are inactive. Planes are always evaluated (their slots are
cheap and few); the broad phase concerns pairs of non-plane shapes.

**Bounding spheres.** Each non-plane shape $i$ is enclosed in a sphere around its
frame origin with radius $b_i = \rho_i + h_i$ (half length $h_i=0$ for spheres),
independent of its orientation. For a pair
$$
g_N \;\ge\; \lVert r_i - r_j\rVert - b_i - b_j .
$$

**Candidates.** A pair is a candidate if the right-hand side of this bound (the gap
of the bounding spheres) was at most `margin` at the last rebuild. Candidates are
found with a `cKDTree` on the frame origins: all pairs closer than
$2 b_{max} + \text{margin}$, filtered by the bounding gap. Gaps of candidate pairs
are computed exactly by the pair kernels. All other pairs report the constant gap
`INACTIVE_GAP = 1e6`. For two spheres the bounding gap is the exact gap.

**Rebuild.** The candidate list is rebuilt

* at assembly,
* in `step_callback` after every accepted time step, and
* before evaluating gaps whenever some frame origin has moved more than
  `margin / 4` since the last rebuild (solvers evaluate the midpoint and
  iterates, which can differ from the last accepted state).

**Guarantee.** After a rebuild the bounding gap of a non-candidate pair satisfies
$\lVert r_i - r_j\rVert - b_i - b_j > \text{margin}$ and hence
$g_0 > \text{margin}$. The center distance changes by at most the sum of the
displacements of the two frame origins, and rotations do not matter because the
bounding spheres cover all orientations. While every origin has moved by at most
$\text{margin}/4$ since the rebuild, this sum is at most $\text{margin}/2$, hence

$$
g \;\ge\; \lVert r_i - r_j\rVert - b_i - b_j
\;>\; \text{margin} - \frac{\text{margin}}{2} \;=\; \frac{\text{margin}}{2} .
$$

A non-candidate pair is therefore separated by more than half the margin, and
reporting a large gap does not change the active set of any solver whose
closeness tolerance is below $\text{margin}/2$. (The default margin is half the
largest bounding radius; the solvers' tolerances are many orders of magnitude
smaller.) The bound is quantitative, not only "strictly positive": a smaller
threshold than $\text{margin}/4$ would only increase the number of rebuilds, a larger
one (for example $\text{margin}/2$) would allow the gap of a skipped pair to drop
arbitrarily close to zero.

The margin therefore only affects efficiency.

## 8. Slots and data layout

* One normal slot per allowed contact pair, `nla_N` in total. With friction
  ($\mu > 0$) there are two friction forces per slot, `nla_F = 2 nla_N`, in the
  order $(2k, 2k+1)$ for slot $k$.
* **Slot order**:
  1. plane-sphere pairs (for each sphere, all planes in the order of the shape list),
  2. plane-capsule pairs (for each capsule, all planes; for each plane the two end
     points, $-h$ first),
  3. all pairs of non-plane shapes in lexicographic order of the shapes, with the
     spheres first and then the capsules (each group in the order of the list). The
     normal points from the earlier to the later shape, so sphere-capsule pairs have
     the sphere as shape 1.

  Pairs on the same subsystem and pairs listed in `exclude` get no slot.
* Slots are fixed after assembly, so warm starts and solution arrays have a
  constant size. The index of the pair $(i, j)$, $i<j$, among $n$ non-plane shapes is
  $i(2n-i-1)/2 + (j-i-1)$ before filtering; a lookup table maps it to the slot.
* **Coordinates**: `qDOF`/`uDOF` of the contribution are the concatenation of the
  coordinates of all involved subsystems (each once, in order of first appearance in the
  shape list). Shapes keep slices into these arrays.
* **Sparse output**: `W_N`, `W_F` are returned as `CooMatrix` with local row
  indices (`u`-coordinates of the contribution) and global slot columns, built
  from flat arrays with `CooMatrix.add_entries`. Entries of the same coordinate
  from several contacts are summed on conversion.
* **Per-state cache**: positions, velocities and Jacobians of all shapes, and the
  geometry (gaps, normals, contact point offsets) of all candidate slots are evaluated
  once per state and reused by all quantities at the same $(t, q, u)$ (the key is the
  raw bytes of the arrays). The cached geometry is dropped whenever the broad phase is
  rebuilt.

## 9. Cost

Per state, shape kinematics cost one loop over the shapes, the geometry is vectorized
over all candidate slots, and velocities are vectorized over the requested slots. The
Jacobian-based quantities ($W_N$, $W_F$) loop over the requested slots only. Per time
step, the transport of the reference bases evaluates the normal of *every* pair slot
(vectorized), so the cost of a step is $O(N^2)$ in the number of non-plane shapes even
if almost no pair is a candidate. This is the price of fixed slots; dynamic slots
(created by the broad phase) would remove it.

## 10. Acceleration level quantities

`g_N_ddot` and `gamma_F_dot` are only used for the computation of consistent initial
conditions (the terms $\zeta_N, \zeta_F$ with $\dot u = 0$).

* **Plane-sphere slots** use the formulas of `Sphere2Plane`. They are the exact time
  derivatives of $\gamma$ including the rotation of the plane frame.
* **Sphere-sphere slots** use the formulas of `Sphere2Sphere`:
  $n\cdot(a_2-a_1)$ for the normal and $t_k\cdot(a_{P_2}-a_{P_1})$ with
  $a_{P_i} = a_i + \Psi_i\times c_i$ for the tangential components. These are
  **not** the exact derivatives: they omit the rotation of the normal. The exact
  derivative of $\gamma_N = n\cdot(v_2 - v_1)$ is $n\cdot(a_2-a_1) + \dot n\cdot(v_2-v_1)$
  with $\dot n = (I - nn^T)(v_2-v_1)/\lVert d\rVert$, so the omitted term is
  $\lVert (I-nn^T)(v_2-v_1)\rVert^2/\lVert d\rVert$ (verified numerically). The world
  reproduces the per-pair class including this inaccuracy; it matters only for sphere
  pairs that are in contact at the initial time with a tangential relative velocity.
* **All other slots** (capsule end points, sphere-capsule, capsule-capsule): the
  witness points move relative to the bodies, which makes closed formulas lengthy. The
  derivative is taken numerically as a central difference of the velocity level
  quantities along the motion,

$$
\frac{d}{dt} f \approx \frac{f(t+\delta,\ q+\delta\dot q,\ u+\delta\dot u)
- f(t-\delta,\ q-\delta\dot q,\ u-\delta\dot u)}{2\delta},
\qquad
\delta = 10^{-5},
$$

  with $\dot q$ from the subsystems' `q_dot` and $\dot u$ the given accelerations. This
  gives the exact derivative up to about $10^{-9}$ (the tangent reference basis is held
  fixed, as in the closed formulas). The test
  `test_acceleration_level_finite_difference_matches_exact_formulas` shows that the
  same difference reproduces the exact plane-sphere formulas to $10^{-5}$.

## 11. What is not covered yet

* **Second derivatives** ($g_{N,q}$, $\partial(W_N\Lambda_N)/\partial q$, $\gamma_{F,q}$,
  $\partial(W_F\Lambda_F)/\partial q$) are required by the Newton-based solvers
  (BackwardEuler, Rattle, static solvers). The world raises
  `NotImplementedError` for them; use Moreau or DualStormerVerlet.
* **Boxes and other shapes**, and contact manifolds with several points per pair
  (needed for parallel capsule axes, Section 3.6) are planned in `SPEED_UP_IDEAS.md`
  (Plan D, phase 4).
* **Parametrized subsystems** (rods with $\xi$) are not supported.

## 12. Code map

| Formula / concept | Code (`world.py`) |
|---|---|
| shape frame pose, velocity, Jacobian, acceleration | `_positions`, `_velocities`, `_jacobians`, `_accelerations` |
| plane-sphere and plane-capsule end point geometry ($g_N$, $n$, $c_1$, $c_2$) | `_ps_geometry` |
| sphere-sphere, sphere-capsule and capsule-capsule geometry | `_pair_geometry` |
| closest points of two segments (Ericson, parallel case) | `_segment_parameters` |
| geometry of all candidate slots, cached per state | `_geometry` |
| gaps of all slots (`INACTIVE_GAP` for non-candidates) | `g_N` |
| relative velocity of the contact points | `_relative_velocity` |
| normal velocity $\gamma_N$ | `g_N_dot` |
| friction velocity $\gamma_F$ and tangents | `_gamma_F`, `_tangents` |
| force directions $W_N$, $W_F$ | `W_N`, `_W_F`, `_force_directions` |
| transport of the reference bases | `step_callback`, `_all_pair_normals` |
| initial reference bases | `assembler_callback` |
| broad phase | `_update_candidates` |
| active-set slots | `_normal_slots`, `_friction_slots` |
| acceleration level (exact plane-sphere / sphere-sphere, central difference otherwise) | `g_N_ddot`, `_gamma_F_dot`, `_gamma_dot_exact`, `_fd_time_derivative` |
| friction projection and active friction indices | `System.friction_prox`, `System.friction_indices` (`system.py`) |

## References

* M. A. Crisfield, *Non-linear Finite Element Analysis of Solids and Structures,
  Vol. 2: Advanced Topics*, Wiley, 1996, Section 16.13 (smallest rotation).
* C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2005, Section 5.1.9
  (closest points of two segments).
* V. Acary, B. Brogliato, *Numerical Methods for Nonsmooth Dynamical Systems*,
  Springer, 2008 (set-valued contact laws, proximal formulation).
* C. Studer, *Numerics of Unilateral Contacts and Friction*, Springer, 2009;
  see also the references in `cardillo/math/prox.py`.
