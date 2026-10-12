# Speed-up ideas

Detailed plans for the next three structural optimizations:

* **Plan A**: batching identical joints,
* **Plan B**: batching all elements of a rod,
* **Plan C**: assembly plans for the sparse matrices of `System` and the solvers.
* **Plan D**: generic contacts between primitive pairs (signed distance functions,
  broad phase, one contribution for the whole collision world).

Plans A and B follow the pattern that already worked for `Sphere2Sphere` (see
`_Sphere2SphereBatch` in `cardillo/contacts/sphere2sphere.py` and
`System._setup_batches` in `cardillo/system.py`): **evaluate many identical
contributions with array operations instead of calling them one by one from
Python**. Plan C removes the cost of turning the resulting blocks into scipy sparse
matrices on every call.

## Where we are (measured)

The benchmarks in `benchmarks/` (`python benchmarks/run.py --check baseline`)
compare against the unmodified code.

| Case | Original | Now |
|---|---|---|
| balls (Moreau, 1035 contacts) | 56.3 s | 2.9 s |
| balls_rattle / balls_backward_euler / balls_dsv | 31.4 / 9.8 / 5.1 s | 4.0 / 3.3 / 1.0 s |
| pendulum_rattle / pendulum_backward_euler (10 spherical joints) | 3.45 / 1.55 s | 1.72 / 0.72 s |
| rod_mixed (20 elements, statics) | 1.13 s | 0.60 s |
| rod_constrained / rod_se3 / rod_r12 | 1.83 / 0.95 / 1.22 s | 0.95 / 0.54 / 0.82 s |

What remains is per-call Python overhead, which grows linearly with the number of
joints, elements and contacts:

* **Joints.** For the constrained pendulum, sampling profiles show roughly 35-45 %
  of the time inside the constraint methods `g`, `g_q`, `W_g`, `Wla_g_q`, `g_dot`,
  `g_dot_u` (about 12-30 us per joint and call), most of it overhead in lambdas,
  caches and tiny numpy calls.
* **Rods.** For `rod_mixed`, roughly 60-70 % of the run is inside the per-element
  functions `W_c_el`, `c_el`, `Wla_c_el_qe`, `f_gyr_el` and their `_eval` / `_deval`
  calls. The rest is the sparse solve and assembly.

Amdahl bound: a joint batch that makes the constraint part 10x faster gives about
**1.5x** on joint-dominated models (more for larger chains, since the fixed
overhead per joint disappears). A rod batch that makes the element part 10-20x
faster gives about **2.5-4x** on rod-dominated models, growing with the number of
elements.

## Shared prerequisites

Both plans need the same three building blocks. Build them first.

1. **Flat entries in `CooMatrix`.** Today `coo[rows, cols] = dense_block` is called
   once per joint or element. Batches produce all blocks at once, so add
   `CooMatrix.add_entries(row, col, data)` that appends three flat arrays as one
   chunk (the chunk storage already exists). Duplicate entries stay allowed; they
   are summed on conversion.
2. **A general batch protocol in `System`.** `_setup_batches` currently knows only
   `g_N` and `step_callback`. Generalize it:
   * a contribution class defines `batch_class` with
     `create(contrs) -> (batch | None, rest)`, as now;
   * a batch declares the properties it implements, e.g.
     `batch.properties = {"g", "g_q", "W_g", ...}`, and implements each one with a
     signature that mirrors the System method but takes the global arrays
     (`batch.g(t, q, g_out)`, `batch.W_g(t, q, coo)`);
   * for each property `System` keeps `__<prop>_batches` and `__<prop>_single`, and
     every System method becomes "loop over batches, then loop over singles".
     Write one small helper so the roughly 20 affected methods do not repeat this.
   * Batched contributions must stay in the per-property contribution lists used by
     other code (`get_contribution_list`, owner arrays for active sets).
3. **Batched rotation kernels** in a new `cardillo/math/batch.py`: `Exp_SO3_quat`,
   `Exp_SO3_quat_P`, `T_SO3_quat`, `T_SO3_quat_P`, `T_SO3_inv_quat`, `ax2skew`,
   `cross3` for arrays with a leading batch axis. Each is tested against the scalar
   function in `cardillo/math/rotations.py` on random inputs (rtol 1e-14). The
   scalar functions stay the reference implementation.

Numerical note: batched arithmetic reorders sums, so results agree with the
unbatched code to about 1e-15 relative, not bitwise. Solver-level comparisons
therefore use the same tolerances as the existing benchmark check (1e-8), and
iterative linear solvers (MINRES) need their looser check.

---

## Plan A: batching identical joints

### Goal

`System.g`, `g_q`, `W_g`, `Wla_g_q`, `g_dot`, `g_dot_q`, `g_dot_u`, `g_ddot` (and
`chi_g`, which calls `g_dot`) evaluate all eligible joints with a few array
operations. The sparse `W_g`, `g_q` and `Wla_g_q` are produced from precomputed
index patterns.

### Which joints qualify

`Spherical`, `Revolute`, `RigidConnection`, `Prismatic`, `Cylindrical` all derive
from `PositionOrientationBase` (`cardillo/constraints/_base.py`). A joint is
described by:

* two subsystems with offsets `B1_r_P1J0`, `B2_r_P2J0` and orientation offsets
  `A_K1J0`, `A_K2J0` (constants fixed at construction),
* `nla_g_trans = 3` translation constraints and a list `projection_pairs` of
  rotation constraints `e_a^(1) . e_b^(2) = 0`.

A joint is **eligible** if both subsystems are `RigidBody` (checked by function
identity of `r_OP`/`A_IB`, as for `Sphere2Sphere`) or the fixed `Frame`/origin
(constant `r_OP`, `A_IB`), and `xi1 = xi2 = None`. Everything else (rods,
`PointMass` with orientation constraints, user subclasses overriding methods) falls
back to the per-joint path. Eligible joints are grouped by
`(class, nla_g, tuple(projection_pairs))`, so each group has uniform shapes.

### Per-joint data to gather (once, at assemble time)

For K joints in a group:

* `idx_q1 (K,7)`, `idx_q2 (K,7)`, `idx_u1 (K,6)`, `idx_u2 (K,6)`: global
  indices (a fixed subsystem uses a flag and constant arrays instead).
* `B1_r (K,3)`, `B2_r (K,3)`, `A_K1 (K,3,3)`, `A_K2 (K,3,3)`.
* `la_gDOF (K, nla_g)` and, for the sparse patterns, row/column index arrays
  (below).

### Batched formulas

With `p` the quaternion of each body, `A = Exp_SO3_quat(p)` of shape `(K,3,3)` and
`A_p = Exp_SO3_quat_P(p)` of shape `(K,3,3,4)` computed once per call and shared by
all properties of the group (one small cache keyed on `(t, q.tobytes())`, valid
for one point):

| Quantity | Formula (batched over K) |
|---|---|
| `r_OJ_i` | `q_i[:, :3] + einsum(A_i, B_i_r)` |
| `g[:, :3]` | `r_OJ_2 - r_OJ_1` |
| `g[:, 3+k]` | `A_IJ1[:, :, a_k] . A_IJ2[:, :, b_k]` with `A_IJ_i = A_i @ A_K_i` |
| `g_q` translation | `-r_OJ_1_q1`, `+r_OJ_2_q2`, with `r_OJ_q = [I, einsum(A_p, B_r)]` |
| `g_q` rotation | `A_IJ2[:, :, b] . A_IJ1_q1[:, :, a]` etc. (existing formulas) |
| `W_g` translation | `-J_J1^T`, `+J_J2^T`, `J_J = [I, -A ax2skew(B_r)]` |
| `W_g` rotation | `n_k @ [J_R1, -J_R2]` with `n_k = e_a x e_b`, `J_R = A B_J_R` |
| `Wla_g_q` | the existing einsum expressions, with a leading K axis |
| `g_dot`, `g_ddot`, `g_dot_q` | same structure with `v_P`, `Omega`, `a_P`, `Psi` of the bodies |

The scalar implementation in `PositionOrientationBase` is the specification;
the batched code is a transcription with a leading axis and must reproduce it.

### Sparse output

A joint's `W_g` block couples `u1, u2` (rows) to its `nla_g` constraint columns.
The row and column indices depend only on the DOF layout, so they are built once:

* `rows = repeat/tile(idx_u(K, 12), nla_g)`, `cols = la_gDOF` broadcast,
* per call only `data` is computed (shape `(K, 12, nla_g)`, flattened) and appended
  with `CooMatrix.add_entries`.

The same applies to `g_q` (`(nla_g, nq1+nq2)` blocks) and `Wla_g_q`. This is the
first place where the *assembly plan* idea (see below) pays off, since the pattern is
fixed and fully known in advance.

### Integration with System

* `_setup_batches` creates one batch per eligible group in `assemble()`.
* Each System method skips batched joints in its single loop (`__g_single`, ...).
* `la_gDOF` handling, `nla_g` counts and `W_g`'s column order stay untouched, so
  solvers, `Wla_g_q`-consumers and `compute consistent initial conditions` do not
  change.
* `System.set_new_initial_state` calls `assemble` again; batches are rebuilt there.
* `deepcopy(System)` copies batches together with the contributions (shared
  references are preserved by `deepcopy`).

### Phases

0. **Benchmarks.** Add `pendulum_chain_50` (Spherical), a Revolute chain, a mixed
   chain with `RigidConnection` and a closed-loop mechanism (four-bar). Record
   baselines from the current code. Also keep `pendulum_rattle` and
   `pendulum_backward_euler`.
1. **Prerequisites** (flat `CooMatrix` entries, protocol, batched kernels).
2. **Spherical**: `g`, `g_q`, `W_g`, `Wla_g_q` (translation only). This already
   covers the BackwardEuler and Rattle hot spots and is the quickest check of the
   whole design.
3. **Velocity-level and second-order properties**: `g_dot`, `g_dot_u`, `g_dot_q`,
   `g_ddot`, `chi_g` (Rattle, Moreau, consistent initial conditions).
4. **Orientation constraints**: `Revolute`, `RigidConnection` (projection pairs), then
   `Prismatic` and `Cylindrical`.
5. **Optional**: joints with a fixed `Frame`, joints attached to `PointMass`, and the
   analogous batch for `gamma` constraints and `Wla_gamma_q`.

### Tests

* Per property, per joint type: batched vs per-joint result on random states
  (random quaternions, random `u`, random multipliers), rtol 1e-12.
* Derivative consistency: batched `g_q` vs finite differences of batched `g`; batched
  `Wla_g_q` vs finite differences of `W_g @ la`; this catches transcription errors
  independent of the scalar code (`cardillo/math/approx_fprime.py` exists).
* Solver equivalence as in `test_performance_paths.py`: run the existing
  `test_spherical_revolute` models with batching on and off; compare `q`, `u`, `la_g`.
* Edge cases: K = 1 (a group of one is not batched, falls back), joints to the origin,
  two joints on the same body pair, mixed joint types, `nla_g = 0` contributions,
  complex-valued `q` if any code path uses complex step.

### Risks and open questions

* Eligibility detection must be strict; a subclass that overrides `r_OP` has to
  fall back. Use function identity checks, never `isinstance` alone.
* The shared per-call cache (`A`, `A_p`) must be keyed on the full `(t, q)`
  and never survive a changed state.
* Duplicated bodies: many joints share a body, so gathers read the same entries
  several times (fine), and scatter-adds into `q`-sized outputs must use summing
  semantics (the COO format sums automatically).
* Gain is bounded by the Amdahl estimate above; measure after phase 2 and stop if the
  remaining time is dominated by `bmat`/LU instead.

---

## Plan B: batching all elements of a rod

### Goal

For a Cosserat rod with `nelement` elements, evaluate `h`, `h_q`, `c`, `c_q`, `W_c`,
`Wla_c_q` (mixed), `f_int`/`h_q` (displacement-based), the constrained family
`g`, `g_q`, `W_g`, `Wla_g_q`, and the gyroscopic terms with array operations over
the element axis `e` and the quadrature-point axis `i`, instead of Python loops over
elements, quadrature points and nodes. The per-element methods remain as the
reference implementation and are still used for point evaluations (contacts, stress
output, `r_OP(xi)`).

### Data available per rod (all precomputed already)

* `qp[e, i]`, `qw[e, i]`: quadrature points and weights.
* Shape functions `N_r[e, i, n]`, `N_r_xi`, `N_p`, `N_p_xi`, `N_la_c[e, i, n]`
  (constant, built in `mesh1D.py`).
* Connectivity `elDOF[e, :]`, `elDOF_u[e, :]`, `elDOF_la_c[e, :]`, and the local
  nodal DOF lists (identical for every element).
* Reference strains `J`, `B_Gamma0`, `B_Kappa0`.

### Step 1: batched kinematics

For the quaternion rod, `r_OP`, `r_OP_xi`, `p`, `p_xi` are **linear** in the element
coordinates. The interpolation matrices depend only on shape functions, so they are
built once for the whole rod (this generalizes the per-point cache
`_interpolation_matrices` that exists now):

```
Mat[e, i] : (12 or 14, nq_el),   z[e, i] = Mat[e, i] @ q[elDOF[e]]
```

Gather `qe = q[elDOF]` with shape `(E, nq_el)` and compute all `(r, r_xi, p, p_xi)`
for all `(e, i)` with one `einsum`. Then, with the batched kernels from the shared
prerequisites:

* `A_IB = Exp_SO3_quat(p)` for all `E*I` points `(E,I,3,3)`
* `B_Gamma_bar = A_IB^T r_xi`, `B_Kappa_bar = T_SO3_quat(p) p_xi`
* for Jacobians: `A_IB_qe = Exp_SO3_quat_P(p) @ p_qe`, `B_Gamma_bar_qe`,
  `B_Kappa_bar_qe` (the exact expressions of `_deval`, with leading axes).

Cache one `RodState` object per `(q.tobytes())` holding these arrays; `h`, `c`,
`W_c`, `c_q`, `Wla_c_q` called at the same `q` reuse it. This fixes the current
cache thrashing (the per-point cache holds `nquadrature + 10` entries and is
overwritten before the next pass).

### Step 2: material law

`Simo1986` is a linear diagonal law, so `B_n = C_n (B_Gamma_bar - B_Gamma0)` and
`B_m` are single `einsum`s. Laws that are not diagonal-linear (e.g. `Harsch2021`,
Sadowsky) get the same interface `material.B_n_B_Kappa_batched(Gamma, Kappa)`; if a
law has no batched version the rod falls back to the per-point call for the law only
while everything else stays batched.

### Step 3: element integrals as contractions

Each current element function is `sum over i` of products of shape-function entries
and per-point tensors. Express them as contractions, for example for the mixed rod's
`W_c` (virtual work of `la_c`):

```
W_c[e, node_r*3 + a, node_la*nmixed + k] -=  N_r_xi[e,i,node_r] * A_IB[e,i,a,mixed_n[k]] * N_la_c[e,i,node_la] * qw[e,i]
W_c[e, node_p*3 + a, ...]               -=  (N_p_xi * I - N_p * skew(B_Kappa_bar))[...] * N_la_c * qw
```

which is `einsum("ein,eiak,eil,ei->enalk", ...)` plus the corresponding `p` and `A_IB`
terms. The exact index sets (`mixed_n`, `mixed_m`, `constraints_gamma`,
`constraints_kappa`) are already attributes of the rod and are applied by slicing.
The result is one dense block per element `(E, nu_el, nla_el)`. The same recipe covers
`c_q`, `Wla_c_q`, `h_q`/`f_int_el_qe`, `f_gyr_el(_ue)`, and the constrained family.

### Step 4: assembly

All elements of a rod have the same block shape and a fixed pattern, so the sparse
output is an index-pattern problem:

* `rows = elDOF_u[:, :, None]` broadcast against `elDOF_la_c[:, None, :]`, flattened,
  computed once;
* per call only `data = blocks.ravel()` changes; append with
  `CooMatrix.add_entries(rows, cols, data)`. Vectors (`h`, `c`) are scatter-added
  with `np.add.at` / `np.bincount` on the same precomputed index arrays.

This replaces `nelement` calls of `coo[elDOF_u, elDOF_la_c] = ...` (about 40 us each
today) with one append, and is the second natural user of the assembly-plan
machinery below (the data array maps directly onto fixed CSC slots).

### Code structure

* New module `cardillo/rods/_batched.py` with the batched kinematics and the
  contraction kernels as free functions of arrays (easy to test in isolation).
* The rod classes get thin overrides: `h`, `h_q`, `c`, `c_q`, `W_c`, `Wla_c_q`,
  `g`, `g_q`, `W_g`, `Wla_g_q` call the batched kernels when
  `self.batched` is true (default true once validated; constructor flag
  `batched=False` keeps the reference path).
* The per-element methods and `_eval` / `_deval` stay unchanged and are the
  reference in tests. `r_OP`, `J_P`, `A_IB` at arbitrary `xi` (contacts, forces,
  constraints attached to rods) keep using them.

### Phases

0. **Benchmarks.** Add larger rod cases (100 and 400 elements, statics and
   `rod2plane`-style dynamics, mixed and constrained, Quaternion and R12) with
   baselines from the current code; keep the 20-element cases.
1. **Prerequisites** (batched kernels, flat `CooMatrix` entries).
2. **Batched kinematics + `RodState` cache**, tested element-wise against
   `_eval`/`_deval`.
3. **Mixed quaternion rod**: `c`, `W_c`, `c_q`, `Wla_c_q`. This is the statics
   hot spot (Newton) and the main benchmark.
4. **Dynamics**: `h` with `f_gyr`, `h_q`, `h_u`, and the constant mass matrix path
   (already assembled once).
5. **Displacement-based rod**: `f_int_el`, `f_int_el_qe`.
6. **Constrained rod family**: `g`, `g_q`, `W_g`, `Wla_g_q`.
7. **R12** (shares most of the code), then **SE3** (needs batched `Exp_SE3_h`,
   `Log_SE3_H`, `T_SO3_psi`; the axis-angle kernels are already vectorized per
   call, the batch axis must be added).

### Tests

* Kinematics: for random configurations, batched `(r, A_IB, Gamma, Kappa)` and all
  Jacobians vs per-point `_eval`/`_deval`, rtol 1e-12.
* Every batched method vs the per-element method on random `q`, `u`, `la`
  (all rod variants), rtol 1e-12, including dense vs sparse equality.
* Jacobian checks by finite differences of the batched residuals.
* Benchmark equivalence: `rod_*` cases against the stored baselines.
* The existing `test_rotations` derivative tests cover the scalar kernels that
  the batched ones are checked against.

### Risks and open questions

* Memory: `A_IB_qe` has shape `(E, I, 3, 3, nq_el)`. For 1000 elements with p = 2
  this is a few hundred MB if all points are held at once; chunk over elements if
  needed (the code is element-parallel, so chunking is trivial).
* Quadrature rules differ between statics (`nquadrature`) and dynamics
  (`nquadrature_dyn`); mass-related terms use their own arrays.
* Materials without a batched implementation fall back per point (slower, still
  correct); decide whether to implement batched versions for all shipped laws.
* The per-element code contains subtle special cases (reduced integration, mixed
  index sets, `idx_impressed`); the contraction formulation must be derived from the
  code, not from the paper, and checked element-wise.
* Expected gain: bounded by the sparse solve and by `bmat`/assembly at the system
  level. If after phase 3 the element part is below about 20 % of the run, stop.

---

## Plan C: assembly plans for the sparse matrices

### Goal

Every global matrix of `System` (`M`, `h_q`, `h_u`, `q_dot_q`, `q_dot_u`, `g_q`,
`W_g`, `Wla_g_q`, `c_q`, `W_c`, `Wla_c_q`, `gamma_q`, ...) and every block matrix in
the solvers (`bmat` in `BackwardEuler._J_x`, `Rattle._J_x1`,
`DualStormerVerlet.A`, Moreau's `A`) is built from a fixed pattern of entries. Build
the pattern once, then only move numbers on later calls.

### What happens today

Per call of a matrix method such as `System.W_g`:

1. A new `CooMatrix` is created.
2. For each contribution, its dense block is appended (`CooMatrix.__setitem__`:
   `flatten`, `repeat`, `tile`, a few small numpy calls, about 8-15 us per block).
3. `asformat` concatenates the chunks and calls scipy's `coo_array`/`csr_array`
   constructor: index dtype checks, min/max of the index arrays, sorting and summing
   of duplicates (about 25-60 us fixed cost plus O(nnz log nnz)).
4. The solver combines several of these matrices with `bmat` or sparse arithmetic
   (`A - B - C`), which repeats the construction work and allocates intermediate
   sparse arrays (`csr_array.__sub__` is visible in the profiles).

Measured: in the constrained pendulum cases, `CooMatrix.__setitem__`,
`tosparse`, `bmat` and the sparse arithmetic together are about 15-25 % of the run
(cProfile: `__setitem__` 0.46 s + `tosparse` 0.59 s of 5.2 s; sampling profile:
`bmat` plus sparse subtraction about 15 % of `_J_x`). In the rod cases the share is
smaller but grows with the element count.

The pattern, i.e. which rows and columns receive entries and in which order, is
identical on every call as long as the active structure of the model does not change.

### Mechanism

An `_AssemblyPlan` describes one matrix (or one block matrix):

**Recording (first call).** While the existing code builds the matrix, remember for
every appended chunk its shape and its `row`/`col` index arrays, in call order.
From them compute once:

* `row_all`, `col_all`: the concatenated raw indices (length `n_raw`, duplicates
  included),
* the CSC structure `indptr`, `indices` of the final matrix: sort the keys
  `col * nrows + row` once with `np.unique(..., return_inverse=True)`; the unique keys
  give the structure, and
* `slot[k]` (= the `return_inverse` array): for raw entry `k` the position in the final
  `data` array. Duplicates share a slot.

Optionally compute the same for CSR (`indptr_r`, `indices_r`, `slot_r`) if solvers ask
for `csr`; or derive CSR from the CSC data with a second precomputed permutation.

**Replay (later calls).**

1. The contributions are evaluated as usual and their blocks are collected as flat
   data chunks in the same order. With batches (Plans A and B) this is already a single
   array.
2. `data_raw = np.concatenate(chunks)`; check `len(data_raw) == n_raw`.
3. `data = np.bincount(slot, weights=data_raw, minlength=nnz)` sums duplicates in
   O(n_raw).
4. Build the matrix directly without validation:
   `csc_array((data, indices, indptr), shape=shape)` with arrays shared between calls
   (treated as read-only; scipy does not modify them unless the user mutates the
   matrix, which the code base does not do).

**Guard.** A cheap check on every replay: number of chunks and each chunk length equal
the recorded ones. Any mismatch drops the plan and falls back to the current code path
for that call (and re-records on the next full call).

**Complex values.** `np.bincount` supports only real weights; for complex `q` (complex
step) use two bincounts (real and imaginary part) or fall back. Complex-step
derivatives of assembled matrices are not used in the library code today, so a
fallback is sufficient.

### What is not planned

A plan is only valid for the **full** assembly of contributions that return **dense
blocks** of fixed shape. These cases stay on the current path:

* contributions returning sparse arrays or `CooMatrix` objects (their `nnz` can vary),
* the subset calls introduced for active sets (`I_N=`, `I_F=`): different calls
  involve different contributions. (They could be planned per contribution subset, but
  the active set changes too often to pay off.)
* matrices whose blocks change shape between calls (none known so far; the guard
  protects against it anyway).

### Where it plugs in

* **`System`.** About 30 methods contain the same loop:

  ```python
  coo = CooMatrix(shape)
  for contr in self.__X_contr:
      coo[rows(contr), cols(contr)] = contr.X(t, q[contr.qDOF], ...)
  return coo.asformat(format)
  ```

  Replace them by one helper
  `_assemble_matrix(name, shape, contrs, rows_of, cols_of, evaluate, format, subset=None)`
  that owns a plan per `(name, shape)`. The first full call records, later calls replay.
  Calls with `subset is not None` always use the plain path.
* **Batches (Plans A, B).** A batch returns `(row, col, data)` flat arrays directly; the
  plan records them like any other chunk, and the index arrays are cached by the batch
  so even recording is cheap.
* **Solver block plans.** The solvers combine several matrices into one block matrix
  per step or per Newton iteration. A *block plan* extends the same idea one level up:
  the offsets of each block inside the final matrix are known, so the sub-plans' raw
  indices are shifted once and merged into one structure. Replay then produces the
  final CSC directly from the blocks' raw data; no `bmat`, no intermediate sparse
  matrices. This also replaces expressions such as `M - dt * (h_u + Wla_tau_u)` for
  the blocks that are plain sums: the data of the summands is combined with the
  appropriate factors before the one `bincount`.
* **Format requests.** `format="coo"` returns a `coo_array` built from `row_all`,
  `col_all` and `data_raw` (no `bincount`, duplicates allowed as today);
  `csc` as above; `csr` via the second slot map.

### Effects on numerical results

* Duplicate sums are accumulated in a different order than scipy's, so values agree to
  rounding level (about 1e-16), not bitwise.
* Explicit zeros stay in the structure (scipy's COO to CSC conversion also keeps them,
  but the `lil_array` code that was replaced earlier dropped them). They are
  harmless but can change the LU column permutation and thus results at rounding
  level.

### Phases

0. **Benchmarks.** The existing constrained and rod cases are enough; add the 100 and
   400 element rod cases from plan B and a 50-body chain from plan A, since
   assembly plans pay off most with many blocks.
1. **Standalone class** `_AssemblyPlan` with `CooMatrix.add_entries`, no `System`
   changes yet. Unit tests: random patterns with duplicates, empty matrices, all three
   formats, shape/length guard fallback, `float` and `complex` data.
2. **First matrices behind a flag** (`System(assembly_plans=True)`): `W_g`, `g_q`,
   `h_q`, `h_u`, `q_dot_q`, `q_dot_u`, `M`. Measure on the benchmarks.
3. **Decision gate.** If the gain on `pendulum_*` and `rod_*` is below about 10 %, stop
   here and keep only the cheap parts (flat `add_entries`, cheaper `CooMatrix`).
4. **All `System` matrix methods**, then **solver block plans** (`BackwardEuler._J_x`,
   `Rattle._J_x1`, `DualStormerVerlet`'s `A`, Moreau's `A`).
5. **Make plans the default** and remove the flag once all tests and benchmarks match.

### Tests

* The plan is an optimization of a function with a trivial reference (`CooMatrix`
  conversion), so the main test is equality: replayed matrix == freshly built matrix
  (`toarray`, rtol 1e-14) for every `System` matrix method on random states, in all
  three formats, for the benchmark systems.
* Pattern changes: a contribution whose block shape changes between calls (test
  double) must trigger the fallback and a re-record, never a wrong matrix.
* Solver equivalence: the benchmark checks against the stored baselines and
  `test_performance_paths.py`-style on/off comparisons for each solver.
* `deepcopy(System)` and `set_new_initial_state` (re-assembly): plans live in the
  `System` instance and are reset in `assemble`.

### Risks and open questions

* **Staleness.** The biggest risk is replaying a plan when the structure silently
  changed. The guard (number and lengths of chunks) catches shape changes but not a
  contribution that keeps the sizes and changes the *indices* (for example after
  changing DOF numbering). Plans are therefore reset in `assemble()` and contributions
  must not change their DOF arrays afterwards; add an assertion in debug mode that
  compares the freshly computed indices with the recorded ones every N calls.
* **Memory** is `O(n_raw)` per matrix and plan; negligible compared with the
  matrices themselves.
* **Shared read-only arrays.** Returned matrices share `indices`/`indptr` with the
  plan. Code that modifies these in place would corrupt later results. Nothing in the
  repository does, but mark them read-only (`flags.writeable = False`) so mistakes
  fail loudly.
* **Gain is moderate.** Expected 1.1-1.25x on the current benchmarks, more for large
  models and many-block matrices; this is why the decision gate sits after the first
  seven matrices.

---

## Plan D: generic contacts with signed distance functions

### Status

* **Phase 1 done**: `CooMatrix.add_entries`, local active sets
  (`supports_active_set`), `System.friction_indices` / `friction_prox`
  (vectorized Coulomb projection used by BackwardEuler, Rattle and
  DualStormerVerlet), `System.active_friction_laws` (vectorized `compute_I_F`).
* **Phase 2 done**: `cardillo/contacts/collision/` with `SphereShape`,
  `PlaneShape` and `CollisionWorld` (sphere-sphere and plane-sphere, fixed
  slots, `cKDTree` broad phase with rebuild on displacement, Moreau and
  DualStormerVerlet). It reproduces the `Sphere2Sphere`/`Sphere2Plane` results
  bitwise on the `balls` benchmarks; with 200 spheres (20100 pairs) it is 2.8x
  faster than the per-pair classes (5.0 s vs 14.0 s).
* Known limits of phase 2: shape kinematics are evaluated by a python loop over
  the shapes, velocity- and force-direction quantities by a loop over the active
  slots, and the tangent bases of all sphere-sphere slots are transported every
  step (O(N^2) memory and work, as with fixed slots in general).

### Goal

Replace the one-contribution-per-pair contact classes by a collision world that

* supports generic primitive pairs (sphere, plane/half-space, capsule, box,
  cylinder, ellipsoid; later meshes and rods) through signed distance functions
  (SDF),
* finds candidate pairs with a broad phase instead of evaluating all O(N^2) pairs,
* evaluates gaps, normals, tangents and force directions for all candidate pairs with
  batched kernels per pair type,
* plugs into all existing solvers through the usual contact interface (`g_N`, `W_N`,
  `g_N_dot`, `gamma_F`, `W_F`, `xi_N`, `xi_F`, `Wla_N_q`, `Wla_F_q`, `step_callback`,
  `friction_laws`, `e_N`, `e_F`).

### Where we are

* `Sphere2Sphere` and `Sphere2Plane` (`cardillo/contacts/`) are one `System`
  contribution per pair. The balls benchmark therefore has 1035 contributions for 45
  spheres, and `System` loops over all of them.
* Only `Sphere2Sphere` is batched (`_Sphere2SphereBatch`: `g_N` and `step_callback`).
* Active-set evaluation (`I_N`, `I_F`, `gamma_F_contributions`) skips inactive
  contacts, but `g_N` and `step_callback` still touch every pair every step.
* Each pair carries state: the reference contact basis for friction, updated in
  `step_callback` by smallest-rotation transport. The recently fixed cache bug (tangent
  caches ignored this state) shows that this state has to be part of any cache key.

### Architecture

```
CollisionWorld (one System contribution)
 |- shapes:   Shape(sdf, grad, support, aabb) + attachment (subsystem, xi, B_r, A_BS)
 |- filters:  collision groups / masks, excluded pairs
 |- materials: (mu, e_N, e_F) per shape pair or material pair
 |- broad phase:  AABBs -> candidate pairs (once per step)
 |- narrow phase: dispatch[(type_A, type_B)] -> batched kernel
 |               -> gap, witness points p_A, p_B (body frames), normal, feature ids
 |- slots:    fixed contact slots per pair (manifold points), persistent state
 `- kinematics: W_N, g_N_dot, gamma_F, W_F from subsystem point kinematics
```

### Shapes

Each primitive is a class in its local frame with

* `sdf(x)`, `grad(x)` (outward normal), optionally `hess(x)` for curvature terms,
* `support(d)` (furthest point in direction `d`, for GJK),
* `aabb(A_IS, r_OS)` (bounding box for the broad phase),
* vectorized versions of all of these over a leading batch axis.

A shape is attached to a subsystem with a local pose `(B_r, A_BS)`. The world pose
comes from the subsystem's `r_OP(t, q, xi, B_r)` and `A_IB(t, q, xi)`, exactly as the
current contact classes do, so rigid bodies, frames, point masses and rods at a
parameter `xi` all work.

**Sphere-swept primitives.** Many shapes are a core plus a radius: sphere = point +
r, capsule = segment + r, rounded box = box + r, disk = circle + r. The distance between
two sphere-swept shapes is the core distance minus both radii. Core distances
(point-point, point-segment, segment-segment, point-box) are cheap, closed form and
mostly smooth, which keeps gaps and normals smooth for the common cases.

### Narrow phase

A dispatch table maps an unordered pair of shape types to a batched kernel. For a batch
of K pairs it returns:

* the gap `g` (K, m) for up to m manifold points per pair,
* witness points `p_A`, `p_B` expressed in the respective body frames,
* the normal `n` (pointing from A to B),
* feature ids (vertex, edge or face index) per manifold point, used to keep slot
  identity stable across steps,
* a smoothness flag (false at box edges and corners, where the normal jumps).

Kernels, in order of implementation:

1. **Closed form**: sphere-sphere, sphere-plane, sphere-capsule, capsule-capsule,
   capsule-plane (2 points), sphere-box, box-plane (up to 4 points).
2. **SDF kernel** for pairs without closed form: minimize `sdf_B` over the surface of A
   with a few projected Newton steps (or Frank-Wolfe on the support function),
   warm-started from the previous step's witness point. For sphere-swept A this
   reduces to minimizing over the core.
3. **GJK/EPA** fallback for general convex pairs (separation: GJK distance and
   witness points; penetration: EPA depth and normal).
4. **Contact manifolds** for flat contacts (box-plane, box-box, cylinder-plane):
   clip the incident face against the reference face, reduce to at most m points,
   assign them to slots by feature id.

### Kinematics and derivatives

Witness points are treated as body-fixed at the current configuration. By the envelope
theorem, the first derivatives of the gap only need point kinematics of the subsystems:

```
g_N_dot = n . (v_B(p_B) - v_A(p_A))
W_N     = J_B(p_B)^T n - J_A(p_A)^T n
gamma_F = [t1, t2] . (v_B(p_B) - v_A(p_A))     (plus rotation terms for rolling)
W_F     = J_B(p_B)^T [t1, t2] - J_A(p_A)^T [t1, t2]
```

with the existing `J_P`, `v_P` (and `a_P` for `g_N_ddot`) of the subsystems. This is
valid for any pair type and is what Moreau and DualStormerVerlet need.

Second derivatives (`g_N_q`, `Wla_N_q`, `Wla_F_q`, `gamma_F_q`) are only needed by
BackwardEuler and Rattle Jacobians:

* analytic for smooth sphere-swept pairs (curvature of the cores, `hess` of the SDF),
* frozen-normal / frozen-witness approximation otherwise (the Jacobian is only used for
  Newton steps and prox parameter estimation, so an approximation slows convergence
  but does not change the solution).

### Contact frames and friction

* Tangents per slot by the smallest-rotation transport already used in
  `Sphere2Sphere` (Crisfield 1997, (16.107)), stored per slot and updated in
  `step_callback`. Slots that become free are reset; slots that are reused for a new
  feature get a fresh basis from the current normal.
* Friction laws: one `Sphere(mu)` law per slot, coupled to that slot's normal
  multiplier, so the existing prox machinery applies.
* Restitution `e_N`, `e_F` per slot from the material pair.

### Broad phase

Once per step, in `step_callback` (the new `q` of the step is known there) and in
`assembler_callback` for the initial state:

1. compute AABBs of all shapes, inflated by a margin of about `max|v| * dt` plus a
   contact skin, so no contact can appear during the next step without being a
   candidate,
2. find overlapping AABB pairs with sweep-and-prune along the axis of largest
   variance, or with `scipy.spatial.cKDTree` on bounding-sphere centers,
3. apply collision filters, then run the narrow phase only for candidates.

Pairs and slots that are not candidates report a large positive gap, zero `g_N_dot`,
and empty columns in `W_N`/`W_F`, so all solvers treat them as inactive.

### The `CollisionWorld` contribution

One contribution with `nla_N = sum of slots`, `nla_F = 2 nla_N` (for slots with
friction), `e_N`, `e_F`, `friction_laws` (one entry per slot), and `qDOF`/`uDOF`
concatenating the DOFs of all involved subsystems.

Important details that differ from per-pair contributions:

* **Sparse returns.** `System` writes `coo[contr.uDOF, contr.la_NDOF] = contr.W_N(...)`.
  For a world contribution this block is `nu_world x nla_N`, so the methods must return
  a `CooMatrix` with local indices (already supported by `CooMatrix.__setitem__`), never a
  dense array. With `CooMatrix.add_entries` (shared prerequisites) this is cheap.
* **Active sets inside one contribution.** `System._contributions_with` selects whole
  contributions, so a world contribution would always be evaluated completely. Extend the
  protocol: contributions that declare `supports_active_set = True` receive the
  **local** active indices (`I_N`, `I_F` restricted to their `la_NDOF`/`la_FDOF`) as an
  argument and evaluate only those slots. This is a small change in `System` (in
  `_contributions_with` and the methods that call it).
* **Friction loops in the solvers.** BackwardEuler, Rattle and DualStormerVerlet loop
  over `contr.friction_laws` for every contribution returned by
  `gamma_F_contributions`. With one world contribution this is a Python loop over all
  slots. Port Moreau's vectorized `Sphere` prox (`Moreau._setup_friction_prox`) to the
  other three solvers, restricted to slots with positive normal percussion. This is
  worth doing anyway.
* **Per-call caches.** Narrow-phase results (gaps, witness points, normals) are cached
  per `(t, q.tobytes())` and include the per-slot state version in the key, so
  `g_N`, `W_N`, `g_N_dot`, `gamma_F`, `W_F` at the same point share one evaluation.

### Fixed versus dynamic slots

* **Fixed slots** (recommended first): every pair that passes the collision filters
  gets m slots, allocated in `assemble()`. `nla_N` stays constant, warm starts map one to
  one, and all solvers and `Solution` arrays work unchanged. Cost: O(N^2) multipliers and
  O(N^2) memory in the solution history; acceptable up to a few thousand pairs because
  only candidate pairs are evaluated.
* **Dynamic slots**: slots are created by the broad phase each step. Scales to large N
  but needs a varying `nla_N` in all solvers, per-pair warm-start mapping, and
  variable-size solution storage.

Keep the slot mapping behind one interface in `CollisionWorld`, so switching to dynamic
slots later only touches that class and the solvers' state handling.

### Phases

0. **Benchmarks.** Reproduce `balls`, `balls_*` and `rod2plane` with the world
   contribution; add a box stack on a plane, capsules in a box, and a 500-sphere case
   that the per-pair classes cannot handle in reasonable time.
1. **Prerequisites.** `CooMatrix.add_entries`, local active sets for contributions
   (`supports_active_set`), vectorized friction prox in BackwardEuler, Rattle and
   DualStormerVerlet.
2. **Spheres and planes.** Shapes sphere and plane, broad phase, first-derivative
   kinematics, fixed slots, Moreau only. Acceptance: the `balls` results match the
   current `Sphere2Sphere`/`Sphere2Plane` results to rounding level (same gap
   formula, same tangent transport).
3. **Capsules** and the other solvers (DualStormerVerlet first, it needs only first
   derivatives).
4. **Boxes** with contact manifolds and feature ids; tests for box-on-plane resting
   contact, box stacks, sliding with friction, energy behavior.
5. **Jacobians** for BackwardEuler and Rattle: analytic for sphere-swept pairs, frozen
   approximation elsewhere, finite-difference checks.
6. **SDF and GJK/EPA kernels** for the remaining convex pairs (cylinder, ellipsoid).
7. **Meshes and rods**: SDF grids or convex decomposition for meshes; rods as chains of
   capsules along `xi`, with witness points moving along the rod (the subsystem's
   `local_qDOF_P(xi)` changes with the element).
8. **Optional**: dynamic slots.

### Tests

* Per kernel: gap, witness points and normal against brute-force references (dense
  sampling of both surfaces, or the SDF evaluated on a fine grid), random poses, both
  separated and penetrating configurations.
* Derivatives: `W_N` against finite differences of `g_N` for smooth pairs (the
  envelope theorem makes them exact there); `Wla_N_q` against finite differences of
  `W_N @ la`.
* Broad phase: candidate pairs are a superset of all pairs with gap below the margin, on
  random scenes, compared with an all-pairs check.
* Equivalence with the existing classes: world contribution vs `Sphere2Sphere` /
  `Sphere2Plane` on the balls scenes for every solver, rtol about 1e-9.
* Physics regressions: resting box on a plane (no drift, correct normal forces
  distributed over 4 points), sliding box with Coulomb friction (analytic
  deceleration), stacking stability, restitution.

### Risks and open questions

* **Non-smooth normals** at box edges and corners: Jacobians are discontinuous there,
  Newton-based solvers (BackwardEuler, Rattle) may converge slowly; sphere-swept
  (rounded) boxes avoid this and could be the default.
* **Deep penetration**: the closest-point problem becomes ambiguous; EPA and the SDF
  minimization need a defined fallback (e.g. keep the previous normal).
* **Slot identity** for manifolds: if feature ids are unstable, warm starts and tangent
  bases jump, which costs fixed-point iterations and can cause jitter.
* **Margin choice**: too small misses contacts that appear within a step, too large
  evaluates many pairs; derive it from the velocity bound and log missed activations
  in debug mode.
* **Scope**: phases 2-4 already deliver the main value (generic primitives, O(N)
  broad phase, batched evaluation); meshes, rods and dynamic slots are separate projects.

---

## Suggested order and dependencies

| Step | Needs | Expected gain | Notes |
|---|---|---|---|
| Shared prerequisites | none | none directly | `CooMatrix.add_entries`, batch protocol, batched kernels |
| Plan B (rods) | prerequisites | 2.5-4x on rod models, growing with element count | largest remaining win |
| Plan A (joints) | prerequisites | up to about 1.5x on joint-heavy models | bounded by the constraint share of the run |
| Plan C (assembly) | `add_entries`; best after A and B | 1.1-1.25x on top, more for big models | do it on the matrices that are still slow |
| Plan D (generic contacts) | `add_entries`, local active sets, vectorized friction prox | new capability; O(N) broad phase instead of O(N^2) pair evaluation | independent of A-C; phases 2-4 deliver most of the value |

Plans A and B make Plan C's preconditions (fixed shapes, known indices, flat data)
true for large parts of the model, which is why Plan C comes last. If you mainly run
rods, start with B; if you mainly run long joint chains or robots, start with A; if
you mostly run mid-size mixed models, the shared prerequisites plus the first
matrices of Plan C are the cheapest first step.
