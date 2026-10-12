# Speed-up ideas

Detailed plans for the next three structural optimizations:

* **Plan A**: batching identical joints,
* **Plan B**: batching all elements of a rod,
* **Plan C**: assembly plans for the sparse matrices of `System` and the solvers.

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

## Suggested order and dependencies

| Step | Needs | Expected gain | Notes |
|---|---|---|---|
| Shared prerequisites | none | none directly | `CooMatrix.add_entries`, batch protocol, batched kernels |
| Plan B (rods) | prerequisites | 2.5-4x on rod models, growing with element count | largest remaining win |
| Plan A (joints) | prerequisites | up to about 1.5x on joint-heavy models | bounded by the constraint share of the run |
| Plan C (assembly) | `add_entries`; best after A and B | 1.1-1.25x on top, more for big models | do it on the matrices that are still slow |

Plans A and B make Plan C's preconditions (fixed shapes, known indices, flat data)
true for large parts of the model, which is why Plan C comes last. If you mainly run
rods, start with B; if you mainly run long joint chains or robots, start with A; if
you mostly run mid-size mixed models, the shared prerequisites plus the first
matrices of Plan C are the cheapest first step.
