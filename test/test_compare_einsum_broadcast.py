import numpy as np
from time import perf_counter
from cardillo.math import ax2skew_squared, ax2skew_a, ax2skew

num = 10000
#################
# Exp_SO3_quat_p
#################
P = np.random.random(4)
p0, p = P[0, None], P[1:]
p_tilde = ax2skew(p)
p_tilde_p = ax2skew_a()
P2 = P @ P
# np.array_split
t0 = perf_counter()
for _ in range(num):
    r1, r2 = np.array_split(P, [1])
t1 = perf_counter()
for _ in range(num):
    r3, r4 = P[0, None], P[1:]
t2 = perf_counter()
assert np.allclose(r1, r3)
assert np.allclose(r2, r4)
print(
    f"Slicing syntax faster than np.array_split: {t2 - t1 < t1 - t0}, ratio: {(t1 - t0)/(t2 - t1):.3f}"
)
# "ij,k->ijk"
t0 = perf_counter()
for _ in range(num):
    r1 = np.einsum("ij,k->ijk", p0 * p_tilde + ax2skew_squared(p), -(4 / (P2 * P2)) * P)
t1 = perf_counter()
for _ in range(num):
    r2 = (p0 * p_tilde + ax2skew_squared(p))[..., None] * (-(4 / (P2 * P2)) * P)
t2 = perf_counter()
assert np.allclose(r1, r2)
print(
    f"Broadcasting faster than einsum: {t2 - t1 < t1 - t0}, ratio: {(t1 - t0)/(t2 - t1):.3f}"
)
# "ijl,jk->ikl"
t0 = perf_counter()
for _ in range(num):
    r1 = np.einsum("ijl,jk->ikl", p_tilde_p, 2 * p_tilde)
t1 = perf_counter()
for _ in range(num):
    r2 = (2 * p_tilde.T) @ p_tilde_p
t2 = perf_counter()
assert np.allclose(r1, r2)
print(
    f"Broadcasting faster than einsum: {t2 - t1 < t1 - t0}, ratio: {(t1 - t0)/(t2 - t1):.3f}"
)
#
t0 = perf_counter()
for _ in range(num):
    r1 = np.einsum("ij,jkl->ikl", 2 * p_tilde, p_tilde_p)
t1 = perf_counter()
for _ in range(num):
    r2 = (p_tilde_p.T @ (-2 * p_tilde)).T
t2 = perf_counter()
assert np.allclose(r1, r2)
print(
    f"Broadcasting faster than einsum: {t2 - t1 < t1 - t0}, ratio: {(t1 - t0)/(t2 - t1):.3f}"
)
#
