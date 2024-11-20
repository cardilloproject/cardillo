import numpy as np
from time import perf_counter
num = 100000
#################
# Exp_SO3_quat_p
#################
p0 = np.random.random(1)
P2 = np.random.random(1)
P = np.random.random(3)
p_tilde = np.random.random((3,3))
p_tilde_p = np.random.random((3,3,3))
# np.array_split
t0 = perf_counter()
for _ in range(num):
    r1, _ = np.array_split(P, [1])
t1 = perf_counter()
for _ in range(num):
    r2, _ = P[0, None], P[1:]
t2 = perf_counter()
print(f"Slicing syntax faster than np.array_split: {t2 - t1 < t1 - t0}, ratio: {(t1 - t0)/(t2 - t1):.3f}")
# "ijl,jk->ikl"
t0 = perf_counter()
for _ in range(num):
    r1 = np.einsum("ijl,jk->ikl", p_tilde_p, 2 * p_tilde)
t1 = perf_counter()
for _ in range(num):
    r2 = (2 * p_tilde.T) @ p_tilde_p
t2 = perf_counter()
assert np.allclose(r1, r2)
print(f"Broadcasting faster than einsum: {t2 - t1 < t1 - t0}, ratio: {(t1 - t0)/(t2 - t1):.3f}")
# "ij,k->ijk"
t0 = perf_counter()
for _ in range(num):
    r1 = np.einsum("ij,k->ijk", p0 * p_tilde + p_tilde @ p_tilde, -(4 / (P2 * P2)) * P)
t1 = perf_counter()
for _ in range(num):
    r2 = (p0 * p_tilde + p_tilde @ p_tilde)[..., None] * (-(4 / (P2 * P2)) * P)
t2 = perf_counter()
assert np.allclose(r1, r2)
print(f"Broadcasting faster than einsum: {t2 - t1 < t1 - t0}, ratio: {(t1 - t0)/(t2 - t1):.3f}")
#