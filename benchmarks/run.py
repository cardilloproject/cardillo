"""Benchmark runner.

python benchmarks/run.py                      # time all cases
python benchmarks/run.py balls chain          # time selected cases
python benchmarks/run.py --profile balls      # cProfile, top 25 cumulative
python benchmarks/run.py --save baseline      # store results in benchmarks/results/baseline/
python benchmarks/run.py --check baseline     # compare against stored results
"""

import argparse
import cProfile
import pstats
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cases import CASES  # noqa: E402

RESULTS = Path(__file__).parent / "results"
# Cases that solve linear systems iteratively (MINRES, rtol=1e-5): results only
# agree up to the solver tolerance when the rounding of a matvec changes.
LOOSE = {"rod2plane": (1e-3, 1e-3)}
# cases that reproduce another case and are compared with its baseline
BASELINE_OF = {
    "balls_world": "balls",
    "balls_world_dsv": "balls_dsv",
    "balls_world_200": "balls_200",
}
FIELDS = ["t", "q", "u", "la_g", "la_gamma", "la_c", "la_N", "la_F", "P_N", "P_F"]


def extract(sol):
    out = {}
    for f in FIELDS:
        v = getattr(sol, f, None)
        if v is not None:
            out[f] = np.asarray(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--save", metavar="NAME")
    ap.add_argument("--check", metavar="NAME")
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-10)
    args = ap.parse_args()
    names = args.cases or list(CASES)
    warnings.filterwarnings("ignore")

    ok = True
    for name in names:
        fun = CASES[name]
        t0 = time.perf_counter()
        if args.profile:
            pr = cProfile.Profile()
            pr.enable()
            sol = fun()
            pr.disable()
            st = pstats.Stats(pr)
            st.sort_stats("cumulative").print_stats(22)
            st.sort_stats("tottime").print_stats(14)
        else:
            sol = fun()
        dt = time.perf_counter() - t0
        res = extract(sol)
        msg = f"{name:26s} {dt:8.2f} s"
        if args.save:
            d = RESULTS / args.save
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{name}.npz", **res, wall_time=dt)
        if args.check:
            ref_name = BASELINE_OF.get(name, name)
            ref = np.load(RESULTS / args.check / f"{ref_name}.npz")
            bad = []
            for k, v in res.items():
                if k not in ref:
                    continue  # field not stored in older baselines
                if ref[k].shape != v.shape:
                    bad.append(f"{k}(shape)")
                elif not np.allclose(
                    v, ref[k], *(LOOSE.get(name) or (args.rtol, args.atol))
                ):
                    bad.append(f"{k}(max|d|={np.max(np.abs(v - ref[k])):.2e})")
            msg += f"   baseline {float(ref['wall_time']):7.2f} s   "
            msg += "OK" if not bad else "MISMATCH " + ", ".join(bad)
            ok &= not bad
        print(msg, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
