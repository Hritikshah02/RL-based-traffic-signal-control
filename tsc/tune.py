"""
tune.py - Grid-search rule-based parameters on the TRAIN hours only.

    python -m tsc.tune

Prints mean delay per value; copy the winners into configs/controllers.yaml.
(Test hours are never used here, so the benchmark stays a fair held-out test.)
"""

import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple

import tsc
from tsc import scenarios
from tsc.controllers import rule_based as rb

GRID: Dict[str, Tuple[str, list]] = {
    "SOTL": ("theta", [50, 100, 200, 400, 800, 1600, 3200]),
    "MaxPressure": ("min_green", [5, 10, 15, 20, 30]),
    "Actuated": ("min_green", [5, 10, 15, 20]),
}


def _make(method: str, value):
    ctrl = {"SOTL": rb.SOTLController, "MaxPressure": rb.MaxPressureController,
            "Actuated": rb.ActuatedController}[method]()
    setattr(ctrl, GRID[method][0], value)
    return ctrl


def _delay(job) -> Tuple[str, object, str, float]:
    from tsc.env import TscEnv, tmp_tripinfo
    from tsc.metrics import read_tripinfo

    method, value, tag = job
    trip = tmp_tripinfo(f"tune_{method}")
    env = TscEnv(tag, tripinfo=trip)
    env.reset()
    ctrl = _make(method, value)
    ctrl.reset(env)
    done = False
    while not done:
        _, _, _, done, _ = env.step(ctrl.act(env.features()))
    return method, value, tag, float(read_tripinfo(trip)["delay"].mean())


def main() -> None:
    scenarios.ensure_built()
    jobs = [(m, v, t) for m, (_, vals) in GRID.items() for v in vals for t in scenarios.TRAIN_TAGS]
    results: Dict[Tuple[str, object], list] = {}
    with ProcessPoolExecutor() as pool:
        for m, v, _, d in pool.map(_delay, jobs):
            results.setdefault((m, v), []).append(d)
    for method, (param, vals) in GRID.items():
        scores = {v: sum(results[(method, v)]) / len(results[(method, v)]) for v in vals}
        best = min(scores, key=scores.get)
        print(f"{method:<12} {param}: " + "  ".join(f"{v}->{s:.1f}s" for v, s in scores.items())
              + f"   BEST {best}")


if __name__ == "__main__":
    main()
