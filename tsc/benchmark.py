"""
benchmark.py - Run every controller on the same real traffic and record results.

    python -m tsc.benchmark --methods all --scenarios test,long_day --seeds 5 --jobs 8

Writes
  results/summary.csv                                  one row per method × scenario × seed
  results/timeseries/<scenario>/<method>_s<seed>.csv   per-minute queue / delay / throughput
Re-running replaces only the rows it recomputes.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd

import tsc
from tsc import RESULTS_DIR, scenarios
from tsc.metrics import SUMMARY_KEYS, per_minute, read_tripinfo, summarize

SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.csv")


def run_one(method: str, scenario: str, seed: int) -> Optional[Dict]:
    """Run one controller on one scenario/seed in this process; return its summary row."""
    import torch

    torch.set_num_threads(1)
    from tsc.controllers import make
    from tsc.env import TscEnv, tmp_tripinfo

    try:
        controller = make(method)
    except FileNotFoundError as e:
        print(f"[skip] {method}: {e}")
        return None

    trip_path = tmp_tripinfo(f"{method}_{scenario}_{seed}")
    env = TscEnv(scenario, seed=seed, tripinfo=trip_path)
    env.reset()
    controller.reset(env)
    t0, done = time.time(), False
    while not done:
        _, _, _, done, _ = env.step(controller.act(env.features()))

    steps = pd.DataFrame(env.step_log)
    trips = read_tripinfo(trip_path)
    os.remove(trip_path)
    demand = _route_vehicle_count(env.scenario.route)
    row = {"method": method, "scenario": scenario, "seed": seed,
           **summarize(trips, steps, demand, env.scenario.seconds),
           "runtime_s": round(time.time() - t0, 1)}

    ts_dir = os.path.join(RESULTS_DIR, "timeseries", scenario)
    os.makedirs(ts_dir, exist_ok=True)
    per_minute(steps, trips).to_csv(os.path.join(ts_dir, f"{method}_s{seed}.csv"), index=False)
    return row


def _route_vehicle_count(route_file: str) -> int:
    with open(route_file) as f:
        return sum(1 for line in f if "<vehicle " in line)


def save_rows(rows: List[Dict]) -> pd.DataFrame:
    """Merge new rows into results/summary.csv (replacing same method/scenario/seed)."""
    new = pd.DataFrame(rows)
    if os.path.exists(SUMMARY_PATH):
        old = pd.read_csv(SUMMARY_PATH)
        keep = ~old.set_index(SUMMARY_KEYS).index.isin(new.set_index(SUMMARY_KEYS).index)
        new = pd.concat([old[keep], new], ignore_index=True)
    new = new.sort_values(SUMMARY_KEYS).reset_index(drop=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    new.to_csv(SUMMARY_PATH, index=False)
    return new


def main() -> None:
    from tsc.controllers import ALL_METHODS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="all", help="comma list or 'all'")
    ap.add_argument("--scenarios", default="test,long_day", help="tags, 'train', 'test', 'all', 'long_day'")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    methods = ALL_METHODS if args.methods == "all" else args.methods.split(",")
    names = scenarios.expand(args.scenarios)
    scenarios.ensure_built()
    jobs = [(m, s, seed) for s in names for seed in range(args.seeds) for m in methods]
    # build jittered route files once, before workers race to create them
    for s in names:
        for seed in range(args.seeds):
            scenarios.get(s, seed)

    print(f"Running {len(jobs)} runs on {args.jobs} processes ...")
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, *j): j for j in jobs}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                rows.append(row)
                print(f"  {row['method']:<12} {row['scenario']:<9} s{row['seed']}  "
                      f"delay {row['mean_delay']:7.1f}s  queue {row['mean_queue']:6.1f}  "
                      f"unfinished {row['unfinished']}")
    if rows:
        df = save_rows(rows)
        table = (df[df["method"].isin(methods) & df["scenario"].isin(names)]
                 .groupby(["scenario", "method"])[["mean_delay", "p95_wait", "mean_queue", "throughput_vph"]]
                 .mean().round(1))
        print(table.to_string())


if __name__ == "__main__":
    main()
