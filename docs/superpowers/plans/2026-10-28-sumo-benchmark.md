# SUMO Real-Data Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark 10 traffic-signal controllers on real Hangzhou traffic in SUMO and show results in Streamlit.

**Architecture:** `tsc/` package. `env.py` wraps sumo-rl's `SumoEnvironment` (single junction, 8 phases,
5 s decisions) and exposes a `JunctionFeatures` snapshot. Controllers implement `act(feat)`.
`benchmark.py` runs controller × scenario × seed in parallel processes and writes CSVs, which `app.py` reads.

**Tech Stack:** Python 3.10, eclipse-sumo + libsumo, sumo-rl, gymnasium, PyTorch (CPU), stable-baselines3, Streamlit, Plotly, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-sumo-benchmark-design.md`

## Global Constraints

- Conda env `tsc` (Python 3.10). All commands use `/home/hritik/anaconda3/envs/tsc/bin/python`.
- delta_time 5 s, yellow 3 s, min green 5 s, max green 60 s (env-enforced).
- Train split = 07:00 hours + bc-tyc_10. Test split = 08:00 hours. long_day = all 11 hours chained.
- Seed 0 = exact departures. Seeds 1–4 = ±30 s jitter plus a different SUMO seed.
- Old modules live in `legacy/`, and `streamlit run app.py` still serves them through the mode switch.

## Review Focus

- Unfinished vehicles at the end of an episode → counted in the summary (`unfinished` column), not dropped silently.
- Max-green: every controller, including the learned ones, is forced off a phase after 60 s.
- Phase choices that sumo-rl can't apply yet (still within min green): the env keeps the current phase, and the logged phase is the applied one.
- Missing `models/<m>.*` when benchmarking → that method is skipped with a printed warning; the rest of the run continues.
- long_day route file: vehicle IDs must be unique across the chained hours (prefix with the hour tag).

---

### Task 1: Repo restructure + data conversion
**Files:** move old modules to `legacy/` and fix the imports; add `.gitignore`, `requirements.txt` and `tsc/data_convert.py`; test in `tests/test_data.py`.
**Produces:** `tsc.data_convert.build_all() -> None` writes `data/sumo/hangzhou_1x1.net.xml`,
`data/sumo/<tag>.rou.xml` for each hour (tag such as `bctyc_07`), and `data/sumo/long_day.rou.xml`.
`tsc.scenarios.get(name, seed) -> Scenario(name, net, route_file, seconds)`. Names: each tag, `train`
(samples a train tag), `long_day`.
- [ ] Test: the vehicle count in each rou.xml equals the length of its flow.json; long_day count = sum of the 11 hours; IDs are unique; departures are sorted.
- [ ] Test: jitter is deterministic for a given seed, and seed 0 matches the exact departures.
- [ ] Implement, run pytest, commit.

### Task 2: Environment + features
**Files:** `tsc/env.py`, `tests/test_env.py`.
**Produces:** `TscEnv(scenario, reward="wait"|"pressure"|"queue", obs="vector"|"presslight"|"frap")`,
a gymnasium Env with `reset(seed) -> (obs, info)` and `step(phase) -> (obs, r, term, trunc, info)`.
`env.features() -> JunctionFeatures` with fields `lane_queue`, `lane_count`, `lane_wait`, `phase_queue[8]`,
`phase_pressure[8]`, `approach_queue[4]`, `movement_count[8]`, `current_phase`, `time_in_phase`,
`approach_moving_near_stop[4]`. `env.close()`. tripinfo is written to a temp path given in `info["tripinfo"]` at the end.
- [ ] Test: a 1-hour run with a constant action finishes; `phase_queue` has length 8; a forced switch after max green is observed.
- [ ] Implement, run, commit.

### Task 3: Rule-based controllers + metrics + benchmark runner
**Files:** `tsc/controllers/{base,fixed,webster,actuated,sotl,max_pressure}.py`, `tsc/metrics.py`, `tsc/benchmark.py`, `tests/test_controllers.py`.
**Produces:** `Controller.reset()`, `Controller.act(feat) -> int`; `metrics.summarize(tripinfo_path, steps_df) -> dict`;
`python -m tsc.benchmark --methods all --scenarios test,long_day --seeds 5 --jobs 8`.
- [ ] Test: Max-Pressure picks the argmax on synthetic features; Fixed cycles through 0,1,2,3 at 30 s; Actuated holds green while a vehicle is moving near the stop line; Webster's split sums to its cycle.
- [ ] Run rule-based methods on a single hour as a smoke test; commit.

### Task 4: Learning controllers + training
**Files:** `tsc/controllers/{qlearning,dqn,ppo,presslight,frap}.py`, `tsc/train.py`, `tests/test_learning.py`.
**Produces:** `python -m tsc.train --method <m> --hours N` saves to `models/`; `load(method) -> Controller`.
- [ ] Test: each method trains for 1 short episode and its controller loads and acts.
- [ ] Train all 5; check they beat Fixed on the train split; tune and iterate.

### Task 5: Kaggle path
**Files:** `kaggle/train_kernel.py`, `kaggle/kernel-metadata.json`, `kaggle/run.sh`.
- [ ] Push the kernel, check it runs, and pull the output back.

### Task 6: Full benchmark + dashboard + README
**Files:** `app.py` (mode switch), `dashboard/sumo_view.py`, `README.md`.
- [ ] Run the full benchmark, check every summary row, render the dashboard, update the README, commit.
