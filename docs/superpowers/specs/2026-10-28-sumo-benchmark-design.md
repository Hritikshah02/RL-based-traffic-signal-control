# SUMO Real-Data Benchmark: Design

Date: 2026-10-28
Status: approved (user delegated remaining decisions: "do what you think is best")

## Goal

Upgrade the toy Q-learning traffic-light project into a credible benchmark:
10 signal-control methods, evaluated on real camera-derived traffic in the
SUMO microscopic simulator, over long (multi-hour) periods with multiple seeds,
with results shown in the Streamlit dashboard. No hardware.

## Decisions

| Topic | Decision |
|---|---|
| Simulator | SUMO (pip `eclipse-sumo`) + `sumo-rl`, libsumo for speed |
| Python | conda env `tsc`, Python 3.10 (matches Kaggle) |
| Data | LibSignal `hangzhou_1x1_*`: 11 real hours, 5 junctions, 743–2231 veh/h |
| Network | Single junction now; package structured so a grid is a new scenario later |
| Old code | Moved to `legacy/`; dashboard has a Legacy / SUMO switch |
| Training | Local by default; Kaggle notebook path for long runs (repo is public → `git clone`) |

## Data and scenarios

All 11 hours share an identical abstract roadnet, so one SUMO network
(`data/sumo/hangzhou_1x1.net.xml`, from the `bc-tyc_10` folder) is used for every
hour; `data_convert.py` turns each `flow.json` into a SUMO `.rou.xml`.

- **Train split:** the 07:00 hours of the 5 junctions plus `bc-tyc_10`. 6 hours.
- **Test split:** the 08:00 hours (`bc-tyc`, `kn-hz`, `qc-yn`, `sb-sx`, `tms-xy`). These are
  the hours that follow the training hours, so this is a "train on the past, deploy on the
  next hour" test. 5 hours.
- **long_day:** all 11 hours concatenated into one continuous 11-hour run (the
  long-period benchmark).
- **Seeds:** seed 0 uses the exact recorded departures. Seeds 1–4 jitter each departure
  uniformly by ±30 s and change the SUMO seed, to measure robustness.

## Junction control model

- 8 green phases taken from the dataset's signal plan: 4 dual phases (NS-through,
  NS-left, EW-through, EW-left) and 4 single-approach phases.
- A decision is made every 5 s. Switching inserts a 3 s yellow (handled by sumo-rl).
  Min green 5 s, max green 60 s; max green is enforced by the env for every controller.
- Fixed, Webster, Actuated and SOTL cycle through the 4 dual phases, which is standard
  practice. Max-Pressure and all learning methods choose freely among the 8.

## Controllers (`tsc/controllers/`)

Interface: `reset()`, `act(feat) -> phase index`. `feat` is a `JunctionFeatures` snapshot
(per-lane queue, count, waiting time; per-phase incoming queue and pressure;
current phase; time in phase). The learning methods also implement `obs_vector(feat)`.

1. **Fixed**: 4 dual phases, 30 s each.
2. **Webster**: cycle `C0=(1.5L+5)/(1-Y)` with green split ∝ critical flow ratio,
   recomputed per hour from that hour's demand (a time-of-day plan). Greens are
   rounded to 5 s.
3. **Actuated**: gap-out logic. Extend green while a vehicle is moving inside the
   last 30 m before the stop line; otherwise go to the next dual phase that has demand.
   Min/max green 10/50 s.
4. **SOTL**: accumulate (vehicles on red × seconds); switch at threshold θ (tuned on
   the train split) unless a small platoon is about to clear.
5. **Max-Pressure**: argmax over phases of Σ(incoming queue − outgoing queue).
6. **Q-learning**: the existing tabular agent; state = per-approach queue buckets
   (0–3 / 4–9 / 10+) × current phase; reward = −Δ total waiting time.
7. **D3QN**: Dueling Double DQN (PyTorch MLP), same reward as #6.
8. **PPO**: stable-baselines3, same observation and reward.
9. **PressLight**: DQN whose state includes incoming-lane counts by segment and
   outgoing-lane counts; reward = −|pressure|.
10. **FRAP**: movement embeddings → phase demand → pairwise phase competition
    (conflict relation mask) → Q per phase; reward = −queue length.

Training budget is the same for every learning method: a fixed number of simulated
hours, with episodes sampled from the train split.

## Metrics (`tsc/metrics.py`)

Per vehicle, from SUMO `tripinfo` with the emissions device: travel time, time loss
(delay), waiting time, stops (waitingCount), CO2. Per decision step: total queue,
throughput, phase. Summary per (method, scenario, seed): mean delay, p95 waiting
time, mean queue, throughput, stops/veh, CO2/veh. Vehicles that have not finished are
counted as inserted-but-unfinished (reported, not dropped silently).

## Outputs

- `models/<method>.{pt,zip,pkl}`
- `results/summary.csv` (one row per method × scenario × seed)
- `results/timeseries/<scenario>/<method>_s<seed>.csv`

## Dashboard (`app.py`)

A mode switch between Legacy and SUMO. The SUMO mode has:
- a leaderboard (summary table plus bar charts with error bars across seeds)
- the long-day time series (queue and delay over 11 h, one line per method)
- a per-scenario breakdown (train vs. test generalisation)
- a replay of one run (phase timeline and queues)

## Kaggle

`kaggle/train_notebook.py` is pushed with the `kaggle` CLI (`kaggle/run.sh`). It clones
the public repo, installs requirements, trains the selected methods, and writes
`models/` to the kernel output, which `kaggle kernels output` pulls back.

## Testing

pytest covers data conversion (vehicle counts preserved), the env (phases and yellow,
features consistent with libsumo), the rule-based controllers on synthetic features,
metric aggregation, and a one-episode smoke test for each learning method.

## Out of scope (later)

Multi-junction grid and coordination methods (CoLight), a live camera pipeline, and
hardware.

## Changes made during implementation

- **Data:** used LibSignal's real single-junction Hangzhou hours instead of cropping the Jinan or Hangzhou grids,
  because they ship a matching SUMO network and need no cropping.
- **Reward (after ablation):** methods 6–8 moved from the waiting-time-difference reward to a queue-length reward.
  Validation gained the heaviest train hour (`tmsxy_07`). Round-1 results are kept in
  `results/ablation_wait_reward.csv`.
- **Q-learning:** the state now uses dual-phase queue buckets, and unseen states keep the current phase
  (no heuristic fallback).
- **Dashboard:** the "replay one run" view was dropped in favour of a per-hour heatmap and the reward ablation
  table. The legacy simulator is a Streamlit page (`pages/`) rather than a toggle.
- The 5 rule-based controllers live in one module (`tsc/controllers/rule_based.py`).
