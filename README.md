# Traffic Signal Control on Real Traffic: a 10-method benchmark

Started as a toy Q-learning traffic light, this is now a benchmark of **10 signal-control methods**
on a **real junction's traffic** in the **SUMO** microscopic simulator:

- **Real data:** camera-recorded vehicles from **5 Hangzhou junctions, 11 hours** (2018-04-16,
  743–2,231 veh/h), via the [LibSignal](https://github.com/DaRL-LibSignal/LibSignal) /
  [TSC benchmark](https://traffic-signal-control.github.io/) datasets.
- **Realistic simulation:** SUMO car-following, 2 lanes per approach (through + left),
  8 signal phases, 3 s yellow, min/max green, per-vehicle delay and CO₂.
- **Fair, long-period evaluation:** every method sees the same vehicles. Learning methods train on the
  07:00 hours and are tested on the **unseen 08:00 hours**. There is also an **11-hour continuous run**,
  and **5 seeds** (seeds 1–4 jitter each departure by ±30 s).

The original toy simulator is kept in `legacy/` and still runs as a page of the dashboard.

---

## Quick start

```bash
conda create -n tsc python=3.10 -y && conda activate tsc
pip install -r requirements.txt

python -m tsc.data_convert            # real data -> SUMO network + routes (auto on first use)
python -m tsc.train --method all      # train the 5 learning methods (CPU, ~1 h; models/ is committed)
python -m tsc.benchmark --methods all --scenarios all,long_day --seeds 5
streamlit run app.py                  # dashboard
pytest                                # tests
```

Optional: `python -m tsc.tune` grid-searches rule-based parameters on the training hours.
`kaggle/run.sh` trains on Kaggle instead (see below).

---

## The 10 methods

| # | Method | Family | Idea |
|---|---|---|---|
| 1 | **Fixed** | rule-based | 4-phase cycle, 30 s greens (the classic baseline) |
| 2 | **Webster** | rule-based | Optimal fixed-time plan per hour from Webster's formula, C₀ = (1.5L+5)/(1−Y) |
| 3 | **Actuated** | rule-based | Gap-out: extend green while vehicles cross a 30 m detector |
| 4 | **SOTL** | rule-based | Self-organising: switch when vehicles-on-red × time exceeds θ |
| 5 | **Max-Pressure** | rule-based | Serve the phase with the largest Σ(incoming − downstream) vehicles |
| 6 | **Q-learning** | learning | The original tabular agent: queue buckets per approach × phase |
| 7 | **D3QN** | learning | Dueling Double DQN over lane queues and counts |
| 8 | **PPO** | learning | Policy-gradient actor-critic (stable-baselines3) |
| 9 | **PressLight** | learning | DQN with position-aware lane counts and a pressure reward (KDD 2019) |
| 10 | **FRAP** | learning | Phase-competition network, invariant to junction symmetry (CIKM 2019) |

Rule-based parameters are in `configs/controllers.yaml` and learning hyper-parameters in
`configs/training.yaml`. Every learning method gets the same budget of simulated training hours.

## Results

RESULTS_PLACEHOLDER

---

## How it works

```
data/raw/ (CityFlow flow.json, 11 h)
   └─ tsc/data_convert.py ──► data/sumo/  net + one route file per hour + long_day (11 h)
tsc/scenarios.py   train / test splits, seeds (±30 s jitter)
tsc/env.py         TscEnv: sumo-rl + libsumo, 8 phases, 5 s decisions, yellow 3 s,
                   min green 5 s, max green 60 s (enforced for every controller),
                   JunctionFeatures snapshot, rewards: wait / pressure / queue
tsc/controllers/   rule_based.py (5)   learning.py (5 + networks)
tsc/train.py       training with best-checkpoint selection on validation hours
tsc/benchmark.py   method × scenario × seed in parallel ─► results/summary.csv, results/timeseries/
tsc/metrics.py     SUMO tripinfo ─► delay, p95, stops, CO₂, throughput, queue
app.py             Streamlit dashboard (+ pages/ legacy toy simulator)
```

**Junction.** 4 approaches × (through lane, left lane). The 8 phases are EW-through, NS-through,
EW-left, NS-left and one per approach. A controller picks a phase every 5 s; switching inserts a 3 s yellow.

**Metrics.** Delay is SUMO `timeLoss` plus insertion delay, counting vehicles that are still in the network
at the end of a run (none are silently dropped). CO₂ comes from SUMO's HBEFA model.

## Training on Kaggle

At about 1–2 s per simulated hour, SUMO runs quickly enough on a laptop CPU. Kaggle is useful for longer
budgets in the background:

```bash
git push -u origin sumo-benchmark          # the kernel clones this branch
KAGGLE=/path/to/recent/kaggle kaggle/run.sh all 300   # methods, hours; pulls models/ back when done
```

This needs a Kaggle API token in `~/.kaggle/` and a phone-verified account (GPU and internet in kernels).

## Project structure

| Path | Purpose |
|---|---|
| `tsc/` | Benchmark package (data, env, controllers, training, benchmark, metrics) |
| `configs/` | Controller and training parameters |
| `data/raw/` | Original real-world data (LibSignal Hangzhou 1×1) |
| `models/` | Trained models and learning curves |
| `results/` | Benchmark summary and per-minute time series |
| `app.py`, `pages/` | Dashboard: SUMO benchmark and the legacy toy simulator |
| `legacy/` | The original toy Q-learning project |
| `tests/` | pytest suite |
| `docs/superpowers/` | Design spec and implementation plan |
