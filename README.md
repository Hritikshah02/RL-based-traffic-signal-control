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
python -m tsc.train --method all      # train the 5 learning methods (~20 min on 8 CPU cores; models/ is committed)
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

**Held-out test hours (5 × 08:00, never seen in training)**, mean over 5 seeds (± std across seeds)

| Rank | Method | Family | Mean delay (s) | vs Fixed | vs Max-Pressure | p95 delay (s) | Stops/veh | CO₂/veh (g) |
|---|---|---|---|---|---|---|---|---|
| 1 | **PressLight** | learning | 26.7 ± 0.9 | -90% | -21% | 86 | 0.67 | 141 |
| 2 | FRAP | learning | 34.0 ± 0.4 | -88% | ±0% | 105 | 0.91 | 153 |
| 3 | MaxPressure | rule-based | 34.1 ± 0.7 | -88% | — | 116 | 0.76 | 153 |
| 4 | Actuated | rule-based | 35.3 ± 0.7 | -87% | +4% | 88 | 0.77 | 154 |
| 5 | Webster | rule-based | 52.7 ± 1.0 | -81% | +55% | 126 | 1.01 | 181 |
| 6 | D3QN | learning | 54.2 ± 1.9 | -80% | +59% | 145 | 1.89 | 178 |
| 7 | PPO | learning | 62.9 ± 2.6 | -77% | +85% | 187 | 1.94 | 191 |
| 8 | SOTL | rule-based | 127.6 ± 8.4 | -54% | +275% | 437 | 1.80 | 240 |
| 9 | QLearning | learning | 141.5 ± 25.0 | -48% | +316% | 446 | 2.13 | 275 |
| 10 | Fixed | rule-based | 274.4 ± 2.6 | — | +706% | 858 | 1.96 | 321 |

**Long day: 11 real hours back to back (~18,000 vehicles)**, mean over 5 seeds (± std across seeds)

| Rank | Method | Family | Mean delay (s) | vs Fixed | vs Max-Pressure | p95 delay (s) | Stops/veh | CO₂/veh (g) |
|---|---|---|---|---|---|---|---|---|
| 1 | **PressLight** | learning | 27.2 ± 1.0 | -98% | -23% | 90 | 0.69 | 143 |
| 2 | MaxPressure | rule-based | 35.2 ± 0.3 | -97% | — | 119 | 0.79 | 156 |
| 3 | Actuated | rule-based | 36.2 ± 0.4 | -97% | +3% | 95 | 0.78 | 157 |
| 4 | FRAP | learning | 36.5 ± 0.8 | -97% | +4% | 121 | 0.99 | 159 |
| 5 | Webster | rule-based | 54.9 ± 1.1 | -95% | +56% | 136 | 1.05 | 187 |
| 6 | D3QN | learning | 55.6 ± 4.2 | -95% | +58% | 223 | 1.85 | 179 |
| 7 | PPO | learning | 72.1 ± 2.0 | -93% | +105% | 249 | 2.16 | 204 |
| 8 | QLearning | learning | 247.8 ± 83.8 | -78% | +604% | 892 | 2.24 | 312 |
| 9 | SOTL | rule-based | 310.1 ± 82.9 | -72% | +780% | 1639 | 2.40 | 282 |
| 10 | Fixed | rule-based | 1107.3 ± 22.1 | — | +3044% | 3386 | 2.93 | 445 |

Delay = time lost compared with free flow, plus time spent waiting to enter, per vehicle. Lower is better.
All 600 runs (10 methods × 12 scenarios × 5 seeds) are in `results/summary.csv`.

### What the results show

1. **PressLight is the best method overall.** On the unseen test hours it cuts mean delay by **21% against
   Max-Pressure**, the strongest classical controller, and by 90% against fixed-time. It also has the lowest
   stops and CO₂. The margin holds over the 11-hour day.
2. **Max-Pressure and Actuated control are hard to beat.** With no training at all, they match or beat
   3 of the 5 learning methods. FRAP ties Max-Pressure.
3. **Generalisation is the real test for RL.** D3QN and PPO match or beat Max-Pressure on the lighter test hours
   (e.g. knhz_08: D3QN 13 s vs MP 18 s), but degrade on `bctyc_08`, whose 2,231 veh/h is **10% above any
   training hour** (D3QN 139 s vs MP 53 s). PressLight's pressure state and reward scale with demand, so it
   generalises (40 s).
4. **Fixed-time collapses under real demand.** Queues carry over from hour to hour in the long-day run, and
   delay grows to about 18 minutes per vehicle. Webster's per-hour optimised plan fixes most of this, with no
   sensing needed.
5. **Tabular Q-learning (the original agent) does not scale.** With 1,296 discrete states it cannot represent a
   real junction with 8 phases. It beats only fixed-time, which is the motivation for the deep methods.

### Ablation: the reward function matters as much as the algorithm

The first training round used the waiting-time-difference reward (the sumo-rl default) for Q-learning, D3QN and PPO,
and checked policies on two moderate hours. D3QN reached 42 s on the training hours, but on the 11-hour day its
delay went to **919 s**: it held phases until the max-green override fired 276 times. Switching to a
**queue-length reward** and adding the heaviest training hour to validation fixed this (919 s → 56 s).
PPO was similar under both rewards (55 s → 63 s on test). Round-1 results are in `results/ablation_wait_reward.csv`
and `models/ablation_wait_reward/`, and the dashboard compares them side by side. Q-learning's round-1 score was
inflated by a "longest queue" fallback for unseen states. That fallback is now removed, so its score reflects only what it learned.

### Next steps

- **Demand augmentation:** train on randomly scaled demand (0.8–1.4×) so D3QN and PPO cover peaks like `bctyc_08`.
- **Multi-junction:** LibSignal's Hangzhou 4×4 and Jinan 3×4 networks, with coordination methods (CoLight, MPLight).
- **Live demand:** feed counts from a camera detector (YOLO + tracking) into `TscEnv`'s route input.


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
`TSC_COPY=0` keeps the downloaded models in `kaggle/out/result/models` instead of overwriting `models/`.

**Checked end to end:** an independent Kaggle retrain of all 5 methods (150 h each, ~25 min) scored as follows on the
test hours (seeds 0–2, mean delay): PressLight 25.3 s (local 26.6), FRAP 38.8 (34.0), D3QN 62.8 (54.1), PPO 63.2 (63.2).
The ranking holds; the differences are run-to-run training variance. The committed models are the local ones.

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
