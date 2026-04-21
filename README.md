# RL Traffic Light Controller

A **Q-learning based traffic light controller** that learns to optimise signal timing at a 4-way intersection. Includes a Streamlit dashboard for real-time visualisation and a fixed-timer baseline for comparison.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the Q-learning agent (creates q_table.pkl)
python qlearning_agent.py

# 3. Launch the dashboard
streamlit run app.py
```

> You can also train directly from the dashboard sidebar.

---

## Project Structure

| File | Purpose |
|------|---------|
| `utils.py` | Vehicle class, constants, traffic generation helpers |
| `traffic_env.py` | 4-way intersection environment (state, step, reward) |
| `qlearning_agent.py` | Q-learning agent, training loop, save/load Q-table |
| `simulator.py` | SimulationRunner + FixedTimerController (baseline) |
| `metrics.py` | MetricsTracker, comparison table generation |
| `app.py` | Streamlit dashboard with intersection viz, graphs, comparison |

---

## How It Works

### Environment

The intersection has four approach directions (N, S, E, W). Each direction has a queue of waiting vehicles. At each time step the controller chooses which direction gets the green signal; up to 2 vehicles pass from that queue while all other queues accumulate new arrivals.

### State Space

The state is a tuple of **5 values**:

```
(bucket_N, bucket_S, bucket_E, bucket_W, current_green_index)
```

Queue lengths are bucketised into 3 levels (low 0-3, medium 4-7, high 8+), giving **3⁴ × 4 = 324 discrete states**.

### Actions

4 actions — set the green signal to N (0), S (1), E (2), or W (3). A minimum-green constraint of 2 steps prevents signal flickering.

### Reward Function

```
reward = -(total_queue) + (passed × 2.0) - (10 × queues_over_10)
```

The agent is penalised for long queues and rewarded for throughput.

### Q-Learning

| Hyperparameter | Value |
|----------------|-------|
| Learning rate (α) | 0.10 |
| Discount factor (γ) | 0.95 |
| Initial ε | 1.0 |
| Min ε | 0.01 |
| ε decay | 0.995 per episode |
| Episodes | 500 |
| Steps/episode | 200 |

Training uses randomly mixed traffic intensities (Low / Medium / Heavy) so the agent generalises.

### Fixed-Timer Baseline

Cycles through N → S → E → W every 5 steps regardless of traffic. Used as a baseline to demonstrate the RL agent's advantage.

---

## Dashboard Features

- **Mode toggle** — switch between RL and Fixed Timer
- **Traffic intensity** — Low / Medium / Heavy
- **Intersection visualisation** — Plotly-based view with traffic lights, vehicles, and queue counts
- **Live metrics** — active signal, queue lengths, wait time, throughput, reward
- **Historical charts** — queue sizes, wait time, throughput, and reward over time
- **Comparison table** — side-by-side RL vs Fixed Timer statistics

---

## Metrics Explained

| Metric | Description |
|--------|-------------|
| Avg Queue Length | Mean total vehicles waiting across all directions |
| Avg Wait Time | Mean steps a vehicle waits before passing |
| Total Throughput | Cumulative vehicles that passed through |
| Max Queue | Peak total queue observed |
| Avg Reward | Mean per-step reward |

---

## Future Improvements

- Deep Q-Network (DQN) for larger/continuous state spaces
- Multi-intersection coordination
- Pedestrian crossing phases
- Real-world traffic data integration
- Turn-lane specific signals
