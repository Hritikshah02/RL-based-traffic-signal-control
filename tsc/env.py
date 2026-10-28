"""
env.py - Single-junction SUMO environment shared by every controller.

Wraps sumo-rl's SumoEnvironment (which handles yellow transitions and min-green)
and adds:
  - a JunctionFeatures snapshot (queues, counts, pressure, ...) after every decision
  - max-green enforcement for every controller
  - per-decision time-series logging and SUMO tripinfo output for evaluation
  - pluggable reward ("wait" | "pressure" | "queue") and observation encoding

Action space : Discrete(8) green phases, taken from the dataset's signal plan:
               0 EW-through, 1 NS-through, 2 EW-left, 3 NS-left,
               4 W-only, 5 E-only, 6 S-only, 7 N-only
"""

import os
import random
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np

import tsc
from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction

from tsc import scenarios
from tsc.scenarios import Scenario

# ── Control timing (seconds) ────────────────────────────────────────────────
DELTA_TIME = 5      # one decision every 5 s
YELLOW_TIME = 3     # inserted automatically on every switch
MIN_GREEN = 5
MAX_GREEN = 60      # enforced by TscEnv for every controller
DETECTOR_M = 30.0   # actuated "loop detector" zone before the stop line
SEGMENTS = 3        # lane segments for PressLight's position-aware counts

PHASE_NAMES = ["EW-through", "NS-through", "EW-left", "NS-left",
               "W-only", "E-only", "S-only", "N-only"]
DUAL_PHASES = [1, 3, 0, 2]   # standard 4-phase cycle: NS-T, NS-L, EW-T, EW-L
APPROACHES = ["N", "E", "S", "W"]
_EDGE_APPROACH = {"road_1_2_3": "N", "road_2_1_2": "E", "road_1_0_1": "S", "road_0_1_0": "W"}
TS_ID = "intersection_1_1"


@dataclass
class JunctionFeatures:
    """Snapshot of the junction after a decision step. Lane order = env.in_lanes."""
    time: float
    lane_queue: np.ndarray            # halted vehicles per incoming lane
    lane_count: np.ndarray            # vehicles per incoming lane
    lane_wait: np.ndarray             # summed waiting time per incoming lane (s)
    lane_segments: np.ndarray         # (lanes, SEGMENTS) counts, segment 0 nearest stop line
    lane_moving_near_stop: np.ndarray # moving vehicles inside the detector zone
    out_count: np.ndarray             # mean vehicles on each lane's downstream lanes
    lane_approach: List[int]          # approach index (N,E,S,W) of each incoming lane
    phase_lanes: List[List[int]]      # incoming lanes served by each green phase
    current_phase: int
    time_in_phase: float              # seconds since the last switch (includes yellow)
    green_time: float                 # seconds the current phase has actually been green
    is_yellow: bool

    @property
    def num_phases(self) -> int:
        return len(self.phase_lanes)

    @property
    def can_switch(self) -> bool:
        return not self.is_yellow and self.time_in_phase >= YELLOW_TIME + MIN_GREEN

    @property
    def lane_pressure(self) -> np.ndarray:
        return self.lane_count - self.out_count

    @property
    def phase_queue(self) -> np.ndarray:
        return np.array([self.lane_queue[l].sum() for l in self.phase_lanes])

    @property
    def phase_pressure(self) -> np.ndarray:
        return np.array([self.lane_pressure[l].sum() for l in self.phase_lanes])

    @property
    def approach_queue(self) -> np.ndarray:
        q = np.zeros(len(APPROACHES))
        np.add.at(q, self.lane_approach, self.lane_queue)
        return q


# ── Observation encodings (one per learning-method family) ───────────────────
def _onehot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1.0
    return v


def obs_vector(f: JunctionFeatures) -> np.ndarray:
    """Queues + counts + phase one-hot + green time + can-switch flag (D3QN / PPO)."""
    return np.concatenate([
        f.lane_queue / 20.0, f.lane_count / 20.0, _onehot(f.current_phase, f.num_phases),
        [min(f.green_time, MAX_GREEN) / MAX_GREEN, float(f.can_switch)],
    ]).astype(np.float32)


def obs_presslight(f: JunctionFeatures) -> np.ndarray:
    """Segmented incoming counts + downstream counts + phase one-hot (PressLight paper)."""
    return np.concatenate([
        f.lane_segments.flatten() / 10.0, f.out_count / 20.0, _onehot(f.current_phase, f.num_phases),
    ]).astype(np.float32)


def obs_frap(f: JunctionFeatures) -> np.ndarray:
    """Per-movement vehicle counts + current phase one-hot (FRAP builds phases internally)."""
    return np.concatenate([f.lane_count / 20.0, _onehot(f.current_phase, f.num_phases)]).astype(np.float32)


OBS_FNS = {"vector": obs_vector, "presslight": obs_presslight, "frap": obs_frap}


class _NullObservation(ObservationFunction):
    """sumo-rl requires an observation function; TscEnv computes its own."""

    def __call__(self):
        return np.zeros(1, dtype=np.float32)

    def observation_space(self):
        return gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)


class _CountingSumo(SumoEnvironment):
    """SumoEnvironment that also counts arrived (finished) vehicles every second."""

    arrived_total = 0

    def _sumo_step(self):
        super()._sumo_step()
        self.arrived_total += self.sumo.simulation.getArrivedNumber()


class TscEnv(gym.Env):
    """
    Gymnasium env for one signalised junction driven by real traffic.

    Parameters
    ----------
    scenario : scenario name ("train", "bctyc_08", "long_day", ...) or Scenario
    seed     : jitter / SUMO seed (for "train", seeds the hour + jitter sampler)
    reward   : "wait" (-Δ accumulated waiting time), "pressure" (-|pressure|), "queue" (-queue)
    obs      : "vector", "presslight" or "frap"
    tripinfo : write SUMO tripinfo (+ emissions) to this path; None disables it (faster training)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Union[str, Scenario] = "train",
        seed: int = 0,
        reward: str = "wait",
        obs: str = "vector",
        tripinfo: Optional[str] = None,
        max_green: int = MAX_GREEN,
    ) -> None:
        self.scenario_arg = scenario
        self.reward_kind = reward
        self.obs_fn = OBS_FNS[obs]
        self.tripinfo = tripinfo
        self.max_green = max_green
        self._rng = random.Random(seed)
        self._seed = seed
        self._sumo: Optional[_CountingSumo] = None
        self.scenario: Optional[Scenario] = None
        # called as on_episode_end(env) after the episode's simulation has closed
        # (lets training run validation sims - libsumo allows one simulation per process)
        self.on_episode_end: Optional[Callable[["TscEnv"], None]] = None

        self.action_space = gym.spaces.Discrete(len(PHASE_NAMES))
        dim = {"vector": 26, "presslight": 8 * SEGMENTS + 16, "frap": 16}[obs]
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (dim,), dtype=np.float32)

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def _pick_scenario(self) -> Scenario:
        if isinstance(self.scenario_arg, Scenario):
            return self.scenario_arg
        if self.scenario_arg == "train":
            # new hour + jitter each episode; seeds 5..9 so evaluation seeds stay unseen
            return scenarios.get("train", seed=self._rng.choice([0, 5, 6, 7, 8, 9]), rng=self._rng)
        return scenarios.get(self.scenario_arg, seed=self._seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.close()
        self.scenario = self._pick_scenario()
        extra = ["--no-step-log", "--duration-log.disable"]
        if self.tripinfo:
            os.makedirs(os.path.dirname(os.path.abspath(self.tripinfo)), exist_ok=True)
            extra += ["--tripinfo-output", self.tripinfo, "--tripinfo-output.write-unfinished",
                      "--device.emissions.probability", "1"]
        self._sumo = _CountingSumo(
            net_file=self.scenario.net,
            route_file=self.scenario.route,
            num_seconds=self.scenario.seconds,
            delta_time=DELTA_TIME,
            yellow_time=YELLOW_TIME,
            min_green=MIN_GREEN,
            max_green=self.max_green,
            single_agent=True,
            reward_fn=lambda ts: 0.0,
            observation_class=_NullObservation,
            add_system_info=False,
            add_per_agent_info=False,
            sumo_seed=self.scenario.seed,
            sumo_warnings=False,
            additional_sumo_cmd=" ".join(extra),
        )
        self._sumo.reset(seed=self.scenario.seed)
        self._sumo.arrived_total = 0
        self._init_topology()
        self.step_log: List[Dict] = []
        self.forced_switches = 0
        self._switched = False
        self._prev_wait = 0.0
        self.episode_reward = 0.0
        self.features_now = self._features()
        return self.obs_fn(self.features_now), {"scenario": self.scenario.name}

    def close(self) -> None:
        if self._sumo is not None:
            self._sumo.close()
            self._sumo = None

    # ── Topology (lanes per phase, downstream lanes) ────────────────────────
    def _init_topology(self) -> None:
        ts = self._sumo.traffic_signals[TS_ID]
        traci = self._sumo.sumo
        self._ts = ts
        self.in_lanes: List[str] = list(ts.lanes)
        links = traci.trafficlight.getControlledLinks(TS_ID)
        lane_idx = {l: i for i, l in enumerate(self.in_lanes)}
        out_of: Dict[int, List[str]] = {i: [] for i in range(len(self.in_lanes))}
        for k, link in enumerate(links):
            if link:
                out_of[lane_idx[link[0][0]]].append(link[0][1])
        self._out_lanes = out_of
        self.phase_lanes = [
            sorted({lane_idx[links[k][0][0]] for k, c in enumerate(p.state) if c in "Gg" and links[k]})
            for p in ts.green_phases
        ]
        self.lane_approach = [APPROACHES.index(_EDGE_APPROACH[l.rsplit("_", 1)[0]]) for l in self.in_lanes]
        self._lane_len = {l: traci.lane.getLength(l) for l in self.in_lanes}

    # ── Features ─────────────────────────────────────────────────────────────
    def _features(self) -> JunctionFeatures:
        traci = self._sumo.sumo
        n = len(self.in_lanes)
        queue, count, wait, near = (np.zeros(n) for _ in range(4))
        segments = np.zeros((n, SEGMENTS))
        for i, lane in enumerate(self.in_lanes):
            queue[i] = traci.lane.getLastStepHaltingNumber(lane)
            count[i] = traci.lane.getLastStepVehicleNumber(lane)
            wait[i] = traci.lane.getWaitingTime(lane)
            length = self._lane_len[lane]
            for v in traci.lane.getLastStepVehicleIDs(lane):
                dist = length - traci.vehicle.getLanePosition(v)   # metres to stop line
                segments[i, min(int(dist / length * SEGMENTS), SEGMENTS - 1)] += 1
                if dist <= DETECTOR_M and traci.vehicle.getSpeed(v) > 0.5:
                    near[i] += 1
        out = np.array([
            np.mean([traci.lane.getLastStepVehicleNumber(o) for o in self._out_lanes[i]])
            for i in range(n)
        ])
        t = self._ts.time_since_last_phase_change
        return JunctionFeatures(
            time=float(self._sumo.sim_step),
            lane_queue=queue, lane_count=count, lane_wait=wait, lane_segments=segments,
            lane_moving_near_stop=near, out_count=out,
            lane_approach=self.lane_approach, phase_lanes=self.phase_lanes,
            current_phase=int(self._ts.green_phase),
            time_in_phase=float(t),
            # the very first phase starts green; every later one starts with a yellow
            green_time=max(t - YELLOW_TIME, 0.0) if self._switched else float(t),
            is_yellow=bool(self._ts.is_yellow),
        )

    # ── Step ─────────────────────────────────────────────────────────────────
    def _reward(self, f: JunctionFeatures) -> float:
        if self.reward_kind == "pressure":
            return -abs(float(f.lane_pressure.sum())) / 10.0
        if self.reward_kind == "queue":
            return -float(f.lane_queue.sum()) / 10.0
        total = float(sum(self._ts.get_accumulated_waiting_time_per_lane()))
        r = (self._prev_wait - total) / 100.0
        self._prev_wait = total
        return r

    def step(self, action: int):
        f = self.features_now
        action = int(action)
        forced = False
        if action == f.current_phase and not f.is_yellow and f.green_time >= self.max_green:
            # max-green reached: hand green to the busiest other phase
            q = f.phase_queue.astype(float)
            q[f.current_phase] = -1.0
            action = int(np.argmax(q))
            forced = True
            self.forced_switches += 1
        if action != f.current_phase and f.can_switch:
            self._switched = True
        _, _, _, truncated, _ = self._sumo.step(action)

        self.features_now = f = self._features()
        reward = self._reward(f)
        self.episode_reward += reward
        self.step_log.append({
            "time": f.time,
            "queue": float(f.lane_queue.sum()),
            "wait": float(f.lane_wait.sum()),
            "vehicles": int(self._sumo.sumo.vehicle.getIDCount()),
            "arrived": int(self._sumo.arrived_total),
            "phase": f.current_phase,
            "forced": forced,
        })
        info = {"scenario": self.scenario.name, "forced": forced}
        if truncated:
            self.close()   # flushes tripinfo
            info["tripinfo"] = self.tripinfo
            if self.on_episode_end:
                self.on_episode_end(self)
        return self.obs_fn(f), reward, False, bool(truncated), info

    def features(self) -> JunctionFeatures:
        return self.features_now


def tmp_tripinfo(tag: str) -> str:
    """Unique tripinfo path under results/tmp/."""
    return os.path.join(tsc.RESULTS_DIR, "tmp", f"{tag}_{uuid.uuid4().hex[:8]}.xml")
