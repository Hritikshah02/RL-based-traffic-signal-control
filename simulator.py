"""
simulator.py - Simulation runners that bridge the environment, agent, and UI.

Provides:
  - FixedTimerController : naïve round-robin green signal (baseline)
  - SimulationRunner     : drives one step at a time, returns data for the dashboard
"""

from typing import Dict

from traffic_env import TrafficEnv
from qlearning_agent import QLearningAgent
from utils import DIRECTIONS, DIR_INDEX


class FixedTimerController:
    """
    Baseline controller: cycles through N → S → E → W every *cycle_length*
    steps, regardless of queue lengths.
    """

    def __init__(self, cycle_length: int = 5) -> None:
        self.cycle_length = cycle_length
        self.timer = 0
        self.phase = 0  # index into DIRECTIONS

    def choose_action(self) -> int:
        """Return the current direction index; advance phase when timer expires."""
        action = self.phase
        self.timer += 1
        if self.timer >= self.cycle_length:
            self.timer = 0
            self.phase = (self.phase + 1) % len(DIRECTIONS)
        return action

    def reset(self) -> None:
        self.timer = 0
        self.phase = 0


class SimulationRunner:
    """
    Wraps an environment together with either an RL agent or a Fixed-Timer
    controller.  Call `run_step()` each tick to advance the simulation by one
    step and get back a data dict suitable for the dashboard.
    """

    def __init__(self, mode: str = "RL") -> None:
        """
        Parameters
        ----------
        mode : str  – "RL" or "Fixed"
        """
        self.mode = mode
        self.env = TrafficEnv()
        self.state = self.env.reset()

        if mode == "RL":
            self.agent = QLearningAgent.load()  # pre-trained, no exploration
            self.controller = None
        else:
            self.agent = None
            self.controller = FixedTimerController()

    # ── Single step ──────────────────────────────────────────────────────────
    def run_step(self, intensity: str = "Medium") -> Dict:
        """
        Advance the simulation by one step.

        Returns
        -------
        dict with keys used by MetricsTracker and the dashboard:
            step, queue_N … queue_W, total_queue, avg_wait, reward,
            passed, total_passed, current_green, queue_vehicles
        """
        # Pick action
        if self.mode == "RL":
            action = self.agent.choose_action(self.state, explore=False)
        else:
            action = self.controller.choose_action()

        # Step the environment
        next_state, reward, info = self.env.step(action, intensity)

        # Build a flat data dict
        ql = info["queue_lengths"]
        step_data = {
            "step":          info["step"],
            "queue_N":       ql["N"],
            "queue_S":       ql["S"],
            "queue_E":       ql["E"],
            "queue_W":       ql["W"],
            "total_queue":   sum(ql.values()),
            "avg_wait":      round(info["avg_wait"], 2),
            "reward":        round(reward, 2),
            "passed":        info["passed"],
            "total_passed":  info["total_passed"],
            "current_green": info["current_green"],
            # For the intersection visualization
            "queue_vehicles": {
                d: list(self.env.queues[d]) for d in DIRECTIONS
            },
            "passed_vehicles": info["passed_vehicles"],
        }

        self.state = next_state
        return step_data

    def reset(self) -> None:
        self.state = self.env.reset()
        if self.controller:
            self.controller.reset()
