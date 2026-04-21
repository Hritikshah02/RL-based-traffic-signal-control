"""
traffic_env.py - Traffic Intersection Environment for Reinforcement Learning.

Models a 4-way intersection with vehicle queues.  The RL agent chooses which
direction gets a green signal; vehicles in that queue pass through while
others accumulate.

State space : (bucket_N, bucket_S, bucket_E, bucket_W, current_green_idx)
              3^4 × 4 = 324 discrete states
Action space: 0=N, 1=S, 2=E, 3=W  (set green for that direction)
"""

from typing import Dict, List, Optional, Tuple

from utils import (
    DIRECTIONS,
    DIR_INDEX,
    NUM_ACTIONS,
    Vehicle,
    bucketize,
    generate_traffic,
)

# ── Environment hyper-parameters ─────────────────────────────────────────────
MIN_GREEN_STEPS = 2     # minimum steps a green must stay before switching
MAX_PASS_PER_STEP = 2   # vehicles that can cross per green step
QUEUE_PENALTY_THRESHOLD = 10  # extra penalty when a queue exceeds this


class TrafficEnv:
    """
    Simulates a 4-way traffic intersection.

    Each step:
      1. Agent picks a direction to give green.
      2. New traffic is generated (based on intensity).
      3. Up to MAX_PASS_PER_STEP vehicles leave the green queue.
      4. Wait times for all remaining vehicles increase.
      5. A reward is computed.
    """

    def __init__(self) -> None:
        self.reset()

    # ── Reset / Init ─────────────────────────────────────────────────────────
    def reset(self) -> Tuple:
        """Reset all queues and return the initial state tuple."""
        # Queues: one list of Vehicle per direction
        self.queues: Dict[str, List[Vehicle]] = {d: [] for d in DIRECTIONS}
        self.current_green: str = "N"       # direction currently green
        self.green_timer: int = 0           # steps since last switch
        self.total_passed: int = 0          # cumulative throughput
        self.step_count: int = 0
        self.last_passed: List[Vehicle] = []  # vehicles that passed this step
        return self._get_state()

    # ── State encoding ───────────────────────────────────────────────────────
    def _get_state(self) -> Tuple[int, int, int, int, int]:
        """Return the discretized state tuple."""
        buckets = tuple(bucketize(len(self.queues[d])) for d in DIRECTIONS)
        return buckets + (DIR_INDEX[self.current_green],)

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(
        self,
        action: int,
        intensity: str = "Medium",
    ) -> Tuple[Tuple, float, dict]:
        """
        Execute one simulation step.

        Parameters
        ----------
        action    : int   (0-3) direction index to set green
        intensity : str   traffic generation intensity

        Returns
        -------
        (next_state, reward, info_dict)
        """
        self.step_count += 1
        desired_dir = DIRECTIONS[action]

        # ── 1. Handle green signal (enforce min-green) ───────────────────────
        if desired_dir == self.current_green:
            self.green_timer += 1
        else:
            if self.green_timer >= MIN_GREEN_STEPS:
                # Switch allowed
                self.current_green = desired_dir
                self.green_timer = 1
            else:
                # Too early to switch – keep current green
                self.green_timer += 1

        # ── 2. Generate new traffic ──────────────────────────────────────────
        new_vehicles = generate_traffic(intensity)
        for d in DIRECTIONS:
            for v in new_vehicles[d]:
                v.queue_pos = len(self.queues[d])
                self.queues[d].append(v)

        # ── 3. Let vehicles pass from the green queue ────────────────────────
        passed: List[Vehicle] = []
        green_q = self.queues[self.current_green]
        num_pass = min(MAX_PASS_PER_STEP, len(green_q))
        for _ in range(num_pass):
            v = green_q.pop(0)
            passed.append(v)
        self.last_passed = passed
        self.total_passed += len(passed)

        # Update queue positions after removal
        for d in DIRECTIONS:
            for idx, v in enumerate(self.queues[d]):
                v.queue_pos = idx

        # ── 4. Increment wait times ─────────────────────────────────────────
        for d in DIRECTIONS:
            for v in self.queues[d]:
                v.wait_time += 1

        # ── 5. Compute reward ────────────────────────────────────────────────
        total_queue = sum(len(self.queues[d]) for d in DIRECTIONS)
        overloaded = sum(
            1 for d in DIRECTIONS
            if len(self.queues[d]) > QUEUE_PENALTY_THRESHOLD
        )
        reward = -(total_queue) + (len(passed) * 2.0) - (10 * overloaded)

        # ── 6. Build info dict (used by simulator / dashboard) ───────────────
        info = {
            "queue_lengths": {d: len(self.queues[d]) for d in DIRECTIONS},
            "current_green": self.current_green,
            "passed": len(passed),
            "passed_vehicles": passed,
            "total_passed": self.total_passed,
            "avg_wait": self._avg_wait(),
            "step": self.step_count,
            # How many new vehicles arrived this step (for arrival animation)
            "new_counts": {d: len(new_vehicles[d]) for d in DIRECTIONS},
        }

        return self._get_state(), reward, info

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _avg_wait(self) -> float:
        """Average wait time across all queued vehicles."""
        all_vehicles = [v for d in DIRECTIONS for v in self.queues[d]]
        if not all_vehicles:
            return 0.0
        return sum(v.wait_time for v in all_vehicles) / len(all_vehicles)

    def get_queue_lengths(self) -> Dict[str, int]:
        return {d: len(self.queues[d]) for d in DIRECTIONS}
