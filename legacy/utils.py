"""
utils.py - Utility classes, constants, and helper functions for the
           RL Traffic Light Controller project.

Contains:
  - Direction and color constants
  - Vehicle dataclass
  - Queue-length bucketizer (for Q-table state encoding)
  - Traffic generation function with three intensity levels
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List

# ── Direction and indexing constants ─────────────────────────────────────────
DIRECTIONS = ["N", "S", "E", "W"]          # The four approaches
DIR_INDEX  = {d: i for i, d in enumerate(DIRECTIONS)}  # "N"->0, "S"->1, …
NUM_ACTIONS = len(DIRECTIONS)               # One action per direction

# ── Color palette (for the Streamlit / Plotly visualization) ─────────────────
COLORS = {
    "road":       "#3a3a3a",
    "background": "#2e7d32",
    "green":      "#00e676",
    "red":        "#ff1744",
    "yellow":     "#ffea00",
    "vehicle_N":  "#42a5f5",   # blue
    "vehicle_S":  "#ef5350",   # red
    "vehicle_E":  "#ffa726",   # orange
    "vehicle_W":  "#ab47bc",   # purple
    "intersection": "#616161",
}

# ── Bucket thresholds for state encoding ─────────────────────────────────────
#   0  (low)    : 0-3 vehicles
#   1  (medium) : 4-7 vehicles
#   2  (high)   : 8+  vehicles
BUCKET_THRESHOLDS = [4, 8]  # boundaries between low/med and med/high
NUM_BUCKETS = 3


def bucketize(queue_length: int) -> int:
    """Convert a raw queue length into a discrete bucket index (0, 1, or 2)."""
    if queue_length < BUCKET_THRESHOLDS[0]:
        return 0   # low
    elif queue_length < BUCKET_THRESHOLDS[1]:
        return 1   # medium
    else:
        return 2   # high


# ── Vehicle dataclass ────────────────────────────────────────────────────────
@dataclass
class Vehicle:
    """Represents a single vehicle waiting at (or passing through) the intersection."""
    direction: str                          # "N", "S", "E", or "W"
    turn: str = "straight"                  # "straight" or "right"
    wait_time: int = 0                      # steps spent waiting in queue
    queue_pos: int = 0                      # position in the queue (0 = front)
    color: str = field(default="", repr=False)

    def __post_init__(self):
        # Assign a direction-based color if not explicitly set
        if not self.color:
            self.color = COLORS.get(f"vehicle_{self.direction}", "#ffffff")


# ── Traffic intensity profiles ───────────────────────────────────────────────
# Maps intensity name -> (min_vehicles, max_vehicles) per direction per step
INTENSITY_PROFILES = {
    "Low":    (0, 1),
    "Medium": (0, 3),
    "Heavy":  (1, 5),
}


def generate_traffic(intensity: str = "Medium") -> Dict[str, List[Vehicle]]:
    """
    Generate new vehicles for one simulation step.

    Parameters
    ----------
    intensity : str
        One of "Low", "Medium", or "Heavy".

    Returns
    -------
    dict  {direction: [Vehicle, …]}
        Newly generated vehicles keyed by direction.
    """
    lo, hi = INTENSITY_PROFILES.get(intensity, INTENSITY_PROFILES["Medium"])
    new_vehicles: Dict[str, List[Vehicle]] = {}

    for d in DIRECTIONS:
        count = random.randint(lo, hi)
        vehicles = []
        for _ in range(count):
            # ~20 % of vehicles turn right, 80 % go straight
            turn = "right" if random.random() < 0.20 else "straight"
            vehicles.append(Vehicle(direction=d, turn=turn))
        new_vehicles[d] = vehicles

    return new_vehicles
