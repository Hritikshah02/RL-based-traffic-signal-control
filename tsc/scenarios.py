"""
scenarios.py - Named traffic scenarios built from the real Hangzhou hours.

  <tag>      one real hour, e.g. "bctyc_07"
  train      a random hour from TRAIN_TAGS (re-sampled every episode)
  long_day   all 11 hours chained into one continuous 11-hour run
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from tsc import data_convert as dc

TRAIN_TAGS = ["bctyc_07", "knhz_07", "qcyn_07", "sbsx_07", "tmsxy_07", "bctyc_10"]
TEST_TAGS = ["bctyc_08", "knhz_08", "qcyn_08", "sbsx_08", "tmsxy_08"]
ALL_TAGS = TRAIN_TAGS + TEST_TAGS


@dataclass
class Scenario:
    name: str
    net: str
    route: str
    seconds: int
    hours: List[str]        # real hour tags in time order (used by Webster's per-hour plans)
    seed: int = 0


def ensure_built() -> None:
    """Convert raw data to SUMO files on first use."""
    if not os.path.exists(dc.DEMAND_FILE):
        dc.build_all()


def get(name: str, seed: int = 0, rng: Optional[random.Random] = None) -> Scenario:
    """Return the scenario `name` for a seed (seed 0 = exact recorded departures)."""
    ensure_built()
    if name == "train":
        name = (rng or random).choice(TRAIN_TAGS)
    if name == "long_day":
        hours = dc.chronological(ALL_TAGS)
    elif name in ALL_TAGS:
        hours = [name]
    else:
        raise ValueError(f"Unknown scenario '{name}'")
    return Scenario(
        name=name,
        net=dc.NET_FILE,
        route=dc.seeded_route(name, seed),
        seconds=dc.HOUR_SECONDS * len(hours),
        hours=hours,
        seed=seed,
    )


def expand(names: str) -> List[str]:
    """Expand a comma list that may contain the groups 'train', 'test', 'all'."""
    out: List[str] = []
    for n in names.split(","):
        n = n.strip()
        out += {"train": TRAIN_TAGS, "test": TEST_TAGS, "all": ALL_TAGS}.get(n, [n])
    return out


def demand() -> Dict[str, Dict[str, int]]:
    """Recorded vehicles per incoming lane, per hour tag."""
    ensure_built()
    with open(dc.DEMAND_FILE) as f:
        return json.load(f)
