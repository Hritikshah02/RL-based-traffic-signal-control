"""
base.py - Controller interface shared by all 10 methods.
"""

import os
from typing import Dict, List, Optional

import yaml

from tsc import ROOT
from tsc.env import DUAL_PHASES, JunctionFeatures, TscEnv

_CONFIG_PATH = os.path.join(ROOT, "configs", "controllers.yaml")


def config(section: str) -> Dict:
    """Return one section of configs/controllers.yaml."""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f).get(section, {})


class Controller:
    """
    A traffic-signal controller.

    reset(env) is called after env.reset() (so env.scenario / env.phase_lanes exist);
    act(features) returns the green phase (0..7) wanted for the next 5 s.
    """

    name = "base"

    def reset(self, env: TscEnv) -> None:
        pass

    def act(self, f: JunctionFeatures) -> int:
        raise NotImplementedError


class CyclicController(Controller):
    """Base for controllers that step through the standard 4-phase dual cycle."""

    cycle: List[int] = DUAL_PHASES

    def reset(self, env: TscEnv) -> None:
        self.idx = 0

    @property
    def target(self) -> int:
        return self.cycle[self.idx]

    def advance(self, f: JunctionFeatures, skip_empty: bool = False) -> int:
        """Move to the next phase in the cycle (optionally skipping phases with no vehicles)."""
        for step in range(1, len(self.cycle) + 1):
            nxt = (self.idx + step) % len(self.cycle)
            if not skip_empty or f.lane_count[f.phase_lanes[self.cycle[nxt]]].sum() > 0:
                self.idx = nxt
                return self.target
        return self.target   # nothing waiting anywhere: keep the current green

    def waiting_for_switch(self, f: JunctionFeatures) -> Optional[int]:
        """If the junction isn't showing our target phase yet, keep requesting it."""
        return self.target if f.current_phase != self.target else None
