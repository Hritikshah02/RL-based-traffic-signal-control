"""
rule_based.py - The five non-learning controllers.

  FixedController        equal greens, fixed 4-phase cycle (the original baseline)
  WebsterController      optimised fixed-time plan per hour (Webster's formula)
  ActuatedController     gap-out vehicle actuation (what most real signals run)
  SOTLController         self-organising traffic lights (Gershenson / Cools)
  MaxPressureController  throughput-optimal max-pressure (Varaiya)
"""

from typing import Dict, List

import numpy as np

from tsc import scenarios
from tsc.controllers.base import Controller, CyclicController, config
from tsc.env import DELTA_TIME, JunctionFeatures, TscEnv


class FixedController(CyclicController):
    name = "Fixed"

    def __init__(self, green: float = None) -> None:
        self.green = green or config("fixed")["green"]

    def act(self, f: JunctionFeatures) -> int:
        pending = self.waiting_for_switch(f)
        if pending is not None:
            return pending
        if f.green_time >= self.green:
            return self.advance(f)
        return self.target


def webster_plan(lane_flow: List[float], lane_is_left: List[bool], phase_lanes: List[List[int]],
                 cycle: List[int], cfg: Dict) -> Dict[str, object]:
    """
    Webster's optimal fixed-time plan.

    y_i = max over phase i's lanes of (flow / saturation flow)
    C0  = (1.5 L + 5) / (1 - Y),   L = lost time per phase × phases,  Y = Σ y_i
    g_i = (C0 - L) × y_i / Y
    Returns {"cycle": C0, "greens": [g per phase in `cycle` order], "Y": Y}.
    """
    sat = [cfg["sat_flow_left"] if left else cfg["sat_flow_through"] for left in lane_is_left]
    y = np.array([max(lane_flow[l] / sat[l] for l in phase_lanes[p]) for p in cycle])
    y = np.maximum(y, 1e-3)
    L = cfg["lost_time_per_phase"] * len(cycle)
    Y = float(y.sum())
    c0 = cfg["max_cycle"] if Y >= 0.95 else (1.5 * L + 5) / (1 - Y)
    c0 = float(np.clip(c0, cfg["min_cycle"], cfg["max_cycle"]))
    greens = (c0 - L) * y / Y
    return {"cycle": c0, "greens": greens.tolist(), "Y": Y}


class WebsterController(CyclicController):
    """Fixed-time, but each hour gets its own Webster plan from that hour's real demand."""

    name = "Webster"

    def __init__(self) -> None:
        self.cfg = config("webster")

    def reset(self, env: TscEnv) -> None:
        super().reset(env)
        demand = scenarios.demand()
        is_left = [lane.endswith("_1") for lane in env.in_lanes]   # lane 1 = left-turn lane
        self.plans = []
        for tag in env.scenario.hours:
            flow = [demand[tag].get(lane, 0) for lane in env.in_lanes]
            plan = webster_plan(flow, is_left, env.phase_lanes, self.cycle, self.cfg)
            # decisions happen every 5 s, so greens are rounded to 5 s
            plan["greens"] = [max(self.cfg["min_green"], DELTA_TIME * round(g / DELTA_TIME))
                              for g in plan["greens"]]
            self.plans.append(plan)

    def act(self, f: JunctionFeatures) -> int:
        pending = self.waiting_for_switch(f)
        if pending is not None:
            return pending
        plan = self.plans[min(int(f.time // 3600), len(self.plans) - 1)]
        if f.green_time >= plan["greens"][self.idx]:
            return self.advance(f)
        return self.target


class ActuatedController(CyclicController):
    """Hold green while vehicles keep arriving at the stop line; gap-out otherwise."""

    name = "Actuated"

    def __init__(self) -> None:
        cfg = config("actuated")
        self.min_green, self.max_green = cfg["min_green"], cfg["max_green"]

    def act(self, f: JunctionFeatures) -> int:
        pending = self.waiting_for_switch(f)
        if pending is not None:
            return pending
        if f.green_time < self.min_green:
            return self.target
        if f.green_time < self.max_green and f.lane_moving_near_stop[f.phase_lanes[self.target]].sum() > 0:
            return self.target   # extend: a vehicle is still crossing the detector
        return self.advance(f, skip_empty=True)


class SOTLController(CyclicController):
    """Switch once vehicles-on-red × seconds exceeds θ, unless a small green platoon is passing."""

    name = "SOTL"

    def __init__(self, theta: float = None) -> None:
        cfg = config("sotl")
        self.theta = theta or cfg["theta"]
        self.mu, self.min_green = cfg["mu"], cfg["min_green"]

    def reset(self, env: TscEnv) -> None:
        super().reset(env)
        self.kappa = 0.0

    def act(self, f: JunctionFeatures) -> int:
        pending = self.waiting_for_switch(f)
        if pending is not None:
            return pending
        green = f.phase_lanes[self.target]
        red = [l for l in range(len(f.lane_count)) if l not in green]
        self.kappa += f.lane_count[red].sum() * DELTA_TIME
        if f.green_time < self.min_green or self.kappa < self.theta:
            return self.target
        platoon = f.lane_moving_near_stop[green].sum()
        if 0 < platoon < self.mu:
            return self.target   # let a small platoon clear first
        self.kappa = 0.0
        return self.advance(f, skip_empty=True)


class MaxPressureController(Controller):
    """Give green to the phase with the largest Σ(incoming − downstream) vehicles."""

    name = "MaxPressure"

    def __init__(self) -> None:
        self.min_green = config("max_pressure")["min_green"]

    def act(self, f: JunctionFeatures) -> int:
        if f.is_yellow or f.green_time < self.min_green:
            return f.current_phase
        p = f.phase_pressure
        best = int(np.argmax(p))
        return f.current_phase if p[f.current_phase] >= p[best] else best
