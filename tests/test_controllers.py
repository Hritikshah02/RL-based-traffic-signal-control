import types

import numpy as np

from tsc.controllers import rule_based as rb
from tsc.env import DUAL_PHASES


def _start(ctrl):
    ctrl.reset(types.SimpleNamespace())
    return ctrl


def test_max_pressure_picks_highest_pressure_phase(make_features):
    count = [0, 0, 0, 9, 0, 0, 0, 9]          # E-left + W-left loaded -> phase 2 (EW-left)
    f = make_features(count=count, phase=0, green=20)
    assert rb.MaxPressureController().act(f) == 2


def test_max_pressure_respects_min_green_and_downstream(make_features):
    ctrl = rb.MaxPressureController()
    assert ctrl.act(make_features(count=[0, 0, 0, 9, 0, 0, 0, 9], phase=0, green=5)) == 0
    # downstream congestion cancels incoming demand
    f = make_features(count=[5, 0, 0, 9, 5, 0, 0, 9], out=[0, 0, 0, 9, 0, 0, 0, 9], phase=0, green=20)
    assert ctrl.act(f) == 1


def test_fixed_cycles_dual_phases_every_30s(make_features):
    ctrl = _start(rb.FixedController(green=30))
    seen = []
    phase, green = 0, 0.0
    for _ in range(60):
        a = ctrl.act(make_features(phase=phase, green=green))
        if a != phase:
            seen.append(a)
            phase, green = a, 0.0
        else:
            green += 5
    assert seen[:5] == DUAL_PHASES + [DUAL_PHASES[0]]


def test_actuated_extends_while_vehicle_near_stop_line(make_features):
    ctrl = _start(rb.ActuatedController())
    target = DUAL_PHASES[0]                   # NS-through, lanes 0 and 4
    near = np.zeros(8); near[0] = 1
    busy = [3] * 8
    assert ctrl.act(make_features(count=busy, near=near, phase=target, green=20)) == target
    assert ctrl.act(make_features(count=busy, phase=target, green=20)) == DUAL_PHASES[1]
    # max green overrides extension
    assert ctrl.act(make_features(count=busy, near=near, phase=DUAL_PHASES[1], green=50)) == DUAL_PHASES[2]


def test_webster_greens_fill_cycle_minus_lost_time():
    cfg = rb.config("webster")
    flow = [500, 100, 400, 150, 450, 120, 380, 90]
    is_left = [i % 2 == 1 for i in range(8)]
    plan = rb.webster_plan(flow, is_left, [[2, 6], [0, 4], [3, 7], [1, 5]] + [[0]] * 4, DUAL_PHASES, cfg)
    lost = cfg["lost_time_per_phase"] * 4
    assert abs(sum(plan["greens"]) + lost - plan["cycle"]) < 1e-6
    assert cfg["min_cycle"] <= plan["cycle"] <= cfg["max_cycle"]
    # the busiest phase (NS-through: 500/1800) gets the longest green
    assert np.argmax(plan["greens"]) == 0


def test_sotl_switches_after_theta(make_features):
    ctrl = _start(rb.SOTLController(theta=100))
    ctrl.reset(None)
    target = DUAL_PHASES[0]
    red_load = [0, 4, 4, 4, 0, 4, 4, 4]       # 24 vehicles on red lanes -> +120 per step
    assert ctrl.act(make_features(count=red_load, phase=target, green=20)) != target
