import json
import os
import xml.etree.ElementTree as ET

import tsc  # noqa: F401  (configures SUMO)
from tsc import data_convert as dc
from tsc import scenarios


def _vehicles(path):
    return ET.parse(path).getroot().findall("vehicle")


def test_each_hour_preserves_vehicle_count():
    scenarios.ensure_built()
    for tag, raw in dc.hour_tags().items():
        with open(os.path.join(raw, "flow.json")) as f:
            assert len(_vehicles(dc.route_path(tag))) == len(json.load(f)), tag


def test_long_day_is_all_hours_unique_and_sorted():
    scenarios.ensure_built()
    total = sum(len(_vehicles(dc.route_path(t))) for t in scenarios.ALL_TAGS)
    vs = _vehicles(dc.route_path("long_day"))
    departs = [float(v.get("depart")) for v in vs]
    assert len(vs) == total
    assert len({v.get("id") for v in vs}) == total
    assert departs == sorted(departs)
    assert departs[-1] < 11 * 3600


def test_jitter_deterministic_and_seed0_exact():
    vehicles = dc.load_vehicles(dc.hour_tags()["bctyc_07"])
    assert dc.jitter(vehicles, 0, 3600) == vehicles
    a, b = dc.jitter(vehicles, 3, 3600), dc.jitter(vehicles, 3, 3600)
    assert a == b and a != vehicles and len(a) == len(vehicles)
    assert all(0 <= d < 3600 for d, _, _ in a)


def test_splits_cover_all_eleven_hours():
    assert sorted(scenarios.ALL_TAGS) == sorted(dc.hour_tags())
    assert not set(scenarios.TRAIN_TAGS) & set(scenarios.TEST_TAGS)
    assert scenarios.get("long_day").seconds == 11 * 3600
