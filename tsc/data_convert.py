"""
data_convert.py - Convert the real Hangzhou CityFlow data into SUMO inputs.

Source: LibSignal `hangzhou_1x1_*` (camera-derived vehicle records, 2018-04-16).
All 11 hours share one abstract junction, so a single SUMO network is used and
each hour's flow.json becomes a SUMO route file.

Run:  python -m tsc.data_convert
"""

import glob
import json
import os
import random
import re
import shutil
from typing import Dict, List, Tuple

from tsc import DATA_RAW, DATA_SUMO

NET_SOURCE = "hangzhou_1x1_bc-tyc_18041610_1h"   # the folder that ships a SUMO net
NET_FILE = os.path.join(DATA_SUMO, "hangzhou_1x1.net.xml")
DEMAND_FILE = os.path.join(DATA_SUMO, "demand.json")
HOUR_SECONDS = 3600
JITTER_SECONDS = 30

_DIR_RE = re.compile(r"hangzhou_1x1_([a-z-]+)_180416(\d\d)_1h")

Vehicle = Tuple[float, str, str]   # (depart, in_edge, out_edge)


def hour_tags() -> Dict[str, str]:
    """Return {tag: raw_dir}, e.g. {"bctyc_07": ".../hangzhou_1x1_bc-tyc_18041607_1h"}."""
    tags = {}
    for path in sorted(glob.glob(os.path.join(DATA_RAW, "hangzhou_1x1_*"))):
        m = _DIR_RE.fullmatch(os.path.basename(path))
        if m:
            tags[f"{m.group(1).replace('-', '')}_{m.group(2)}"] = path
    return tags


def chronological(tags) -> List[str]:
    """Sort tags by hour, then junction (the order used for the long-day run)."""
    return sorted(tags, key=lambda t: (t.split("_")[1], t.split("_")[0]))


def load_vehicles(raw_dir: str) -> List[Vehicle]:
    """Read one flow.json; each record is a single vehicle (startTime == endTime)."""
    with open(os.path.join(raw_dir, "flow.json")) as f:
        flows = json.load(f)
    vehicles = []
    for rec in flows:
        t = rec["startTime"]
        while t <= rec["endTime"]:
            vehicles.append((float(t), rec["route"][0], rec["route"][-1]))
            t += max(rec.get("interval", 1), 1)
    vehicles.sort()
    return vehicles


def jitter(vehicles: List[Vehicle], seed: int, horizon: float) -> List[Vehicle]:
    """Seed 0 -> exact recorded departures; seed > 0 -> each depart shifted by ±30 s."""
    if seed == 0:
        return list(vehicles)
    rng = random.Random(seed)
    out = [
        (min(max(d + rng.uniform(-JITTER_SECONDS, JITTER_SECONDS), 0.0), horizon - 1), i, o)
        for d, i, o in vehicles
    ]
    out.sort()
    return out


def write_routes(path: str, vehicles: List[Vehicle], id_prefix: str) -> None:
    """Write a SUMO .rou.xml (vehicles must be sorted by depart)."""
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<routes>",
        '    <vType id="car" accel="2.0" decel="4.5" length="5.0" minGap="2.5" '
        'maxSpeed="11.111" sigma="0.5"/>',
    ]
    for n, (depart, in_edge, out_edge) in enumerate(vehicles):
        lines.append(
            f'    <vehicle id="{id_prefix}{n}" type="car" depart="{depart:.1f}" '
            f'departLane="best" departSpeed="max"><route edges="{in_edge} {out_edge}"/></vehicle>'
        )
    lines.append("</routes>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def lane_of_movement() -> Dict[Tuple[str, str], str]:
    """Map (in_edge, out_edge) -> incoming lane id using the SUMO net connections."""
    import sumolib

    net = sumolib.net.readNet(NET_FILE)
    mapping = {}
    for edge in net.getEdges():
        for out_edge, conns in edge.getOutgoing().items():
            mapping[(edge.getID(), out_edge.getID())] = conns[0].getFromLane().getID()
    return mapping


def build_all() -> None:
    """Write the net, one route file per real hour, long_day.rou.xml and demand.json."""
    os.makedirs(DATA_SUMO, exist_ok=True)
    shutil.copy(os.path.join(DATA_RAW, NET_SOURCE, f"{NET_SOURCE}.net.xml"), NET_FILE)
    tags = hour_tags()
    lane_map = lane_of_movement()
    demand: Dict[str, Dict[str, int]] = {}

    long_day: List[Vehicle] = []
    for k, tag in enumerate(chronological(tags)):
        vehicles = load_vehicles(tags[tag])
        write_routes(route_path(tag), vehicles, f"{tag}_")
        long_day += [(d + k * HOUR_SECONDS, i, o) for d, i, o in vehicles]
        counts: Dict[str, int] = {}
        for _, i, o in vehicles:
            lane = lane_map[(i, o)]
            counts[lane] = counts.get(lane, 0) + 1
        demand[tag] = counts

    write_routes(route_path("long_day"), long_day, "d")   # one numbering -> unique ids
    with open(DEMAND_FILE, "w") as f:
        json.dump(demand, f, indent=1, sort_keys=True)
    print(f"Built {len(tags)} hours + long_day into {DATA_SUMO}")


def route_path(tag: str, seed: int = 0) -> str:
    """Path of the route file for a tag (or 'long_day') and seed."""
    if seed == 0:
        return os.path.join(DATA_SUMO, f"{tag}.rou.xml")
    return os.path.join(DATA_SUMO, "seeded", f"{tag}_s{seed}.rou.xml")


def seeded_route(tag: str, seed: int) -> str:
    """Return (building if needed) the jittered route file for a tag and seed."""
    path = route_path(tag, seed)
    if seed == 0 or os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tags = hour_tags()
    if tag == "long_day":
        vehicles = []
        for k, t in enumerate(chronological(tags)):
            hour = jitter(load_vehicles(tags[t]), seed * 1000 + k, HOUR_SECONDS)
            vehicles += [(d + k * HOUR_SECONDS, i, o) for d, i, o in hour]
        vehicles.sort()
        write_routes(path, vehicles, "d")
    else:
        write_routes(path, jitter(load_vehicles(tags[tag]), seed, HOUR_SECONDS), f"{tag}_")
    return path


if __name__ == "__main__":
    build_all()
