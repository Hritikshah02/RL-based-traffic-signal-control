"""
metrics.py - Turn a finished run (SUMO tripinfo + per-decision log) into numbers.

Per vehicle (tripinfo, incl. vehicles unfinished at the end of the run):
  delay   = timeLoss + departDelay   (time lost vs. free-flow, incl. waiting to enter)
  wait    = waitingTime              (seconds stopped)
  stops   = waitingCount
  co2     = CO2_abs (mg -> g)
Vehicles that never entered the network are counted in `not_inserted`.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List

import numpy as np
import pandas as pd


def read_tripinfo(path: str) -> pd.DataFrame:
    rows = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "tripinfo":
            continue
        em = el.find("emissions")
        rows.append({
            "depart": float(el.get("depart")),
            "arrival": float(el.get("arrival")),
            "duration": float(el.get("duration")),
            "delay": float(el.get("timeLoss")) + float(el.get("departDelay")),
            "wait": float(el.get("waitingTime")),
            "stops": int(el.get("waitingCount")),
            "co2_g": float(em.get("CO2_abs")) / 1000.0 if em is not None else np.nan,
        })
        el.clear()
    return pd.DataFrame(rows)


def summarize(trips: pd.DataFrame, steps: pd.DataFrame, demand: int, seconds: int) -> Dict[str, float]:
    """One summary row for a run."""
    finished = trips[trips["arrival"] >= 0]
    return {
        "demand": demand,
        "finished": int(len(finished)),
        "unfinished": int(demand - len(finished)),
        "not_inserted": int(demand - len(trips)),
        "throughput_vph": len(finished) / (seconds / 3600.0),
        "mean_delay": float(trips["delay"].mean()),
        "p95_delay": float(trips["delay"].quantile(0.95)),
        "mean_wait": float(trips["wait"].mean()),
        "p95_wait": float(trips["wait"].quantile(0.95)),
        "mean_travel_time": float(finished["duration"].mean()),
        "stops_per_veh": float(trips["stops"].mean()),
        "co2_g_per_veh": float(trips["co2_g"].mean()),
        "mean_queue": float(steps["queue"].mean()),
        "max_queue": float(steps["queue"].max()),
        "phase_switches": int((steps["phase"].diff().fillna(0) != 0).sum()),
        "forced_switches": int(steps["forced"].sum()),
    }


def per_minute(steps: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """Down-sample a run to one row per minute (for long-period time-series plots)."""
    s = steps.copy()
    s["minute"] = (s["time"] // 60).astype(int)
    out = s.groupby("minute").agg(queue=("queue", "mean"), wait=("wait", "mean"),
                                  vehicles=("vehicles", "mean"), arrived=("arrived", "last"))
    # mean delay of vehicles by the minute they departed
    t = trips.assign(minute=(trips["depart"] // 60).astype(int))
    out["delay"] = t.groupby("minute")["delay"].mean()
    out["throughput_vpm"] = out["arrived"].diff().fillna(out["arrived"])
    return out.reset_index()


SUMMARY_KEYS: List[str] = ["method", "scenario", "seed"]
