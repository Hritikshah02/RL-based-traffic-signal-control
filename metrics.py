"""
metrics.py - Metrics tracking and comparison utilities.

Stores per-step data for both RL and Fixed-Timer simulation runs
and generates summary DataFrames for the Streamlit dashboard.
"""

from typing import Dict, List

import pandas as pd


class MetricsTracker:
    """
    Collects step-by-step metrics for RL and Fixed-Timer modes.

    Each step_dict should contain:
        step, queue_N, queue_S, queue_E, queue_W, total_queue,
        avg_wait, reward, passed, total_passed, current_green
    """

    def __init__(self) -> None:
        self.rl_history: List[Dict] = []
        self.fixed_history: List[Dict] = []

    def record(self, mode: str, step_data: Dict) -> None:
        """Append a step dict to the appropriate history list."""
        if mode == "RL":
            self.rl_history.append(step_data)
        else:
            self.fixed_history.append(step_data)

    def clear(self, mode: str) -> None:
        if mode == "RL":
            self.rl_history.clear()
        else:
            self.fixed_history.clear()

    def clear_all(self) -> None:
        self.rl_history.clear()
        self.fixed_history.clear()

    # ── DataFrames for plotting ──────────────────────────────────────────────
    def get_dataframe(self, mode: str) -> pd.DataFrame:
        """Return the history for *mode* as a pandas DataFrame."""
        data = self.rl_history if mode == "RL" else self.fixed_history
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    # ── Comparison table ─────────────────────────────────────────────────────
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Side-by-side comparison of RL vs Fixed-Timer performance.

        Returns a DataFrame with metrics as rows and modes as columns.
        """
        rows = []
        for mode, history in [("RL", self.rl_history), ("Fixed", self.fixed_history)]:
            if not history:
                continue
            df = pd.DataFrame(history)
            rows.append({
                "Mode": mode,
                "Avg Queue Length": round(df["total_queue"].mean(), 2),
                "Avg Wait Time": round(df["avg_wait"].mean(), 2),
                "Total Throughput": int(df["passed"].sum()),
                "Max Queue": int(df["total_queue"].max()),
                "Avg Reward": round(df["reward"].mean(), 2),
            })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("Mode")
