"""
app.py - Streamlit dashboard for the SUMO real-data benchmark.

Run with:  streamlit run app.py
(The original toy simulator is still available as the "Legacy Toy Simulator" page.)

Reads results/summary.csv, results/timeseries/ and models/*_train.csv produced by
    python -m tsc.train --method all
    python -m tsc.benchmark --methods all --scenarios all,long_day --seeds 5
"""

import glob
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")

RULE_BASED = ["Fixed", "Webster", "Actuated", "SOTL", "MaxPressure"]
LEARNING = ["QLearning", "D3QN", "PPO", "PressLight", "FRAP"]
TRAIN_TAGS = ["bctyc_07", "knhz_07", "qcyn_07", "sbsx_07", "tmsxy_07", "bctyc_10"]
TEST_TAGS = ["bctyc_08", "knhz_08", "qcyn_08", "sbsx_08", "tmsxy_08"]

METRICS = {
    "mean_delay": ("Mean delay per vehicle", "s"),
    "p95_delay": ("95th-percentile delay", "s"),
    "mean_wait": ("Mean stopped time per vehicle", "s"),
    "mean_queue": ("Mean queue (halted vehicles)", "veh"),
    "throughput_vph": ("Throughput", "veh/h"),
    "stops_per_veh": ("Stops per vehicle", "stops"),
    "co2_g_per_veh": ("CO₂ per vehicle", "g"),
    "unfinished": ("Vehicles not served by end of run", "veh"),
}
HIGHER_IS_BETTER = {"throughput_vph"}

# Categorical palette (validated reference palette, light / dark steps).
# 10 methods > 8 safe hues, so identity = hue (5 pairs) + line style (family):
# rule-based dashed, learning solid. Bars use the 2 family hues only.
HUES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

st.set_page_config(page_title="Traffic Signal Benchmark", page_icon="🚦", layout="wide")


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


MODE = theme_mode()


def method_color(m: str) -> str:
    group = RULE_BASED if m in RULE_BASED else LEARNING
    return HUES[MODE][group.index(m)]


def family_color(m: str) -> str:
    return HUES[MODE][0] if m in RULE_BASED else HUES[MODE][1]


def dash(m: str) -> str:
    return "dash" if m in RULE_BASED else "solid"


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig


@st.cache_data
def load_summary(mtime: float) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, "summary.csv"))


@st.cache_data
def load_timeseries(scenario: str, mtime: float) -> pd.DataFrame:
    frames = []
    for path in glob.glob(os.path.join(RESULTS, "timeseries", scenario, "*.csv")):
        method, seed = os.path.basename(path)[:-4].rsplit("_s", 1)
        frames.append(pd.read_csv(path).assign(method=method, seed=int(seed)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


# ── Header ───────────────────────────────────────────────────────────────────
st.title("🚦 Traffic Signal Control on Real Traffic")
st.caption(
    "10 controllers on one signalised junction in **SUMO**, driven by camera-recorded vehicles from "
    "**5 Hangzhou junctions (11 real hours, 2018-04-16)**. Learning methods trained on the 07:00 hours; "
    "the 08:00 hours are held out. Seeds 1–4 jitter each departure by ±30 s."
)

summary_path = os.path.join(RESULTS, "summary.csv")
if not os.path.exists(summary_path):
    st.warning("No results yet. Run `python -m tsc.benchmark --methods all --scenarios all,long_day --seeds 5`.")
    st.stop()
df = load_summary(_mtime(summary_path))

# ── Filters (one row) ────────────────────────────────────────────────────────
VIEWS = {
    "Held-out test hours (08:00)": TEST_TAGS,
    "Training hours (07:00 + bctyc 10:00)": TRAIN_TAGS,
    "Long day (11 h continuous)": ["long_day"],
}
c1, c2, c3 = st.columns([2, 2, 3])
view = c1.selectbox("Traffic", list(VIEWS) + [f"Single hour: {t}" for t in TRAIN_TAGS + TEST_TAGS])
metric = c2.selectbox("Metric", list(METRICS), format_func=lambda k: METRICS[k][0])
available = [m for m in RULE_BASED + LEARNING if m in set(df["method"])]
methods = c3.multiselect("Methods", available, default=available)
tags = VIEWS.get(view, [view.split(": ")[-1]])
label, unit = METRICS[metric]

sel = df[df["scenario"].isin(tags) & df["method"].isin(methods)]
if sel.empty:
    st.info("No runs for this selection yet.")
    st.stop()

# per method: average over hours within a seed, then mean ± std across seeds
per_seed = sel.groupby(["method", "seed"])[list(METRICS)].mean().reset_index()
agg = per_seed.groupby("method")[list(METRICS)].agg(["mean", "std"])
ranked = agg[(metric, "mean")].sort_values(ascending=metric not in HIGHER_IS_BETTER)

# ── Headline tiles ───────────────────────────────────────────────────────────
best = ranked.index[0]
t1, t2, t3, t4 = st.columns(4)
t1.metric("Best method", best, help=f"Lowest {label.lower()} on this selection")
t2.metric(f"Best {label.lower()}", f"{ranked.iloc[0]:.1f} {unit}")
if "Fixed" in ranked.index and metric not in HIGHER_IS_BETTER and ranked["Fixed"] > 0:
    t3.metric("vs. Fixed-time", f"{100 * (ranked.iloc[0] - ranked['Fixed']) / ranked['Fixed']:+.0f}%",
              delta_color="inverse", delta=f"{ranked.iloc[0] - ranked['Fixed']:+.1f} {unit}")
t4.metric("Vehicles per run", f"{int(sel.groupby(['method', 'seed'])['demand'].sum().mean()):,}",
          help=f"{len(tags)} hour(s) × {sel['seed'].nunique()} seed(s)")

# ── Leaderboard ──────────────────────────────────────────────────────────────
st.subheader(f"Leaderboard: {label.lower()}")
order = list(ranked.index)[::-1]   # best at the top of a horizontal bar chart
fig = go.Figure()
for fam, members in (("Rule-based", RULE_BASED), ("Learning", LEARNING)):
    ms = [m for m in order if m in members]
    if not ms:
        continue
    fig.add_bar(
        y=ms, x=[agg.loc[m, (metric, "mean")] for m in ms], orientation="h", name=fam,
        marker=dict(color=family_color(ms[0]), cornerradius=4),
        error_x=dict(type="data", array=[agg.loc[m, (metric, "std")] for m in ms], thickness=1.5, width=4),
        hovertemplate="%{y}: %{x:.1f} " + unit + "<extra>" + fam + "</extra>",
    )
# value labels sit just past each error bar so they never collide with it
ends = [agg.loc[m, (metric, "mean")] + (agg.loc[m, (metric, "std")] if pd.notna(agg.loc[m, (metric, "std")]) else 0)
        for m in order]
fig.add_scatter(y=order, x=ends, mode="text", text=[f"  {agg.loc[m, (metric, 'mean')]:.1f}" for m in order],
                textposition="middle right", showlegend=False, hoverinfo="skip")
fig.update_xaxes(range=[0, max(ends) * 1.12])
fig.update_layout(barmode="overlay", bargap=0.35, yaxis=dict(categoryorder="array", categoryarray=order))
fig.update_xaxes(title=f"{label} ({unit}), mean ± std over seeds")
st.plotly_chart(style(fig, 60 + 38 * len(order)), width="stretch")

with st.expander("Table view: all metrics"):
    table = agg.xs("mean", axis=1, level=1).loc[list(ranked.index)].round(1)
    table.insert(0, "family", ["Rule-based" if m in RULE_BASED else "Learning" for m in table.index])
    st.dataframe(table, width="stretch")

# ── Per-hour heatmap ─────────────────────────────────────────────────────────
hourly = df[df["method"].isin(methods) & df["scenario"].isin(TRAIN_TAGS + TEST_TAGS)]
if not hourly.empty:
    st.subheader(f"Every real hour: {label.lower()}")
    st.caption("Columns left of the gap are training hours; right of it are the held-out test hours.")
    pivot = hourly.groupby(["method", "scenario"])[metric].mean().unstack()
    cols = [t for t in TRAIN_TAGS + TEST_TAGS if t in pivot.columns]
    rows = [m for m in ranked.index if m in pivot.index] + [m for m in pivot.index if m not in ranked.index]
    pivot = pivot.loc[rows, cols]
    scale = SEQ_BLUE if metric not in HIGHER_IS_BETTER else SEQ_BLUE[::-1]
    heat = go.Figure(go.Heatmap(
        z=pivot.values, x=cols, y=rows, colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)],
        text=pivot.round(0).values, texttemplate="%{text:.0f}", xgap=2, ygap=2,
        hovertemplate="%{y} on %{x}: %{z:.1f} " + unit + "<extra></extra>",
        colorbar=dict(title=unit, thickness=10),
    ))
    if set(TRAIN_TAGS) & set(cols) and set(TEST_TAGS) & set(cols):
        heat.add_vline(x=len([c for c in cols if c in TRAIN_TAGS]) - 0.5, line_width=3)
    heat.update_yaxes(autorange="reversed")
    st.plotly_chart(style(heat, 70 + 34 * len(rows)), width="stretch")

# ── Long-day time series ─────────────────────────────────────────────────────
ts = load_timeseries("long_day", _mtime(os.path.join(RESULTS, "timeseries", "long_day")))
if not ts.empty:
    st.subheader("Long period: 11 hours of real traffic, back to back")
    s1, s2 = st.columns([2, 3])
    series_names = {"queue": "Queue (veh)", "delay": "Delay of departing vehicles (s)",
                    "throughput_vpm": "Throughput (veh/min)"}
    series = s1.radio("Series", list(series_names), horizontal=True, format_func=series_names.get)
    default = [m for m in ["Webster", "MaxPressure", "PressLight", "D3QN", "FRAP"] if m in methods]
    shown = s2.multiselect("Lines", methods, default=default or methods[:5])
    lines = (ts[ts["method"].isin(shown)].groupby(["method", "minute"])[series].mean()
             .groupby(level=0).transform(lambda s: s.rolling(10, min_periods=1).mean()).reset_index())
    line_fig = go.Figure()
    for m in [m for m in RULE_BASED + LEARNING if m in shown]:
        d = lines[lines["method"] == m]
        line_fig.add_scatter(x=d["minute"] / 60, y=d[series], name=m, mode="lines",
                             line=dict(color=method_color(m), width=2, dash=dash(m)),
                             hovertemplate=m + ": %{y:.1f}<extra></extra>")
    for h in range(1, 11):
        line_fig.add_vline(x=h, line_width=1, line_dash="dot", opacity=0.3)
    line_fig.update_layout(hovermode="x unified")
    line_fig.update_yaxes(title=series_names[series])
    line_fig.update_xaxes(title="Hour of the run (each hour is a different real recording; 10-min rolling mean)")
    st.plotly_chart(style(line_fig, 420), width="stretch")

# ── Learning curves ──────────────────────────────────────────────────────────
curves = {}
for m in LEARNING:
    path = os.path.join(MODELS, f"{m}_train.csv")
    c = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if "val_queue" in c and c["val_queue"].notna().any():   # still training: no validation yet
        curves[m] = c.dropna(subset=["val_queue"])
if curves:
    st.subheader("Training: validation queue during learning")
    st.caption("Greedy policy scored every 10 episodes on two training hours; the best checkpoint is kept.")
    cf = go.Figure()
    for m, c in curves.items():
        cf.add_scatter(x=c["episode"] + 1, y=c["val_queue"], name=m, mode="lines+markers",
                       line=dict(color=method_color(m), width=2), marker=dict(size=8),
                       hovertemplate=m + " ep %{x}: %{y:.1f} veh<extra></extra>")
    cf.update_layout(hovermode="x unified")
    cf.update_xaxes(title="Training episode (1 episode = 1 simulated hour)")
    cf.update_yaxes(title="Mean queue (veh)", type="log")
    st.plotly_chart(style(cf, 360), width="stretch")

# ── Reward ablation ──────────────────────────────────────────────────────────
ablation_path = os.path.join(RESULTS, "ablation_wait_reward.csv")
if os.path.exists(ablation_path):
    st.subheader("Ablation: what the reward function changed")
    st.caption("First training round: Q-learning, D3QN and PPO used the waiting-time-difference reward and "
               "validated on 2 moderate hours. Final round: queue-length reward + the heaviest training hour "
               "added to validation (PressLight keeps its pressure reward in both). Q-learning also moved to dual-phase "
               "queue states and lost a longest-queue fallback that had flattered its round-1 score. "
               "Mean delay (s), mean over seeds.")
    abl = pd.read_csv(ablation_path)
    groups = {"Test hours": TEST_TAGS, "Long day": ["long_day"], "Training hours": TRAIN_TAGS}
    rows = []
    for m in [m for m in LEARNING if m in set(abl["method"])]:
        row = {"method": m}
        for g, gtags in groups.items():
            before = abl[(abl["method"] == m) & abl["scenario"].isin(gtags)]["mean_delay"].mean()
            after = df[(df["method"] == m) & df["scenario"].isin(gtags)]["mean_delay"].mean()
            row[f"{g}: round 1"] = round(before, 1)
            row[f"{g}: final"] = round(after, 1)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("method"), width="stretch")

st.caption("Delay = SUMO timeLoss + insertion delay, including vehicles still in the network when the run ends. "
           "CO₂ from SUMO's HBEFA emission model.")
