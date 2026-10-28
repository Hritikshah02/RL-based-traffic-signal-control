"""
app.py - Streamlit Dashboard for the RL Traffic Light Controller.

Run with:  streamlit run app.py

Layout:
  Sidebar  – mode toggle, intensity, speed, controls, train button
  Main     – intersection visualization | live metrics
             tabbed charts (queue, wait, throughput, reward)
             comparison table
"""

import time
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Ensure project root is on the path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils import DIRECTIONS, COLORS
from simulator import SimulationRunner
from metrics import MetricsTracker
from qlearning_agent import train_agent, Q_TABLE_PATH

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RL Traffic Light Controller",
    page_icon="🚦",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ──────────────────────────────────────────────────────────────────────────────
defaults = {
    "running": False,
    "mode": "RL",
    "intensity": "Medium",
    "speed": 0.3,
    "metrics": MetricsTracker(),
    "runner_rl": None,
    "runner_fixed": None,
    "step_data": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def _get_runner(mode: str) -> SimulationRunner:
    """Return (and lazily create) the SimulationRunner for the given mode."""
    key = "runner_rl" if mode == "RL" else "runner_fixed"
    if st.session_state[key] is None:
        st.session_state[key] = SimulationRunner(mode)
    return st.session_state[key]


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚦 Traffic Controller")
    st.markdown("---")

    mode = st.radio("Controller Mode", ["RL (Q-Learning)", "Fixed Timer"], index=0)
    st.session_state.mode = "RL" if "RL" in mode else "Fixed"

    st.session_state.intensity = st.select_slider(
        "Traffic Intensity",
        options=["Low", "Medium", "Heavy"],
        value=st.session_state.intensity,
    )

    st.session_state.speed = st.slider(
        "Step Delay (seconds)", 0.05, 1.0,
        value=st.session_state.speed, step=0.05,
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Start", use_container_width=True):
            st.session_state.running = True
    with col2:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.running = False
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.running = False
            st.session_state.runner_rl = None
            st.session_state.runner_fixed = None
            st.session_state.metrics.clear_all()
            st.session_state.step_data = None
            st.rerun()

    st.markdown("---")
    st.subheader("Training")
    q_exists = os.path.exists(Q_TABLE_PATH)
    st.caption(f"Q-table: {'✅ found' if q_exists else '❌ not found'}")
    if st.button("🧠 Train Agent (500 eps)", use_container_width=True):
        with st.spinner("Training Q-learning agent …"):
            train_agent(episodes=500, steps_per_episode=200, verbose=False)
        st.success("Training complete! Q-table saved.")
        # Reload runner so it picks up the new table
        st.session_state.runner_rl = None
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Intersection Visualization (Plotly)
# ──────────────────────────────────────────────────────────────────────────────
# Coordinate system: intersection center at (0,0).
# Roads are 3.0 units wide (1.5 per lane), view extends to ±10.

ROAD_HW  = 1.5   # road half-width (2 lanes each 1.5 wide → total 3.0)
LANE_W   = 1.5   # single lane width
FIELD    = 10.0   # visible extent
SIDEWALK = 0.35   # sidewalk strip width

# Car dimensions (rectangle) – vertical cars are tall+narrow, horizontal wide+short
CAR_L = 0.70   # length along travel direction
CAR_W = 0.55   # width across travel direction
CAR_GAP = 0.18 # gap between bumpers

# How many cars to draw per lane (cap so we don't kill performance)
MAX_CARS_DRAWN = 18

# ── Per-direction config ─────────────────────────────────────────────────────
# "lane_center" = cross-axis position of the incoming lane center
# "start"       = along-axis position of the first car (closest to intersection)
# "step"        = increment per car (positive = away from center)
# "orient"      = "v" (vertical) or "h" (horizontal)
DIR_CFG = {
    "N": {"orient": "v", "lane_center":  0.75, "start": -(ROAD_HW + 0.6), "step": -(CAR_L + CAR_GAP)},
    "S": {"orient": "v", "lane_center": -0.75, "start":  (ROAD_HW + 0.6), "step":  (CAR_L + CAR_GAP)},
    "E": {"orient": "h", "lane_center": -0.75, "start":  (ROAD_HW + 0.6), "step":  (CAR_L + CAR_GAP)},
    "W": {"orient": "h", "lane_center":  0.75, "start": -(ROAD_HW + 0.6), "step": -(CAR_L + CAR_GAP)},
}

# Traffic-light pole positions and label positions
LIGHT_POS = {
    "N": ( ROAD_HW + 0.55, -(ROAD_HW + 0.15)),
    "S": (-(ROAD_HW + 0.55),  (ROAD_HW + 0.15)),
    "E": ( (ROAD_HW + 0.15),  (ROAD_HW + 0.55)),
    "W": (-(ROAD_HW + 0.15), -(ROAD_HW + 0.55)),
}
LABEL_POS = {
    "N": ( ROAD_HW + 1.8, -5.5),
    "S": (-(ROAD_HW + 1.8),  5.5),
    "E": ( 5.5,  (ROAD_HW + 1.8)),
    "W": (-5.5, -(ROAD_HW + 1.8)),
}

# Windshield offset inside the car (gives it a "front")
WINDSHIELD = {
    "N": (0, +0.18),
    "S": (0, -0.18),
    "E": (-0.18, 0),
    "W": (+0.18, 0),
}

# Car body colors per direction (softer palette)
CAR_COLORS = {
    "N": "#64b5f6",  # blue
    "S": "#ef5350",  # red
    "E": "#ffb74d",  # orange
    "W": "#ba68c8",  # purple
}
# Slightly darker shade for roof / windshield accent
CAR_ROOF = {
    "N": "#1e88e5",
    "S": "#c62828",
    "E": "#ef6c00",
    "W": "#7b1fa2",
}


def _add_car(fig, cx, cy, direction, color, roof_color):
    """Draw one car as a rounded rectangle + a small windshield rectangle."""
    cfg = DIR_CFG[direction]
    if cfg["orient"] == "v":
        hw, hh = CAR_W / 2, CAR_L / 2   # half-width, half-height
    else:
        hw, hh = CAR_L / 2, CAR_W / 2

    # Car body
    fig.add_shape(
        type="rect",
        x0=cx - hw, y0=cy - hh, x1=cx + hw, y1=cy + hh,
        fillcolor=color,
        line=dict(color="rgba(0,0,0,0.35)", width=1),
        layer="above",
    )
    # Windshield accent (small rectangle inside, offset toward the front)
    wx, wy = WINDSHIELD[direction]
    if cfg["orient"] == "v":
        fig.add_shape(
            type="rect",
            x0=cx - hw * 0.55 + wx, y0=cy - hh * 0.30 + wy,
            x1=cx + hw * 0.55 + wx, y1=cy + hh * 0.30 + wy,
            fillcolor=roof_color, line_width=0, layer="above",
        )
    else:
        fig.add_shape(
            type="rect",
            x0=cx - hw * 0.30 + wx, y0=cy - hh * 0.55 + wy,
            x1=cx + hw * 0.30 + wx, y1=cy + hh * 0.55 + wy,
            fillcolor=roof_color, line_width=0, layer="above",
        )


def build_intersection_figure(step_data) -> go.Figure:
    """Create a Plotly figure showing the intersection, lights, and cars."""
    fig = go.Figure()

    # ── 1. Grass background ──────────────────────────────────────────────────
    fig.add_shape(type="rect", x0=-FIELD, y0=-FIELD, x1=FIELD, y1=FIELD,
                  fillcolor="#4a7c59", line_width=0, layer="below")

    # ── 2. Sidewalks (slightly lighter strips bordering roads) ───────────────
    sw = SIDEWALK
    for rect in [
        # Vertical road sidewalks
        dict(x0=-(ROAD_HW + sw), y0=-FIELD, x1=-ROAD_HW, y1=FIELD),
        dict(x0=ROAD_HW, y0=-FIELD, x1=ROAD_HW + sw, y1=FIELD),
        # Horizontal road sidewalks
        dict(x0=-FIELD, y0=-(ROAD_HW + sw), x1=FIELD, y1=-ROAD_HW),
        dict(x0=-FIELD, y0=ROAD_HW, x1=FIELD, y1=ROAD_HW + sw),
    ]:
        fig.add_shape(type="rect", **rect,
                      fillcolor="#b0bec5", line_width=0, layer="below")

    # ── 3. Roads (dark asphalt) ──────────────────────────────────────────────
    fig.add_shape(type="rect", x0=-ROAD_HW, y0=-FIELD, x1=ROAD_HW, y1=FIELD,
                  fillcolor="#37474f", line_width=0)
    fig.add_shape(type="rect", x0=-FIELD, y0=-ROAD_HW, x1=FIELD, y1=ROAD_HW,
                  fillcolor="#37474f", line_width=0)

    # ── 4. Intersection box (slightly lighter) ──────────────────────────────
    fig.add_shape(type="rect", x0=-ROAD_HW, y0=-ROAD_HW, x1=ROAD_HW, y1=ROAD_HW,
                  fillcolor="#455a64", line_width=0)

    # ── 5. Lane divider lines (dashed yellow center, white edge) ─────────────
    # Center dividers
    for seg in [
        dict(x0=0, y0=-FIELD, x1=0, y1=-ROAD_HW),
        dict(x0=0, y0=ROAD_HW, x1=0, y1=FIELD),
        dict(x0=-FIELD, y0=0, x1=-ROAD_HW, y1=0),
        dict(x0=ROAD_HW, y0=0, x1=FIELD, y1=0),
    ]:
        fig.add_shape(type="line", **seg,
                      line=dict(color="#fdd835", width=2, dash="dash"))

    # Road edge lines (solid white)
    for seg in [
        # Vertical road edges
        dict(x0=-ROAD_HW, y0=-FIELD, x1=-ROAD_HW, y1=-ROAD_HW),
        dict(x0=-ROAD_HW, y0=ROAD_HW, x1=-ROAD_HW, y1=FIELD),
        dict(x0=ROAD_HW, y0=-FIELD, x1=ROAD_HW, y1=-ROAD_HW),
        dict(x0=ROAD_HW, y0=ROAD_HW, x1=ROAD_HW, y1=FIELD),
        # Horizontal road edges
        dict(x0=-FIELD, y0=-ROAD_HW, x1=-ROAD_HW, y1=-ROAD_HW),
        dict(x0=ROAD_HW, y0=-ROAD_HW, x1=FIELD, y1=-ROAD_HW),
        dict(x0=-FIELD, y0=ROAD_HW, x1=-ROAD_HW, y1=ROAD_HW),
        dict(x0=ROAD_HW, y0=ROAD_HW, x1=FIELD, y1=ROAD_HW),
    ]:
        fig.add_shape(type="line", **seg,
                      line=dict(color="rgba(255,255,255,0.5)", width=1))

    # ── 6. Crosswalk stripes ─────────────────────────────────────────────────
    stripe_w = 0.20
    stripe_gap = 0.35
    for i in range(6):
        offset = -ROAD_HW + 0.25 + i * stripe_gap
        # North crosswalk
        fig.add_shape(type="rect",
                      x0=offset, y0=-(ROAD_HW + 0.05),
                      x1=offset + stripe_w, y1=-ROAD_HW + 0.35,
                      fillcolor="rgba(255,255,255,0.7)", line_width=0)
        # South crosswalk
        fig.add_shape(type="rect",
                      x0=offset, y0=ROAD_HW - 0.35,
                      x1=offset + stripe_w, y1=ROAD_HW + 0.05,
                      fillcolor="rgba(255,255,255,0.7)", line_width=0)
        # East crosswalk
        fig.add_shape(type="rect",
                      x0=ROAD_HW - 0.35, y0=offset,
                      x1=ROAD_HW + 0.05, y1=offset + stripe_w,
                      fillcolor="rgba(255,255,255,0.7)", line_width=0)
        # West crosswalk
        fig.add_shape(type="rect",
                      x0=-(ROAD_HW + 0.05), y0=offset,
                      x1=-ROAD_HW + 0.35, y1=offset + stripe_w,
                      fillcolor="rgba(255,255,255,0.7)", line_width=0)

    # ── 7. Stop lines (solid white bar at each approach) ─────────────────────
    stop_t = 0.12  # thickness
    fig.add_shape(type="rect", x0=0, y0=-(ROAD_HW), x1=ROAD_HW, y1=-(ROAD_HW) + stop_t,
                  fillcolor="white", line_width=0)  # N approach (right lane)
    fig.add_shape(type="rect", x0=-ROAD_HW, y0=ROAD_HW - stop_t, x1=0, y1=ROAD_HW,
                  fillcolor="white", line_width=0)  # S approach
    fig.add_shape(type="rect", x0=ROAD_HW - stop_t, y0=-ROAD_HW, x1=ROAD_HW, y1=0,
                  fillcolor="white", line_width=0)  # E approach
    fig.add_shape(type="rect", x0=-ROAD_HW, y0=0, x1=-(ROAD_HW) + stop_t, y1=ROAD_HW,
                  fillcolor="white", line_width=0)  # W approach

    if step_data is None:
        fig.update_layout(_base_layout())
        return fig

    current_green = step_data["current_green"]

    # ── 8. Traffic lights ────────────────────────────────────────────────────
    for d in DIRECTIONS:
        lx, ly = LIGHT_POS[d]
        is_green = (d == current_green)
        light_color = "#00e676" if is_green else "#ff1744"

        # Light housing (dark box)
        fig.add_shape(type="rect",
                      x0=lx - 0.32, y0=ly - 0.32, x1=lx + 0.32, y1=ly + 0.32,
                      fillcolor="#212121",
                      line=dict(color="#424242", width=2),
                      layer="above")
        # Glow behind the light
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="markers",
            marker=dict(size=28, color=light_color, opacity=0.25),
            hoverinfo="skip", showlegend=False,
        ))
        # Light circle
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="markers",
            marker=dict(size=18, color=light_color,
                        line=dict(width=2, color="white")),
            hoverinfo="text",
            hovertext=f"{d}: {'🟢 GREEN' if is_green else '🔴 RED'}",
            showlegend=False,
        ))

    # ── 9. Queued vehicles (car rectangles) ──────────────────────────────────
    for d in DIRECTIONS:
        vehicles = step_data["queue_vehicles"].get(d, [])
        if not vehicles:
            continue
        cfg = DIR_CFG[d]
        n_draw = min(len(vehicles), MAX_CARS_DRAWN)
        color = CAR_COLORS[d]
        roof  = CAR_ROOF[d]

        for i in range(n_draw):
            along = cfg["start"] + i * cfg["step"]
            cross = cfg["lane_center"]
            if cfg["orient"] == "v":
                _add_car(fig, cross, along, d, color, roof)
            else:
                _add_car(fig, along, cross, d, color, roof)

    # ── 10. Overflow indicator ("+" badge if more cars than drawn) ───────────
    # plus queue-count labels
    for d in DIRECTIONS:
        qlen = step_data[f"queue_{d}"]
        lx, ly = LABEL_POS[d]
        badge_color = "#00e676" if d == current_green else "#ff1744"
        fig.add_annotation(
            x=lx, y=ly,
            text=f"<b>{d}<br>{qlen}</b>",
            showarrow=False,
            font=dict(color="white", size=13, family="Arial Black"),
            bgcolor=badge_color,
            bordercolor="white",
            borderwidth=1,
            borderpad=5,
            opacity=0.9,
        )

    # ── 11. Direction arrows on road (subtle) ────────────────────────────────
    arrow_col = "rgba(255,255,255,0.18)"
    arrow_cfg = [
        # (x, y, symbol) – placed in the outgoing lane to show flow
        ("N", -0.75, ROAD_HW + 3.0, "triangle-up"),
        ("S",  0.75, -(ROAD_HW + 3.0), "triangle-down"),
        ("E",  0.75, (ROAD_HW + 3.0), "triangle-right"),   # wrong, let me fix
        ("W", -(ROAD_HW + 3.0), -0.75, "triangle-left"),
    ]
    # Outgoing-lane arrows (vehicles that already passed, going away)
    for d, sym in [("N", "triangle-up"), ("S", "triangle-down"),
                   ("E", "triangle-right"), ("W", "triangle-left")]:
        cfg = DIR_CFG[d]
        # Place arrow on the OPPOSITE lane (outgoing)
        if cfg["orient"] == "v":
            ax = -cfg["lane_center"]
            ay = -cfg["start"]  # opposite side of intersection
        else:
            ax = -cfg["start"]
            ay = -cfg["lane_center"]
        fig.add_trace(go.Scatter(
            x=[ax], y=[ay], mode="markers",
            marker=dict(size=14, symbol=sym, color=arrow_col),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(_base_layout())
    return fig


def _base_layout() -> dict:
    return dict(
        xaxis=dict(range=[-FIELD, FIELD], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-FIELD, FIELD], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", fixedrange=True),
        width=600, height=600,
        margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="#4a7c59",
        paper_bgcolor="#1e1e1e",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Charting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_queue_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for d in DIRECTIONS:
        fig.add_trace(go.Scatter(
            x=df["step"], y=df[f"queue_{d}"], mode="lines",
            name=d, line=dict(color=COLORS[f"vehicle_{d}"], width=2)))
    fig.update_layout(title="Queue Lengths", xaxis_title="Step",
                      yaxis_title="Vehicles", height=300,
                      margin=dict(l=40, r=10, t=40, b=40))
    return fig


def plot_wait_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["avg_wait"], mode="lines",
        line=dict(color="#ff9800", width=2), name="Avg Wait"))
    fig.update_layout(title="Average Wait Time", xaxis_title="Step",
                      yaxis_title="Steps Waiting", height=300,
                      margin=dict(l=40, r=10, t=40, b=40))
    return fig


def plot_throughput_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["total_passed"], mode="lines",
        line=dict(color="#4caf50", width=2), name="Cumulative"))
    fig.update_layout(title="Throughput (Cumulative)", xaxis_title="Step",
                      yaxis_title="Vehicles Passed", height=300,
                      margin=dict(l=40, r=10, t=40, b=40))
    return fig


def plot_reward_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["reward"], mode="lines",
        line=dict(color="#2196f3", width=2), name="Reward"))
    fig.update_layout(title="Reward per Step", xaxis_title="Step",
                      yaxis_title="Reward", height=300,
                      margin=dict(l=40, r=10, t=40, b=40))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main page
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 🚦 RL Traffic Light Controller")
st.caption(
    f"Mode: **{st.session_state.mode}** · "
    f"Intensity: **{st.session_state.intensity}** · "
    f"{'▶ Running' if st.session_state.running else '⏸ Paused'}"
)

# Two-column layout: intersection (left) + live metrics (right)
left_col, right_col = st.columns([3, 2], gap="large")

# Placeholders so we can update in the loop
with left_col:
    intersection_ph = st.empty()
with right_col:
    metrics_ph = st.empty()

# Tabs for historical charts
tab_queue, tab_wait, tab_throughput, tab_reward = st.tabs(
    ["📊 Queue Sizes", "⏱ Wait Time", "🚗 Throughput", "🎯 Reward"]
)
chart_phs = {
    "queue": tab_queue.empty(),
    "wait": tab_wait.empty(),
    "throughput": tab_throughput.empty(),
    "reward": tab_reward.empty(),
}

# Comparison table
st.markdown("### RL vs Fixed Timer Comparison")
comparison_ph = st.empty()

# ── Show current state (even if paused) ──────────────────────────────────────
fig0 = build_intersection_figure(st.session_state.step_data)
intersection_ph.plotly_chart(fig0, use_container_width=True, key="init_inter")

# Show any existing comparison data
comp_df = st.session_state.metrics.get_comparison_table()
if not comp_df.empty:
    comparison_ph.dataframe(comp_df, use_container_width=True)

# ── Simulation loop ──────────────────────────────────────────────────────────
if st.session_state.running:
    runner = _get_runner(st.session_state.mode)
    tracker: MetricsTracker = st.session_state.metrics

    for i in range(500):  # max steps per "run"; user can pause/reset anytime
        if not st.session_state.running:
            break

        step_data = runner.run_step(st.session_state.intensity)
        st.session_state.step_data = step_data
        tracker.record(st.session_state.mode, step_data)

        # ── Update intersection viz ──────────────────────────────────────────
        fig = build_intersection_figure(step_data)
        intersection_ph.plotly_chart(fig, use_container_width=True,
                                     key=f"inter_{i}")

        # ── Update live metrics panel ────────────────────────────────────────
        with metrics_ph.container():
            green_dir = step_data['current_green']
            signal_colors = {d: "🟢" if d == green_dir else "🔴" for d in DIRECTIONS}
            st.markdown(f"### Step {step_data['step']}")
            st.markdown(
                f"**Signals:**  "
                f"{signal_colors['N']} N  &nbsp; "
                f"{signal_colors['S']} S  &nbsp; "
                f"{signal_colors['E']} E  &nbsp; "
                f"{signal_colors['W']} W"
            )
            st.markdown("---")
            m1, m2 = st.columns(2)
            m1.metric("Total Queue", step_data["total_queue"])
            m2.metric("Avg Wait", f"{step_data['avg_wait']:.1f}")
            m3, m4 = st.columns(2)
            m3.metric("Passed / Step", step_data["passed"])
            m4.metric("Throughput", step_data["total_passed"])
            st.metric("Reward", f"{step_data['reward']:.2f}")
            st.markdown("---")
            st.markdown(
                f"**Queues:**  "
                f"N: {step_data['queue_N']}  &nbsp;|&nbsp;  "
                f"S: {step_data['queue_S']}  &nbsp;|&nbsp;  "
                f"E: {step_data['queue_E']}  &nbsp;|&nbsp;  "
                f"W: {step_data['queue_W']}"
            )

        # ── Update charts every 5 steps ──────────────────────────────────────
        if step_data["step"] % 5 == 0:
            df = tracker.get_dataframe(st.session_state.mode)
            if not df.empty:
                chart_phs["queue"].plotly_chart(
                    plot_queue_chart(df), use_container_width=True,
                    key=f"q_{i}")
                chart_phs["wait"].plotly_chart(
                    plot_wait_chart(df), use_container_width=True,
                    key=f"w_{i}")
                chart_phs["throughput"].plotly_chart(
                    plot_throughput_chart(df), use_container_width=True,
                    key=f"t_{i}")
                chart_phs["reward"].plotly_chart(
                    plot_reward_chart(df), use_container_width=True,
                    key=f"r_{i}")

            # Update comparison table
            comp_df = tracker.get_comparison_table()
            if not comp_df.empty:
                comparison_ph.dataframe(comp_df, use_container_width=True)

        time.sleep(st.session_state.speed)
