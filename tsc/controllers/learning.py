"""
learning.py - The five learning controllers and their networks.

  QLearning   tabular Q-learning (the original project's agent, ported to SUMO)
  D3QN        Dueling Double DQN over queue/count features
  PressLight  DQN with position-aware lane counts, pressure reward (Wei et al., KDD 2019)
  FRAP        phase-competition network (Zheng et al., CIKM 2019)
  PPO         stable-baselines3 Proximal Policy Optimisation

Training lives in tsc/train.py; this module only defines models and greedy controllers.
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

from tsc import MODELS_DIR, ROOT
from tsc.controllers.base import Controller
from tsc.env import APPROACHES, OBS_FNS, JunctionFeatures, MAX_GREEN, PHASE_NAMES

NUM_PHASES = len(PHASE_NAMES)
# incoming lanes served by each green phase (fixed by the dataset's junction; env.phase_lanes)
PHASE_LANES = [[2, 6], [0, 4], [3, 7], [1, 5], [6, 7], [2, 3], [4, 5], [0, 1]]


def train_config(method: str) -> Dict:
    """configs/training.yaml: common + method section (+ shared dqn section)."""
    with open(os.path.join(ROOT, "configs", "training.yaml")) as f:
        cfg = yaml.safe_load(f)
    out = dict(cfg["common"])
    if method in ("D3QN", "PressLight", "FRAP"):
        out.update(cfg["dqn"])
    out.update(cfg.get(method, {}))
    return out


def model_path(method: str) -> str:
    ext = {"QLearning": "pkl", "PPO": "zip"}.get(method, "pt")
    return os.path.join(MODELS_DIR, f"{method}.{ext}")


# ── Q-learning ───────────────────────────────────────────────────────────────
QUEUE_BUCKETS = [4, 10]   # per-approach queue: 0-3 low, 4-9 medium, 10+ high


def qtable_state(f: JunctionFeatures) -> Tuple[int, ...]:
    """Discrete state: queue bucket per approach (N,E,S,W) + current phase + can-switch."""
    buckets = tuple(int(np.searchsorted(QUEUE_BUCKETS, q, side="right")) for q in f.approach_queue)
    return buckets + (f.current_phase, int(f.can_switch))


class QLearningController(Controller):
    name = "QLearning"

    def __init__(self, q_table: Dict) -> None:
        self.q = q_table

    def act(self, f: JunctionFeatures) -> int:
        values = self.q.get(qtable_state(f))
        if values is None:   # unseen state: fall back to the longest queue
            return int(np.argmax(f.phase_queue)) if f.can_switch else f.current_phase
        return int(np.argmax(values))


# ── Networks ─────────────────────────────────────────────────────────────────
class MLPQNet(nn.Module):
    """Plain Q-network (PressLight)."""

    def __init__(self, obs_dim: int, n_actions: int = NUM_PHASES, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNet(nn.Module):
    """Dueling Q-network: Q = V + A - mean(A) (D3QN)."""

    def __init__(self, obs_dim: int, n_actions: int = NUM_PHASES, hidden: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        a = self.adv(h)
        return self.value(h) + a - a.mean(dim=1, keepdim=True)


class FRAPNet(nn.Module):
    """
    FRAP: score phases by pairwise competition.

    movement m:   e_m = fc([emb(demand_m), emb(is_green_m)])
    phase p:      d_p = Σ_{m in p} e_m
    pair (p, q):  h_pq = conv1x1(d_p ⊕ d_q) ⊙ conv1x1(emb(relation_pq))
                  relation = 1 if p and q share a movement (partially compatible) else 0
    Q_p = Σ_{q≠p} conv1x1(h_pq)
    Input is obs_frap: [8 lane counts, 8-way current-phase one-hot].
    """

    def __init__(self, phase_lanes: List[List[int]] = PHASE_LANES, dim: int = 16, hidden: int = 20) -> None:
        super().__init__()
        n_lanes = 1 + max(l for p in phase_lanes for l in p)
        p = len(phase_lanes)
        member = torch.zeros(p, n_lanes)
        for i, lanes in enumerate(phase_lanes):
            member[i, lanes] = 1.0
        self.register_buffer("member", member)                       # (P, M)
        rel = torch.tensor([[float((member[i] * member[j]).sum() > 0) for j in range(p)] for i in range(p)])
        others = [[j for j in range(p) if j != i] for i in range(p)]
        self.register_buffer("others", torch.tensor(others))          # (P, P-1)
        self.register_buffer("relation", torch.stack([rel[i, others[i]] for i in range(p)]).long())

        self.demand_emb = nn.Sequential(nn.Linear(1, 4), nn.Sigmoid())
        self.green_emb = nn.Embedding(2, 4)
        self.move = nn.Sequential(nn.Linear(8, dim), nn.ReLU())
        self.pair = nn.Sequential(nn.Conv2d(2 * dim, hidden, 1), nn.ReLU())
        self.rel_emb = nn.Embedding(2, 4)
        self.rel = nn.Sequential(nn.Conv2d(4, hidden, 1), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(hidden, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.member.shape[1]
        counts, phase_onehot = x[:, :m], x[:, m:]
        green = (phase_onehot @ self.member).clamp(max=1).long()                 # (B, M)
        e = self.move(torch.cat([self.demand_emb(counts.unsqueeze(-1)), self.green_emb(green)], -1))
        d = torch.einsum("pm,bmh->bph", self.member, e)                            # (B, P, H)
        dp = d.unsqueeze(2).expand(-1, -1, self.others.shape[1], -1)                # (B, P, P-1, H)
        dq = d[:, self.others]                                                     # (B, P, P-1, H)
        pair = self.pair(torch.cat([dp, dq], -1).permute(0, 3, 1, 2))              # (B, C, P, P-1)
        rel = self.rel(self.rel_emb(self.relation).permute(2, 0, 1).unsqueeze(0))  # (1, C, P, P-1)
        return self.out(pair * rel).sum(dim=3).squeeze(1)                          # (B, P)


DQN_SPECS = {
    "D3QN": lambda obs_dim: DuelingQNet(obs_dim),
    "PressLight": lambda obs_dim: MLPQNet(obs_dim),
    "FRAP": lambda obs_dim: FRAPNet(),
}


class QNetController(Controller):
    """Greedy controller for any of the DQN-family networks."""

    def __init__(self, name: str, net: nn.Module, obs: str) -> None:
        self.name, self.net, self.obs_fn = name, net.eval(), OBS_FNS[obs]

    @torch.no_grad()
    def act(self, f: JunctionFeatures) -> int:
        q = self.net(torch.as_tensor(self.obs_fn(f)).unsqueeze(0))
        return int(q.argmax(dim=1).item())


class PPOController(Controller):
    name = "PPO"

    def __init__(self, model) -> None:
        self.model = model

    def act(self, f: JunctionFeatures) -> int:
        action, _ = self.model.predict(OBS_FNS["vector"](f), deterministic=True)
        return int(action)


def load(method: str) -> Controller:
    """Load a trained controller from models/ (FileNotFoundError if not trained yet)."""
    path = model_path(method)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run: python -m tsc.train --method {method}")
    if method == "QLearning":
        with open(path, "rb") as f:
            return QLearningController(pickle.load(f))
    if method == "PPO":
        from stable_baselines3 import PPO

        return PPOController(PPO.load(path, device="cpu"))
    ckpt = torch.load(path, map_location="cpu")
    net = DQN_SPECS[method](ckpt["obs_dim"])
    net.load_state_dict(ckpt["state_dict"])
    return QNetController(method, net, ckpt["obs"])
