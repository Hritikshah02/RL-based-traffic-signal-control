"""
train.py - Train the learning controllers on the TRAIN hours.

    python -m tsc.train --method D3QN            # one method
    python -m tsc.train --method all --jobs 5    # all five in parallel
    python -m tsc.train --method FRAP --hours 2  # quick smoke run

Each episode is one real hour (random train hour + departure jitter). Every
`eval_every` episodes the greedy policy is scored on the validation hours and
the best checkpoint is kept in models/. A learning curve goes to models/<m>_train.csv.
"""

import argparse
import copy
import os
import pickle
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import tsc
from tsc import MODELS_DIR
from tsc.controllers import LEARNING
from tsc.controllers.base import Controller
from tsc.controllers.learning import (DQN_SPECS, NUM_PHASES, PPOController, QLearningController,
                                      QNetController, model_path, qtable_state, train_config)
from tsc.env import TscEnv


# ── Shared helpers ───────────────────────────────────────────────────────────
def validate(controller: Controller, hours: List[str]) -> float:
    """Mean queue length of a greedy controller over the validation hours (lower is better)."""
    scores = []
    for tag in hours:
        env = TscEnv(tag)
        env.reset()
        controller.reset(env)
        done = False
        while not done:
            _, _, _, done, _ = env.step(controller.act(env.features()))
        scores.append(np.mean([s["queue"] for s in env.step_log]))
    return float(np.mean(scores))


class Tracker:
    """Records the learning curve and keeps the best checkpoint by validation score."""

    def __init__(self, method: str, cfg: Dict, save: Callable[[], None], make_controller: Callable[[], Controller]):
        self.method, self.cfg, self.save, self.make = method, cfg, save, make_controller
        self.rows: List[Dict] = []
        self.best = float("inf")
        self.t0 = time.time()

    def episode_done(self, ep: int, env: TscEnv, ep_reward: float, eps: float) -> None:
        row = {"episode": ep, "scenario": env.scenario.name, "reward": ep_reward,
               "mean_queue": float(np.mean([s["queue"] for s in env.step_log])), "epsilon": eps}
        if (ep + 1) % self.cfg["eval_every"] == 0 or ep + 1 == self.cfg["hours"]:
            row["val_queue"] = validate(self.make(), self.cfg["validation"])
            if row["val_queue"] < self.best:
                self.best = row["val_queue"]
                self.save()
            print(f"[{self.method}] ep {ep + 1}/{self.cfg['hours']}  train queue {row['mean_queue']:.1f}  "
                  f"val queue {row['val_queue']:.1f}  best {self.best:.1f}  eps {eps:.2f}  "
                  f"{time.time() - self.t0:.0f}s", flush=True)
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(os.path.join(MODELS_DIR, f"{self.method}_train.csv"), index=False)


# ── Tabular Q-learning ───────────────────────────────────────────────────────
def train_qlearning(cfg: Dict, seed: int) -> None:
    rng = random.Random(seed)
    q: Dict = defaultdict(lambda: np.zeros(NUM_PHASES))
    env = TscEnv("train", seed=seed, reward=cfg["reward"])

    def save():
        with open(model_path("QLearning"), "wb") as f:
            pickle.dump({k: v.copy() for k, v in q.items()}, f)

    tracker = Tracker("QLearning", cfg, save, lambda: QLearningController(dict(q)))
    for ep in range(cfg["hours"]):
        eps = cfg["eps_start"] + (cfg["eps_end"] - cfg["eps_start"]) * ep / max(cfg["hours"] - 1, 1)
        env.reset()
        s, done, total = qtable_state(env.features()), False, 0.0
        while not done:
            a = rng.randrange(NUM_PHASES) if rng.random() < eps else int(np.argmax(q[s]))
            _, r, _, done, _ = env.step(a)
            s2 = qtable_state(env.features())
            q[s][a] += cfg["alpha"] * (r + cfg["gamma"] * np.max(q[s2]) * (not done) - q[s][a])
            s, total = s2, total + r
        tracker.episode_done(ep, env, total, eps)


# ── DQN family (D3QN / PressLight / FRAP) ────────────────────────────────────
class Replay:
    def __init__(self, size: int, obs_dim: int) -> None:
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros(size, np.int64)
        self.rew = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.size, self.n, self.i = size, 0, 0

    def add(self, o, a, r, o2, d) -> None:
        self.obs[self.i], self.act[self.i], self.rew[self.i], self.next[self.i], self.done[self.i] = o, a, r, o2, d
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, k: int, device):
        idx = np.random.randint(0, self.n, k)
        t = lambda x: torch.as_tensor(x[idx], device=device)
        return t(self.obs), t(self.act), t(self.rew), t(self.next), t(self.done)


def train_dqn(method: str, cfg: Dict, seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TscEnv("train", seed=seed, reward=cfg["reward"], obs=cfg["obs"])
    obs_dim = env.observation_space.shape[0]
    net = DQN_SPECS[method](obs_dim).to(device)
    target = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    buf = Replay(cfg["buffer"], obs_dim)
    total_steps = cfg["hours"] * 720
    decay_steps = max(int(cfg["eps_fraction"] * total_steps), 1)

    def save():
        torch.save({"state_dict": copy.deepcopy(net).cpu().state_dict(), "obs_dim": obs_dim, "obs": cfg["obs"]},
                   model_path(method))

    def controller():
        return QNetController(method, copy.deepcopy(net).cpu(), cfg["obs"])

    tracker = Tracker(method, cfg, save, controller)
    step = 0
    for ep in range(cfg["hours"]):
        o, _ = env.reset()
        done, total = False, 0.0
        while not done:
            eps = max(cfg["eps_end"], cfg["eps_start"] - (cfg["eps_start"] - cfg["eps_end"]) * step / decay_steps)
            if step < cfg["warmup"] or rng.random() < eps:
                a = rng.randrange(NUM_PHASES)
            else:
                with torch.no_grad():
                    a = int(net(torch.as_tensor(o, device=device).unsqueeze(0)).argmax(1).item())
            o2, r, _, done, _ = env.step(a)
            # episode end is a time limit, not a terminal state -> keep bootstrapping
            buf.add(o, a, r, o2, 0.0)
            o, total, step = o2, total + r, step + 1

            if step >= cfg["warmup"]:
                bo, ba, br, bo2, bd = buf.sample(cfg["batch"], device)
                with torch.no_grad():   # Double DQN target
                    a2 = net(bo2).argmax(1, keepdim=True)
                    y = br + cfg["gamma"] * (1 - bd) * target(bo2).gather(1, a2).squeeze(1)
                loss = F.smooth_l1_loss(net(bo).gather(1, ba.unsqueeze(1)).squeeze(1), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
            if step % cfg["target_update"] == 0:
                target.load_state_dict(net.state_dict())
        tracker.episode_done(ep, env, total, eps)


# ── PPO (stable-baselines3) ──────────────────────────────────────────────────
def train_ppo(cfg: Dict, seed: int) -> None:
    from stable_baselines3 import PPO

    env = TscEnv("train", seed=seed, reward=cfg["reward"], obs=cfg["obs"])
    model = PPO("MlpPolicy", env, learning_rate=cfg["lr"], n_steps=cfg["n_steps"], batch_size=cfg["batch"],
                n_epochs=cfg["epochs"], gamma=cfg["gamma"], ent_coef=cfg["ent_coef"], seed=seed,
                device="cpu", verbose=0)
    tracker = Tracker("PPO", cfg, lambda: model.save(model_path("PPO")), lambda: PPOController(model))
    episode = iter(range(10 ** 9))
    env.on_episode_end = lambda e: tracker.episode_done(next(episode), e, e.episode_reward, 0.0)
    model.learn(total_timesteps=cfg["hours"] * 720)


def train(method: str, hours: int = None, seed: int = 0) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.set_num_threads(1)
    cfg = train_config(method)
    if hours:
        cfg["hours"] = hours
        cfg["eval_every"] = min(cfg["eval_every"], hours)
        cfg["warmup"] = min(cfg.get("warmup", 0), hours * 360)
    if method == "QLearning":
        train_qlearning(cfg, seed)
    elif method == "PPO":
        train_ppo(cfg, seed)
    else:
        train_dqn(method, cfg, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="all", help=f"one of {LEARNING} or 'all'")
    ap.add_argument("--hours", type=int, default=None, help="override training budget (episodes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=len(LEARNING))
    args = ap.parse_args()
    methods = LEARNING if args.method == "all" else args.method.split(",")
    if len(methods) == 1:
        train(methods[0], args.hours, args.seed)
        return
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for fut in [pool.submit(train, m, args.hours, args.seed) for m in methods]:
            fut.result()


if __name__ == "__main__":
    main()
