import pandas as pd
import pytest
import torch

from tsc import train as train_mod
from tsc.controllers import LEARNING, learning, make
from tsc.controllers.learning import FRAPNet
from tsc.env import TscEnv


def test_frap_net_scores_every_phase_and_is_permutation_aware():
    net = FRAPNet()
    x = torch.zeros(3, 16)
    x[:, 8] = 1.0                       # current phase 0
    x[1, [2, 6]] = 0.5                  # demand on phase 0's lanes
    q = net(x)
    assert q.shape == (3, 8)
    assert not torch.allclose(q[0], q[1])


@pytest.mark.parametrize("method", LEARNING)
def test_each_method_trains_one_episode_and_loads(method, tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(train_mod, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(learning, "train_config",
                        lambda m, f=learning.train_config: {**f(m), "validation": ["knhz_08"]})
    monkeypatch.setattr(train_mod, "train_config", learning.train_config)
    train_mod.train(method, hours=1)
    curve = pd.read_csv(tmp_path / f"{method}_train.csv")
    assert len(curve) == 1 and "val_queue" in curve

    ctrl = make(method)
    env = TscEnv("knhz_08")
    env.reset()
    ctrl.reset(env)
    assert 0 <= ctrl.act(env.features()) < 8
    env.close()
