import numpy as np

from tsc.env import MAX_GREEN, TscEnv, tmp_tripinfo
from tsc.metrics import read_tripinfo


def test_env_runs_an_hour_and_enforces_max_green():
    trip = tmp_tripinfo("test")
    env = TscEnv("knhz_08", tripinfo=trip)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    f = env.features()
    assert len(f.phase_queue) == 8 and len(env.in_lanes) == 8
    done, longest = False, 0.0
    while not done:
        obs, r, _, done, info = env.step(0)        # always ask for EW-through
        longest = max(longest, env.features_now.green_time)
    assert env.forced_switches > 0                 # max green forced it off
    assert longest <= MAX_GREEN + 5
    trips = read_tripinfo(info["tripinfo"])
    assert len(trips) > 300 and np.isfinite(trips["co2_g"]).all()


def test_each_obs_encoding_matches_space():
    for kind in ["vector", "presslight", "frap"]:
        env = TscEnv("knhz_08", obs=kind)
        obs, _ = env.reset()
        obs2, *_ = env.step(1)
        assert obs.shape == obs2.shape == env.observation_space.shape, kind
        env.close()
