import numpy as np
import pytest

import tsc  # noqa: F401  (configures SUMO before sumo-rl is imported)
from tsc.env import APPROACHES, JunctionFeatures

# Same topology the real junction produces (env.phase_lanes / env.lane_approach)
PHASE_LANES = [[2, 6], [0, 4], [3, 7], [1, 5], [6, 7], [2, 3], [4, 5], [0, 1]]
LANE_APPROACH = [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.fixture
def make_features():
    def _make(count=None, queue=None, out=None, near=None, phase=0, green=20.0, yellow=False, time=100.0):
        z = np.zeros(8)
        count = np.array(count if count is not None else z, dtype=float)
        return JunctionFeatures(
            time=time,
            lane_queue=np.array(queue if queue is not None else count, dtype=float),
            lane_count=count, lane_wait=z.copy(), lane_segments=np.zeros((8, 3)),
            lane_moving_near_stop=np.array(near if near is not None else z, dtype=float),
            out_count=np.array(out if out is not None else z, dtype=float),
            lane_approach=LANE_APPROACH, phase_lanes=PHASE_LANES,
            current_phase=phase, time_in_phase=green + 3, green_time=green, is_yellow=yellow,
        )
    return _make
