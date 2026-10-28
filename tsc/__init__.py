"""
tsc - Traffic Signal Control benchmark on real Hangzhou traffic in SUMO.

Importing this package configures SUMO (SUMO_HOME + libsumo) before sumo-rl is
imported anywhere, so always `import tsc` first.
"""

import os

try:
    import sumo as _sumo  # provided by the eclipse-sumo wheel

    os.environ.setdefault("SUMO_HOME", _sumo.SUMO_HOME)
except ImportError:  # SUMO installed system-wide instead
    pass

# libsumo runs SUMO in-process (much faster than a TraCI socket).
# Limitation: one simulation per process - the benchmark uses multiprocessing.
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_SUMO = os.path.join(ROOT, "data", "sumo")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
