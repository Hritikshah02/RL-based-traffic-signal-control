"""
Legacy page - the original toy Q-learning simulator (kept for before/after comparison).
"""

import os
import runpy
import sys

LEGACY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "legacy")
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)

runpy.run_path(os.path.join(LEGACY_DIR, "legacy_app.py"), run_name="__main__")
