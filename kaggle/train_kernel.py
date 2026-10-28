"""
Kaggle kernel: train the learning controllers on Kaggle and save models/ as output.

Pushed by kaggle/run.sh. Settings (env vars set by run.sh at the top of the pushed copy):
  TSC_BRANCH   git branch to clone            (default: sumo-benchmark)
  TSC_METHODS  comma list or "all"            (default: all)
  TSC_HOURS    training hours per method, "" = configs/training.yaml
"""

import os
import shutil
import subprocess
import sys

REPO = "https://github.com/Hritikshah02/RL-based-traffic-signal-control.git"
BRANCH = os.environ.get("TSC_BRANCH", "sumo-benchmark")
METHODS = os.environ.get("TSC_METHODS", "all")
HOURS = os.environ.get("TSC_HOURS", "")


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone --depth 1 -b {BRANCH} {REPO} /kaggle/tmp/tsc")
sh(f"{sys.executable} -m pip install -q eclipse-sumo==1.28.0 libsumo==1.28.0 sumo-rl==1.4.5 stable-baselines3 pyyaml")
os.chdir("/kaggle/tmp/tsc")
sh(f"{sys.executable} -m tsc.train --method {METHODS}" + (f" --hours {HOURS}" if HOURS else ""))
shutil.copytree("models", "/kaggle/working/models", dirs_exist_ok=True)
print("Saved models to /kaggle/working/models")
