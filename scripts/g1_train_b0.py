#!/usr/bin/env python3
"""B0 baseline launcher (gradmotion): G1-AMP with the v71 FIXED exogenous
gait clock — the control arm of the pre-registered exp chain (exp_a7fb423e).
C1a differs ONLY in G1_PRG=1 (single variable). Full training: 4096 envs,
4000 iters, save every 500.
"""
import os
import subprocess
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
_train = _root / "robolab" / "scripts" / "rsl_rl" / "train.py"

os.environ.setdefault("G1_PRG", "0")            # B0: FIXED clock (control)
os.environ.setdefault("X1_GAIT_CLOCK", "1")     # v71 clock obs+rewards ON
os.environ.setdefault("X1_STYLE_SCALE", "0")    # task-first: AMP style OFF
os.environ.setdefault("G1_CMD_LO", "0.2")
os.environ.setdefault("G1_CMD_HI", "1.5")

_parts = [str(_root / "rsl_rl"), str(_root / "robolab")]
_old = os.environ.get("PYTHONPATH", "")
if _old:
    _parts.append(_old)
env = dict(os.environ)
env["PYTHONPATH"] = ":".join(_parts)

argv = [
    sys.executable, str(_train),
    "--task=G1-AMP",
    "--headless",
    "--logger=tensorboard",
    "--num_envs=" + os.environ.get("G1_TRAIN_ENVS", "4096"),
    "--max_iterations=" + os.environ.get("G1_TRAIN_ITERS", "4000"),
    "--run_name=" + os.environ.get("G1_RUN_NAME", "b0_fixed_clock_v1"),
]
print(f"[B0] argv={argv}", flush=True)
print(f"[B0] G1_PRG={os.environ['G1_PRG']} X1_GAIT_CLOCK=1 STYLE=0", flush=True)
sys.exit(subprocess.call(argv, cwd=str(_root), env=env))
