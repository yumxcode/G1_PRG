#!/usr/bin/env python3
"""C1a main-experiment launcher (gradmotion): G1-AMP with the PRG adaptive
gait clock (contact-triggered HARD phase reset, reset-only, freq_mod off).
Single-variable twin of the B0 launcher: ONLY G1_PRG differs (1 vs 0);
envs/iters identical (1024 x 8000).

Evidence chain pre-registered in exp_a7fb423e: judge-run 2x2 shows the
PRG arm survives ~6x longer at 1024 envs (ep_len 59 vs 10); this run
decides H1 at full budget.
"""
import os
import subprocess
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
_train = _root / "robolab" / "scripts" / "rsl_rl" / "train.py"

os.environ.setdefault("G1_PRG", "1")            # C1a: PRG adaptive clock ON
os.environ.setdefault("PRG_RESET_MODE", "hard") # touchdown re-anchoring
os.environ.setdefault("X1_GAIT_CLOCK", "1")     # clock obs+rewards ON
os.environ.setdefault("X1_STYLE_SCALE", "0")    # task-first: AMP style OFF
os.environ.setdefault("X1_ACT_LPF_HZ", "30")    # action LPF (shared w/ B0)
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
    "--num_envs=" + os.environ.get("G1_TRAIN_ENVS", "1024"),
    "--max_iterations=" + os.environ.get("G1_TRAIN_ITERS", "8000"),
    "--run_name=" + os.environ.get("G1_RUN_NAME", "c1a_prg_hard_v1"),
]
print(f"[C1a] argv={argv}", flush=True)
print(f"[C1a] G1_PRG={os.environ['G1_PRG']} PRG_RESET_MODE={os.environ['PRG_RESET_MODE']}", flush=True)
sys.exit(subprocess.call(argv, cwd=str(_root), env=env))
