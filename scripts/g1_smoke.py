#!/usr/bin/env python3
"""G1 smoke launcher (gradmotion): C1a PRG mode — G1_PRG=1 (reset-only
adaptive clock) on the G1-AMP task, short run to validate the END-TO-END
chain before any real training:
  repo clone -> URDF load (g1_custom_collision_29dof) -> X1->G1 name
  retarget ([G1-RETARGET]) -> PRG cfg wiring ([G1-PRG-CFG]) -> PRG state
  build/step ([PRG]/[PRG-AUDIT]) -> PPO learning signal.

Success criteria (pre-registered): reaches iteration 100 without crash,
all four audit prints present, mean_reward trending up or stable, AMP disc
terms not NaN. Failure layer is directly readable from which print is the
last one seen.
"""
import os
import subprocess
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
_train = _root / "robolab" / "scripts" / "rsl_rl" / "train.py"

# --- smoke configuration (env vars; single-variable chain kept for real runs)
os.environ.setdefault("G1_PRG", "1")            # C1a: PRG adaptive clock ON
os.environ.setdefault("PRG_RESET_MODE", "hard") # hard reset first (C1a)
os.environ.setdefault("X1_GAIT_CLOCK", "1")     # obs/reward weight plumbing
os.environ.setdefault("G1_CMD_LO", "0.2")       # walk band
os.environ.setdefault("G1_CMD_HI", "1.5")

argv = [
    sys.executable, str(_train),
    "--task=G1-AMP",
    "--headless",
    "--logger=tensorboard",
    "--num_envs=1024",
    "--max_iterations=100",
    "--run_name=g1_smoke_prg_c1a",
]
print(f"[G1-SMOKE] argv={argv}", flush=True)
print(f"[G1-SMOKE] G1_PRG={os.environ['G1_PRG']} "
      f"PRG_RESET_MODE={os.environ['PRG_RESET_MODE']}", flush=True)
# PYTHONPATH shadowing (no pip; X1 launcher convention): vendored rsl_rl
# (AMPRunner lives here, not in the image's site-packages) + robolab source
_parts = [str(_root / "rsl_rl"), str(_root / "robolab")]
_old = os.environ.get("PYTHONPATH", "")
if _old:
    _parts.append(_old)
env = dict(os.environ)
env["PYTHONPATH"] = ":".join(_parts)
print(f"[G1-SMOKE] PYTHONPATH shadowing: {env['PYTHONPATH']}", flush=True)

# subprocess (not runpy): running train.py as a SCRIPT puts its own dir at
# sys.path[0], so its sibling module `cli_args` imports — the X1 launcher
# convention. Stream stdout/stderr straight through so the platform log
# shows every audit print unbuffered.
sys.exit(subprocess.call(argv, cwd=str(_root), env=env))
