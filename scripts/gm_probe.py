#!/usr/bin/env python3
"""Minimal gradmotion container probe: locate the cloned repo, dump env and
directory layout, so a zero-log task death becomes diagnosable."""
import os
import subprocess
import sys

print("[PROBE] cwd =", os.getcwd(), flush=True)
print("[PROBE] python =", sys.version, flush=True)
print("[PROBE] argv =", sys.argv, flush=True)
for root in ("/", os.getcwd()):
    try:
        entries = sorted(os.listdir(root))
        print(f"[PROBE] ls {root} -> {entries[:40]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[PROBE] ls {root} failed: {e}", flush=True)
# find the repo dir (any dir containing scripts/gm_probe.py's parent name)
for cand in ("/workspace", "/code", "/root", "/G1_PRG"):
    print(f"[PROBE] exists {cand}: {os.path.exists(cand)}", flush=True)
r = subprocess.run(["bash", "-lc", "ls -la; find / -maxdepth 3 -name 'gm_probe.py' 2>/dev/null | head -3"],
                   capture_output=True, text=True, timeout=60)
print("[PROBE] find output:\n", r.stdout[-2000:], r.stderr[-500:], flush=True)
print("[PROBE] DONE", flush=True)
