#!/usr/bin/env python3
import sys
print("[HELLO] gm-run works, python", sys.version, flush=True)
print("[HELLO] cwd probe", flush=True)
import os
print("[HELLO] cwd =", os.getcwd(), flush=True)
