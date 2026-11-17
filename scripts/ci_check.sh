#!/usr/bin/env bash
# Minimal CI check for g1_prg (runs on macOS sandbox, no IsaacSim needed)
# 1) AST-parse every python file (no bytecode written; excludes venv/pycache/git/build)
# 2) run pure-logic unit tests if a python with pytest is available
set -u
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
FAIL=0

echo "== AST parse sweep =="
python3 - <<'EOF' || FAIL=1
import ast, os, re, sys
skip = re.compile(r'.*(/venv/|/__pycache__/|/\.git/|/build/|\.venv39)')
bad, n = [], 0
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if not skip.search(f'/{d}/')]
    for f in files:
        if not f.endswith('.py'): continue
        p = os.path.join(root, f)
        if skip.search(p): continue
        n += 1
        try:
            with open(p, 'r', encoding='utf-8', errors='replace') as fh:
                ast.parse(fh.read(), filename=p)
        except SyntaxError as e:
            bad.append(f'{p}: {e}')
print(f'parsed {n} files, {len(bad)} syntax errors')
for b in bad: print(' SYNTAX:', b)
sys.exit(1 if bad else 0)
EOF
echo "AST sweep exit: $?"

echo "== unit tests (pytest if available, else plain scripts) =="
if [ -d tests ]; then
  PYTEST=""
  for c in python3 python3.11 /Users/yumx/code/X1_29_AMP/_gmr_local/venv/bin/python; do
    if "$c" -c "import pytest" 2>/dev/null; then PYTEST="$c"; break; fi
  done
  if [ -n "$PYTEST" ]; then
    "$PYTEST" -m pytest tests/ -q || FAIL=1
  else
    for t in tests/test_*.py; do
      [ -e "$t" ] || continue
      python3 "$t" || FAIL=1
    done
  fi
else
  echo "no tests/ dir yet — skipped"
fi

exit $FAIL
