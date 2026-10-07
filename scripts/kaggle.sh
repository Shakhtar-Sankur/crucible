#!/bin/bash
# crucible on Kaggle: what the machine allows, the tests, the spawn benchmark.
# Paste from "== crucible" to "== done".
set -e
cd /kaggle/working 2>/dev/null || cd /tmp
rm -rf crucible && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/crucible && cd crucible
echo "== crucible $(git log -1 --format='%h %s')"
pip install -q pytest 2>&1 | tail -1 || true
echo "== probe"; python -m crucible.probe
echo "== tests"; python -m pytest -q tests 2>&1 | tail -15
echo "== spawn benchmark"; python bench/spawn.py 2>&1 | tail -4
echo "== done"
