#!/bin/bash
# crucible M3 on Kaggle (Settings: Accelerator "GPU T4 x2", Internet on): GRPO on MBPP with
# ratchet (relay generates on GPU 1, the trainer updates on GPU 0) and crucible's sandboxed
# grader giving the rewards.
#   RUN=smoke (default): 3 steps, 64 test tasks, ~15 minutes; finds problems first.
#   RUN=full:            100 steps, all test tasks, ~1.5 hours ("Save Version -> Save & Run All").
# Paste everything from "== crucible M3" to the end back into the chat.
set -e
RUN=${RUN:-smoke}
cd /kaggle/working 2>/dev/null || cd /tmp
rm -rf crucible ratchet
git clone -q --depth 1 https://github.com/Shakhtar-Sankur/crucible
git clone -q --recursive https://github.com/Shakhtar-Sankur/ratchet
echo "== crucible M3 ($RUN): crucible $(git -C crucible log -1 --format=%h) | ratchet $(git -C ratchet log -1 --format=%h) | relay $(git -C ratchet/relay log -1 --format=%h)"
if ! command -v nvidia-smi >/dev/null || [ "$(nvidia-smi -L | wc -l)" -lt 2 ]; then
  echo "== needs two GPUs: Settings -> Accelerator -> \"GPU T4 x2\" (and Internet on), then run again"; exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
export PATH=/usr/local/cuda/bin:$PATH
(cd ratchet && scripts/build_relay.sh 75 2>&1 | tail -1)
pip install -q transformers huggingface_hub pyarrow 2>&1 | tail -1 || true
python - <<'PY'
import sys
sys.path.insert(0, "crucible")
from huggingface_hub import snapshot_download
from crucible.tasks import download_mbpp
snapshot_download("Qwen/Qwen2.5-Coder-0.5B-Instruct", local_dir="models/coder",
                  allow_patterns=["*.json", "*.safetensors", "*.txt"])
download_mbpp("data/mbpp")
PY
python -c "import sys; sys.path.insert(0, 'crucible'); from crucible.probe import features; import json; print(json.dumps(features()))"
cd crucible && mkdir -p runs
if [ "$RUN" = full ]; then STEPS=100; EVAL=""; else STEPS=3; EVAL="--eval-limit 64"; fi
echo "== train: $STEPS steps, 8 tasks x 8 answers per step, one step ahead"
python -m crucible.rl.mbpp_grpo train --model ../models/coder --mbpp ../data/mbpp --ratchet ../ratchet \
  --train-device cuda:0 --relay-device 1 --steps $STEPS --ahead $EVAL --out runs/m3.jsonl \
  | grep --line-buffered -E '"phase"|"step": [0-9]*0,|"step": [1-3],' || true
echo "== summary"
python scripts/summarize_m3.py runs/m3.jsonl
echo "== done: copy from '== crucible M3' to here and send it back"
