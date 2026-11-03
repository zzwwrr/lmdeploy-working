#!/usr/bin/env bash
set -euo pipefail
cd /workspace/dev/lmdeploy-decision-parser
export PYTHONPATH=/workspace/dev/lmdeploy-decision-parser:/tmp/lmdeploy-rebase-main-20261030-site:/workspace/dev/cache/uv/archive-v0/H54tyfQwHbSJOSGYJch_7
export LD_LIBRARY_PATH=/tmp/lmdeploy-rebase-main-20261030-site/z3/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
root=${1:-scratch/results/pr5075_live_20261103/attempt3}
if [[ -e "$root" ]]; then
    echo "Output directory already exists: $root; pass a fresh path as the first argument." >&2
    exit 1
fi
mkdir -p "$root"
for size in 0.8B 2B 4B; do
    dest="$root/$size"
    mkdir -p "$dest"
    /opt/py3/bin/python scratch/pr5075_live_eval.py --size "$size" --output "$dest" > "$dest/lmdeploy.log" 2>&1
    /opt/py3/bin/python scratch/pr5075_live_eval.py --size "$size" --output "$dest" --hf-latency > "$dest/hf.log" 2>&1
done
touch "$root/complete"
