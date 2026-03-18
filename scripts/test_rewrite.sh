#!/bin/bash
module purge
module load python/3.12.1
module load cuda/12.8.1

cd ~/Medcompreviser
source .venv312/bin/activate

export HF_HOME=/scratch/mihalcea_root/mihalcea0/jnwatu/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/mihalcea_root/mihalcea0/jnwatu/hf_cache
export XDG_CACHE_HOME=/scratch/mihalcea_root/mihalcea0/jnwatu/xdg_cache
export VLLM_CACHE_ROOT=/scratch/mihalcea_root/mihalcea0/jnwatu/vllm_cache
export TMPDIR=/scratch/mihalcea_root/mihalcea0/jnwatu/tmp
export HF_HUB_DISABLE_XET=1
export VLLM_NO_USAGE_STATS=1

python scripts/run_pipeline.py \
  --input data/de_novo/1.pdf \
  --output outputs/sample_run.json \
  --target-grade 6 \
  --model-name qwen14b