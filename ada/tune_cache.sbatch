#!/bin/bash
# Tuning step 1: encode corpora (bge-base), cache top-100 runs for every query variant, BM25 k1/b grid.
# All steps are resumable; re-submit if the job hits the time limit.
#   sbatch ada/tune_cache.sbatch
#SBATCH --job-name=mtrag-tune-cache
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -e
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mtrag
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8

nvidia-smi --query-gpu=name,memory.total --format=csv
date; python -m taska.dense --model BAAI/bge-base-en-v1.5 --batch_size 128
date; python -m taska.cache runs
date; python -m taska.cache bm25grid
date; echo CACHE DONE
