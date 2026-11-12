#!/bin/bash
# Tuning step 2: cross-encoder scores on the candidate pools built by `python -m taska.sweep pool`.
# Cheap rerankers first; every model is resumable in chunks of 100 questions, so re-submit if cut off.
#   sbatch ada/tune_rerank.sbatch
#SBATCH --job-name=mtrag-tune-rerank
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
R="python -m taska.cache rerank"

nvidia-smi --query-gpu=name,memory.total --format=csv
for q in t5 last weighted gold; do
  date; $R --pool work/cache/pool100.json --model cross-encoder/ms-marco-MiniLM-L6-v2 --query $q --batch_size 64
done
date; $R --pool work/cache/pool100.json --model cross-encoder/ms-marco-MiniLM-L12-v2 --query t5 --batch_size 64
date; $R --pool work/cache/pool100.json --model BAAI/bge-reranker-base --query t5 --batch_size 32
date; $R --pool work/cache/pool50.json --model BAAI/bge-reranker-v2-m3 --query t5 --batch_size 16
date; echo RERANK DONE
