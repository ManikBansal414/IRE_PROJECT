#!/bin/bash
# Encode all corpora, then run the scoring pipeline, in ONE job
# (the cs4406 QOS allows only one submitted job per user, so jobs cannot be chained).
# Encoding is resumable: shards already on disk are skipped.
#   sbatch ada/full_run.sbatch
#SBATCH --job-name=mtrag-full
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
DENSE=${DENSE:-BAAI/bge-base-en-v1.5}

nvidia-smi --query-gpu=name,memory.total --format=csv
python -m taska.dense --model "$DENSE" --batch_size 128
SLURM_SUBMIT_DIR="$SLURM_SUBMIT_DIR" bash ada/run_pipeline.sbatch
