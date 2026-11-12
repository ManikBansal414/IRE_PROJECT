#!/bin/bash
# Tuned Task A system (results/tuning/README.md): weighted RRF of 8 BM25/dense runs, BM25 k1=2.2 b=0.5,
# top 50 reranked by bge-reranker-v2-m3 (T5 rewrite) + MiniLM-L6 (last question), alpha 0.7.
# Needs the bge-base embeddings from ada/encode_dense.sbatch. Builds the k1=2.2/b=0.5 BM25 index on first use.
#   sbatch ada/best_run.sbatch                                   # dev set
#   INPUT=data/taskA_test_input.jsonl NAME=best_test sbatch ada/best_run.sbatch
#SBATCH --job-name=mtrag-best
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
# Stop at once if CUDA is unusable on this node: on CPU the rerankers would run past the 6 h limit.
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available on '+__import__('socket').gethostname())"
python -m taska.run --preset best --input "${INPUT:-data/taskA_dev_input.jsonl}" --name "${NAME:-best}" --write
# Everything goes to results/best_system/; the run stops at once if a file with that name already exists.
