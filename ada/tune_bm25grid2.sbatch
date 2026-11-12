#!/bin/bash
# Tuning step 1b: extend the BM25 grid past k1=1.5 (the edge of the first grid). CPU only, no GPU requested.
#   sbatch ada/tune_bm25grid2.sbatch
#SBATCH --job-name=mtrag-bm25grid2
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -e
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mtrag
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8

date; python -m taska.cache bm25grid --k1 1.5 1.8 2.2 2.6 3.0 --b 0.3 0.4 0.5 0.6
date; echo GRID2 DONE
