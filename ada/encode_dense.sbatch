#!/bin/bash
# Encode MTRAG corpora with a dense retriever on one GPU (ADA).
# On the laptop CPU this ran at ~8 passages/s (~12 h for bge-small); on a GPU it is minutes.
#   sbatch ada/encode_dense.sbatch BAAI/bge-base-en-v1.5            # all 4 corpora
#   sbatch ada/encode_dense.sbatch BAAI/bge-base-en-v1.5 fiqa       # smoke test, smallest corpus
# Account limits (cs4406): 1 GPU, 10 CPUs, 3000 MB per CPU, 6 h, 1 running job.
#SBATCH --job-name=mtrag-encode
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

MODEL=${1:-BAAI/bge-base-en-v1.5}
shift
DOMAINS=${@:-clapnq cloud fiqa govt}

cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mtrag
export HF_HUB_OFFLINE=1   # models are pre-downloaded on the login node
nvidia-smi
python -m taska.dense --model "$MODEL" --domains $DOMAINS --batch_size 128
