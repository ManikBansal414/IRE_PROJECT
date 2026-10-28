#!/bin/bash
# Score dense, hybrid (BM25 + dense via RRF) and hybrid + reranker runs on ADA.
# Needs ada/encode_dense.sbatch to have finished for $DENSE.
#   sbatch ada/run_pipeline.sbatch
#SBATCH --job-name=mtrag-pipeline
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

DENSE=${DENSE:-BAAI/bge-base-en-v1.5}
RERANKER=${RERANKER:-BAAI/bge-reranker-base}
RW=file:data/rewrites_t5canard.jsonl

cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mtrag
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8

python -m taska.run --sources bm25@last --name bm25_last
python -m taska.run --sources dense:$DENSE@last --name dense_last
python -m taska.run --sources dense:$DENSE@$RW --name dense_t5
python -m taska.run --sources dense:$DENSE@weighted --name dense_weighted
python -m taska.run --sources dense:$DENSE@$RW dense:$DENSE@weighted bm25@$RW bm25@weighted \
    --name hybrid4
python -m taska.run --sources dense:$DENSE@$RW dense:$DENSE@weighted bm25@$RW bm25@weighted \
    --rerank $RERANKER --rerank_query $RW --rerank_depth 100 --name hybrid4__rr --write
