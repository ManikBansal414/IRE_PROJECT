#!/bin/bash
# Follow-up runs after full_run.sbatch: MiniLM reranking on dense / hybrid candidates
# (bge-reranker-base lowered nDCG@10 on hybrid4, so MiniLM is used here).
#   sbatch ada/extra_runs.sbatch
#SBATCH --job-name=mtrag-extra
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

DENSE=${DENSE:-BAAI/bge-base-en-v1.5}
MINILM=cross-encoder/ms-marco-MiniLM-L6-v2
RW=file:data/rewrites_t5canard.jsonl

cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mtrag
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8

python -m taska.run --sources dense:$DENSE@$RW \
    --rerank $MINILM --rerank_query $RW --rerank_depth 50 --name dense_t5__rr_minilm50 --write
python -m taska.run --sources dense:$DENSE@$RW dense:$DENSE@last bm25@$RW bm25@weighted bm25@last \
    --name hybrid5
python -m taska.run --sources dense:$DENSE@$RW dense:$DENSE@last bm25@$RW bm25@weighted bm25@last \
    --rerank $MINILM --rerank_query $RW --rerank_depth 50 --name hybrid5__rr_minilm50 --write
