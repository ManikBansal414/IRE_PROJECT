#!/bin/bash
# Task B/C RB_alg on a GPU: the official run_algorithmic.py (BERTScore deberta-xlarge-mnli + ROUGE-L), unchanged,
# for every work/taskb_eval/<run>.merged.jsonl that has no <run>.alg.jsonl yet (made by taskb.evaluate --prepare_alg).
# Needs the ~/mtrag-eval environment (evaluate, bert_score, rouge-score, bs4, lxml on top of the mtrag env) and
# microsoft/deberta-xlarge-mnli in the Hugging Face cache.
#   sbatch ada/rb_alg.sbatch
#SBATCH --job-name=mtrag-rbalg
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
source ~/mtrag-eval/bin/activate
export PYTHONIOENCODING=utf-8  # not offline: the evaluate library resolves its metric scripts online

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available on '+__import__('socket').gethostname())"
EVAL=external/mt-rag-benchmark/scripts/evaluation
for merged in work/taskb_eval/*.merged.jsonl; do
  run=${merged%.merged.jsonl}
  [ -f "$run.alg.jsonl" ] && continue
  n=$(wc -l < "$merged")
  date; echo "scoring $run ($n tasks)"
  python $EVAL/run_algorithmic.py -i "$merged" -o "$run.alg.part.jsonl" -e taskb/alg_config.yaml &
  pid=$!
  # The official script sometimes hangs on exit after writing every row, so a complete file is accepted.
  while kill -0 $pid 2>/dev/null; do
    if [ -f "$run.alg.part.jsonl" ] && [ "$(wc -l < "$run.alg.part.jsonl")" -ge "$n" ]; then
      sleep 10; kill $pid 2>/dev/null || true; break
    fi
    sleep 5
  done
  wait $pid || true
  [ "$(wc -l < "$run.alg.part.jsonl")" -eq "$n" ] || { echo "incomplete: $run"; exit 1; }
  mv "$run.alg.part.jsonl" "$run.alg.jsonl"
done
date; echo RBALG DONE
