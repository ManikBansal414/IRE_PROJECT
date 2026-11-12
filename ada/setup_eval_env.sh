#!/bin/bash
# One-time setup on the ADA login node (has internet) for ada/rb_alg.sbatch: the official RB_alg scorer of
# Tasks B/C (BERTScore with microsoft/deberta-xlarge-mnli + ROUGE-L). Needs ada/setup_env.sh to have run first.
#   cd ~/mtrag && bash ada/setup_eval_env.sh
set -e
cd ~/mtrag
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mtrag

echo "== ~/mtrag-eval environment =="
# Separate environment on top of mtrag: the scorer needs evaluate and transformers<5, which must not change mtrag.
[ -d ~/mtrag-eval ] || python -m venv --system-site-packages ~/mtrag-eval
source ~/mtrag-eval/bin/activate
pip install -q "evaluate==0.4.3" bert_score rouge-score beautifulsoup4 lxml "transformers<5"

echo "== official scoring code =="
E=external/mt-rag-benchmark/scripts/evaluation
R=https://raw.githubusercontent.com/IBM/mt-rag-benchmark/main/scripts/evaluation
mkdir -p $E work/taskb_eval logs
[ -s $E/run_algorithmic.py ] || wget -q -O $E/run_algorithmic.py $R/run_algorithmic.py

echo "== scoring model (2.9 GB) and metric scripts =="
python - <<'EOF'
import evaluate
from huggingface_hub import snapshot_download
# PyTorch weights only: the repo also holds a 3 GB TensorFlow copy.
print(snapshot_download("microsoft/deberta-xlarge-mnli", allow_patterns=["*.json", "*.txt", "pytorch_model.bin"]))
evaluate.load("bertscore")
evaluate.load("rouge")
EOF
python -c "import evaluate, bert_score, rouge_score, bs4, lxml, transformers; print('ok, transformers', transformers.__version__)"
echo "EVAL SETUP DONE"
