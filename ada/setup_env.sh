#!/bin/bash
# One-time setup on the ADA login node (has internet): corpora, conda env, models.
#   cd ~/mtrag && nohup bash ada/setup_env.sh > logs/setup.log 2>&1 &
set -e
cd ~/mtrag

echo "== corpora =="
B=external/mt-rag-benchmark
R=https://raw.githubusercontent.com/IBM/mt-rag-benchmark/main
mkdir -p $B/corpora/passage_level
for d in clapnq cloud fiqa govt; do
  f=$B/corpora/passage_level/$d.jsonl
  [ -s $f ] || { wget -q -O $f.zip $R/corpora/passage_level/$d.jsonl.zip && unzip -oq $f.zip -d $B/corpora/passage_level && rm $f.zip; }
  echo "$d: $(wc -l < $f) passages"
done
for d in clapnq cloud fiqa govt; do
  mkdir -p $B/mtrag-human/retrieval_tasks/$d/qrels
  q=$B/mtrag-human/retrieval_tasks/$d/qrels/dev.tsv
  [ -s $q ] || wget -q -T 60 -t 20 -O $q $R/mtrag-human/retrieval_tasks/$d/qrels/dev.tsv
done

echo "== conda =="
if [ ! -d ~/miniconda3 ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc_$USER.sh
  bash /tmp/mc_$USER.sh -b -p ~/miniconda3 && rm /tmp/mc_$USER.sh
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda env list | grep -q "^mtrag " || conda create -n mtrag python=3.11 -y -q \
  --override-channels -c conda-forge
conda activate mtrag
pip install -q "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip install -q -r requirements.txt
python -c "import torch, bm25s, faiss, sentence_transformers, pytrec_eval; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"

echo "== models =="
python - <<'EOF'
from huggingface_hub import snapshot_download as s
for m in ["BAAI/bge-base-en-v1.5", "BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L6-v2"]:
    print(s(m, allow_patterns=["*.json", "*.txt", "*.model", "model.safetensors", "1_Pooling/*", "sentencepiece.bpe.model"]))
EOF
echo "SETUP DONE"
