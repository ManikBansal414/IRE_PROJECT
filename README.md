# MTRAGEval (SemEval-2026 Task 8) — Task A: Retrieval

Given a multi-turn conversation ending in a user question, return the 10 most relevant passages
from the domain corpus (ClapNQ, Cloud, FiQA, Govt). Ranked by nDCG; scored on answerable questions only.

## Results (MTRAG dev set, 777 answerable questions)

| # | System | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | R@10 |
|---|---|---|---|---|---|---|
| 1 | BM25, last question only (baseline; paper reports 0.21) | 0.179 | 0.174 | 0.195 | 0.224 | 0.286 |
| 2 | BM25, all user questions concatenated | 0.149 | 0.143 | 0.160 | 0.185 | 0.239 |
| 3 | BM25, last 2 questions | 0.192 | 0.175 | 0.191 | 0.220 | 0.280 |
| 4 | BM25, T5-CANARD rewrite | 0.189 | 0.184 | 0.207 | 0.237 | 0.304 |
| 5 | BM25, weighted history (last question ×2 + previous 2) | 0.197 | 0.188 | 0.211 | 0.241 | 0.310 |
| 6 | RRF of BM25 runs (#4 + #5 + #1) | 0.206 | 0.192 | 0.219 | 0.250 | 0.321 |
| **7** | **#6 + MiniLM cross-encoder rerank of top 50** | **0.349** | **0.313** | **0.334** | **0.366** | **0.427** |
| — | *BM25 on gold human rewrites (upper bound only, not available at test time)* | 0.203 | 0.196 | 0.216 | 0.254 | 0.330 |

Per-domain numbers: [results/results_log.md](results/results_log.md) and `results/*.metrics.json`.
Submission-format output for #7: `results/bm25rrf3__rr_minilm50.predictions.jsonl`
(passes the official `format_checker.py --mode retrieval_taska`).

**In progress (ADA GPU):** dense retrieval (BGE-base), hybrid BM25 + dense via RRF, and a stronger
reranker (bge-reranker-base). On the laptop CPU dense encoding ran at ~8 passages/s (~12 h for the
366k passages), so it is run on ADA instead.

## Pipeline

```
conversation ──► query variants ──► BM25 / dense retrieval (top 100 each)
                 (last turn,          │
                  weighted history,   ▼
                  T5 rewrite)     RRF fusion ──► cross-encoder rerank (top 50) ──► top 10
```

- `taska/queries.py` — query construction from the conversation (no gold info used).
- `taska/rewrite.py` — conversational query rewriting with `castorini/t5-base-canard`.
- `taska/bm25.py` — BM25 (bm25s, k1=0.9, b=0.4, English stemming + stopwords), one index per domain.
- `taska/dense.py` — sentence-transformers encoder + exact FAISS search; resumable sharded encoding.
- `taska/fusion.py` — Reciprocal Rank Fusion (k=60).
- `taska/rerank.py` — cross-encoder reranking.
- `taska/evaluate.py` — nDCG/Recall @1,3,5,10 via pytrec_eval, same aggregation as the official script.
- `taska/run.py` — runs a pipeline, scores it, writes official-format predictions.

Evaluation-time inputs contain no metadata (answerability, question type, rewrites), so
`taska/data.py` strips those fields on load. The gold rewrites are used only for the upper-bound row.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate        # Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Data (not committed): download the [MTRAG benchmark](https://github.com/IBM/mt-rag-benchmark) into
`external/mt-rag-benchmark/` (passage-level corpora unzipped under `corpora/passage_level/`,
`mtrag-human/retrieval_tasks/*/qrels/dev.tsv`, `mtrag-human/generation_tasks/RAG.jsonl`), then:

```bash
python -m taska.prepare_inputs                 # -> data/taskA_dev_input.jsonl
python -m taska.rewrite                        # -> data/rewrites_t5canard.jsonl (already committed)
RW=file:data/rewrites_t5canard.jsonl
python -m taska.run --sources bm25@last        # baseline
python -m taska.run --sources bm25@$RW bm25@weighted bm25@last \
    --rerank cross-encoder/ms-marco-MiniLM-L6-v2 --rerank_query $RW --rerank_depth 50 --write
```

On a new test file: `python -m taska.run --input <test.jsonl> ... --write`.

### ADA (IIIT-H cluster)
`ada/setup_env.sh` (one-time, login node), `ada/encode_dense.sbatch`, `ada/run_pipeline.sbatch`.
Account limits: 1 GPU, 10 CPUs, 6 h per job, 1 running job.
