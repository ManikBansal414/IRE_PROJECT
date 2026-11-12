# MTRAGEval (SemEval-2026 Task 8) — Task A: Retrieval

The final systems for Tasks A, B and C and their scores are in [results/BENCHMARKS.md](results/BENCHMARKS.md);
this file documents the Task A development runs.

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
| 7 | #6 + MiniLM cross-encoder rerank of top 50 | 0.349 | 0.313 | 0.334 | 0.366 | 0.427 |
| 8 | Dense (BGE-base-en-v1.5), last question | 0.323 | 0.305 | 0.327 | 0.370 | 0.456 |
| 9 | Dense, T5-CANARD rewrite | 0.345 | 0.322 | 0.348 | 0.393 | 0.485 |
| 10 | Dense, weighted history | 0.224 | 0.221 | 0.252 | 0.288 | 0.371 |
| 11 | RRF hybrid: dense #9 + #10, BM25 #4 + #5 | 0.308 | 0.282 | 0.312 | 0.362 | 0.464 |
| 12 | #11 + bge-reranker-base rerank of top 100 | 0.288 | 0.287 | 0.317 | 0.360 | 0.455 |
| 13 | RRF hybrid: dense #9 + #8, BM25 #4 + #5 + #1 | 0.306 | 0.282 | 0.312 | 0.353 | 0.442 |
| 14 | #9 + MiniLM rerank of top 50 | 0.378 | 0.350 | 0.384 | 0.428 | 0.522 |
| 15 | #13 + MiniLM rerank of top 50 (mid-submission system) | 0.391 | 0.354 | 0.386 | 0.432 | 0.521 |
| **16** | **Tuned pipeline: 8-source weighted RRF (BM25 k1=2.2, b=0.5; k=5) + bge-reranker-v2-m3 & MiniLM ensemble (final; `--preset best`, [results/tuning/README.md](results/tuning/README.md))** | **0.456** | **0.416** | **0.451** | **0.502** | **0.603** |
| — | *BM25 on gold human rewrites (upper bound only, not available at test time)* | 0.203 | 0.196 | 0.216 | 0.254 | 0.330 |

Per-domain numbers: [results/results_log.md](results/results_log.md) and `results/*.metrics.json`.
Submission-format output for the final system (#16): `results/best_system/best.predictions.jsonl`;
mid-submission system (#15): `results/hybrid5__rr_minilm50.predictions.jsonl`;
BM25-only (#7): `results/bm25rrf3__rr_minilm50.predictions.jsonl`. All pass the official
`format_checker.py --mode retrieval_taska`.

Observations:
- Dense retrieval beats BM25 by a wide margin (0.370 vs 0.224 on the last question alone).
- The T5 rewrite helps both retrievers; the "weighted history" query helps BM25 but hurts dense.
- RRF fusion by itself does not beat the best single dense run, but the fused candidate pool
  gives the reranker the most to work with (#15 > #14).
- bge-reranker-base did not help in run #12 (0.362 → 0.360), but that run used a different pool, depth 100 and no
  score interpolation, so it is not a fair comparison; in the tuning sweep it reaches 0.458 (results/tuning/rerank.md).
  The MS MARCO MiniLM cross-encoder adds +0.04–0.12 on the same pools.

Rows #8–#16 were run on ADA (GPU); on the laptop CPU, dense encoding ran at ~8 passages/s
(~12 h for the 366k passages) vs ~68 passages/s on a GTX 1080 Ti.

## Pipeline

```
conversation ──► query variants ──► BM25 / dense retrieval (top 100 each)
                 (last turn,          │
                  weighted history,   ▼
                  T5 rewrite)     RRF fusion ──► cross-encoder rerank (top 50) ──► top 10
```

- `taska/queries.py` — query construction from the conversation (no gold info used).
- `taska/rewrite.py` — conversational query rewriting with `castorini/t5-base-canard`.
- `taska/bm25.py` — BM25 (bm25s, English stemming + stopwords), one index per domain; k1=0.9, b=0.4 in rows
  #1–#15, k1=2.2, b=0.5 (tuned) in the `best` preset.
- `taska/dense.py` — sentence-transformers encoder + exact FAISS search; resumable sharded encoding.
- `taska/fusion.py` — (weighted) Reciprocal Rank Fusion; k=60 and equal weights in rows #6–#15, k=5 and tuned weights in the `best` preset.
- `taska/rerank.py` — cross-encoder reranking.
- `taska/evaluate.py` — nDCG/Recall @1,3,5,10 via pytrec_eval, same aggregation as the official script.
- `taska/run.py` — runs a pipeline, scores it, writes official-format predictions.
- `taska/cache.py`, `taska/sweep.py` — cached runs and the CPU-side parameter sweeps behind results/tuning/.

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
python -m taska.run --preset best --write    # tuned system (results/tuning/README.md), nDCG@10 0.502
RW=file:data/rewrites_t5canard.jsonl
python -m taska.run --sources bm25@last        # baseline
python -m taska.run --sources bm25@$RW bm25@weighted bm25@last \
    --rerank cross-encoder/ms-marco-MiniLM-L6-v2 --rerank_query $RW --rerank_depth 50 --write
```

On a new test file: `python -m taska.run --input <test.jsonl> ... --write`.

### ADA (IIIT-H cluster)
`ada/setup_env.sh` (one-time, login node), then `sbatch ada/full_run.sbatch` (encodes all corpora,
~90 min, then runs `ada/run_pipeline.sbatch`), then `sbatch ada/extra_runs.sbatch` (MiniLM-reranked
runs #13–#15). The tuned system: `sbatch ada/best_run.sbatch`. Account limits: 1 GPU, 10 CPUs, 6 h per job, only 1 submitted job at a time
(so jobs cannot be chained with `--dependency`).
