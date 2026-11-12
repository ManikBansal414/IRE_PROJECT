# Task A parameter tuning (run 2026-10-31 → 2026-11-01, Ada account cs4406.04)

All numbers: MTRAG dev set, 777 scored questions (110 conversations). Metric for selection: nDCG@10
(the official script may rank by a smaller cutoff; the chosen system is also best on nDCG@1/3/5, see `rerank.md`).

## Method

- **No GPU in the sweeps.** Ada jobs cache (1) the top-100 of every retriever × query variant, (2) BM25 runs
  for a k1/b grid, (3) cross-encoder scores for candidate pools. Sweeps then run on CPU (`taska/sweep.py`).
- **Selection by 5-fold cross-validation over conversations** (all turns of a conversation in one fold):
  parameters chosen on 4 folds, scored on the 5th. The *CV held-out* number estimates how the tuning
  procedure generalises; the *in-sample* number is optimistic.
- **Significance:** paired bootstrap over questions (10,000 resamples). "Tied" = 95% CI of the difference
  to the best config includes 0.
- **Checks:** sweep metrics reproduce the README (dense@t5 0.3932, hybrid5 0.3531, dense@last 0.3705)
  and the submitted system (0.4321 vs 0.4315 logged; difference from score ties). Vectorised RRF verified
  against `taska/fusion.rrf` on synthetic data.

## 1. BM25 (k1, b) — `bm25.md`

Grid: k1 ∈ {0.6, 0.9, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0} × b ∈ {0.3, 0.4, 0.5, 0.6, 0.75}; 33 of the 40 points were run
(b = 0.5 only for k1 ≥ 1.5, b = 0.75 only for k1 ≤ 1.5),
objective = mean nDCG@10 over the 5 query variants.

| setting | mean nDCG@10 |
|---|---|
| **k1=2.2, b=0.5 (chosen)** | **0.2324** |
| k1=0.9, b=0.4 (previous default) | 0.2214 |

- Chosen vs default: **+0.0109** [+0.0049, +0.0169], p<0.001. CV picked k1=2.2 in all 5 folds.
- The optimum is interior to the grid; 12 of 33 settings are tied with it (flat plateau around k1 1.8–2.6, b 0.4–0.6).

## 2. Fusion (source weights, RRF k, depth) — `fusion.md`

Search: coordinate ascent over 10 sources (BM25 and bge-base × {last, last2, weighted, all, t5}),
weights {0, 0.25, 0.5, 1, 1.5, 2, 3}, RRF k {1, 2, 3, 5, 10, 20, 30, 60, 100}, depth {20, 30, 50, 100}, 4 starts.

| system (no reranker) | nDCG@10 |
|---|---|
| hybrid5, equal weights, k=60 (previous) | 0.3531 |
| dense@t5 alone | 0.3932 |
| **tuned fusion, CV held-out** | **0.4338** (in-sample 0.4429) |

- CV held-out vs hybrid5 (tuned BM25): **+0.075** [+0.059, +0.091]; vs dense@t5: **+0.041** [+0.028, +0.054]; both p<0.001.
- Pattern in every fold: dense runs weighted 2–3× over BM25, small RRF k (5–10), full depth 100.
- Chosen config (`tuned_ndcg`): bge-base@t5 ×2, bge-base@last ×1.5, bm25@last ×1, bge-base@last2 ×0.5,
  bm25@all ×0.5, bge-base@weighted ×0.25, bm25@weighted ×0.25, bm25@last2 ×0.25; k=5; depth=100.
- A second pool tuned for R@50 (`tuned_r50`, CV R@50 0.758 vs 0.709 for hybrid5) was also passed to reranking.

## 3. Reranking — `rerank.md`

Grid: 5 candidate pools × 42 rerankers/ensembles × depth {10, 20, 30, 50, 75, 100} × alpha {1.0 … 0.3}
= 6,160 configs. Final score = alpha·minmax(CE) + (1−alpha)·minmax(fusion score); ensembles average the
min-max-normalised CE scores. Rerankers: MiniLM-L6 (queries: T5 rewrite, last question, weighted history),
MiniLM-L12 (T5), bge-reranker-base (T5), bge-reranker-v2-m3 (T5; scored on the top-50 pool only).

Best config per single reranker (each on its best pool/depth/alpha):

| reranker | nDCG@10 |
|---|---|
| bge-reranker-base @T5 | 0.4580 |
| MiniLM-L6 @T5 | 0.4677 |
| MiniLM-L12 @T5 | 0.4685 |
| MiniLM-L6 @last question | 0.4737 |
| bge-reranker-v2-m3 @T5 | 0.4746 |
| *MiniLM-L6 @gold rewrite (upper bound, not usable at test time)* | *0.4964* |

Pairing a T5-rewrite reranker with a last-question reranker gives the largest gain:

| config | nDCG@10 |
|---|---|
| MiniLM-L12 @T5 + MiniLM-L6 @last | 0.4935 |
| **v2-m3 @T5 + MiniLM-L6 @last (best)** | **0.5021** |
| 3-way ensembles (best: v2-m3 @T5 + MiniLM-L12 @T5 + MiniLM-L6 @last) | 0.4987 (tied, not better) |

## Recommended system

`tuned_ndcg` fusion (section 2, with BM25 k1=2.2, b=0.5) → top 50 → score with
**bge-reranker-v2-m3 (query: T5 rewrite)** and **MiniLM-L6 (query: last user question)**, average the
min-max-normalised scores → final = 0.7·CE + 0.3·minmax(fusion score) → top 10.

| | previous submitted system | recommended |
|---|---|---|
| nDCG@1 | 0.391 | 0.456 |
| nDCG@3 | 0.354 | 0.416 |
| nDCG@5 | 0.386 | 0.451 |
| **nDCG@10** | **0.432** | **0.502** |
| R@10 | 0.521 | 0.603 |
| nDCG@10 ClapNQ / Cloud / FiQA / Govt | 0.516 / 0.403 / 0.326 / 0.467 | 0.574 / 0.477 / 0.412 / 0.532 |

- **Gain: +0.070 nDCG@10** [+0.055, +0.085], p<0.001.
- **CV held-out = 0.5021:** all 5 folds independently chose this exact config, so the choice does not depend on which questions it was tuned on.
- The gold-rewrite figure (0.4964) was measured with a single MiniLM-L6 reranker, so it is an upper bound for that
  reranker only, not for the two-query ensemble (which scores above it). Whether an LLM query rewriter would add
  anything on top of the ensemble was not measured.

## How close to best is this?

Within the searched space, the evidence is:
1. Every tuned parameter's optimum is interior: BM25 k1/b, RRF k (5–10 chosen, 1–3 and 20+ worse), alpha (0.7, lower at 0.6 and 0.8).
2. CV selection is stable (same config in 5/5 folds for BM25 and for the final reranking setup).
3. Neighbouring configs form a plateau of statistically tied results, so small parameter changes do not matter.

Not covered (would need more GPU time):
- **v2-m3 depth > 50.** Depth is at the edge of what was scored. The trend is flattening (depth 30 → 50: +0.005, within noise), and for MiniLM depths 50/75/100 are tied, so the expected gain is small.
- **v2-m3 with the last-question query.** For MiniLM, the T5 + last pairing was the main win, so v2-m3 @last could help.
- Other dense encoders (bge-large, e5) and an LLM query rewriter were not tested.

## Reproduce

```bash
# Ada
sbatch ada/tune_cache.sbatch        # encode + cached runs + BM25 grid (k1 ≤ 1.5), ~1h50
sbatch ada/tune_bm25grid2.sbatch    # BM25 grid k1 1.5–3.0, ~30 min, no GPU
# local (after copying work/cache/runs, work/cache/bm25grid and the qrels back)
python -m taska.sweep bm25
python -m taska.sweep fusion        # ~8 min
python -m taska.sweep pool          # -> work/cache/pool{50,100}.json, copy to Ada
# Ada
sbatch ada/tune_rerank.sbatch       # 7 cross-encoder caches, 2h31
# local (after copying work/cache/rerank back)
python -m taska.sweep rerank        # ~7 min
```

The recommended system is `python -m taska.run --preset best` (`sbatch ada/best_run.sbatch` on Ada).
Offline check of the wired-in code against these caches: nDCG@10 0.5021 (identical to the sweep).
