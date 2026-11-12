# Fusion sweep (BM25 params: k1=2.2_b=0.5)

Search: coordinate ascent over weights [0, 0.25, 0.5, 1, 1.5, 2, 3] per source, RRF k [1, 2, 3, 5, 10, 20, 30, 60, 100], per-source depth [20, 30, 50, 100]; 4 starting points. Selection by 5-fold CV over conversations.

## Single sources (nDCG@5 / R@50 / R@100)

| source | nDCG@5 | R@50 | R@100 |
|---|---|---|---|
| bm25@last | 0.2023 | 0.4963 | 0.5637 |
| bm25@last2 | 0.2066 | 0.4953 | 0.5939 |
| bm25@weighted | 0.2215 | 0.5418 | 0.6263 |
| bm25@all | 0.1668 | 0.4536 | 0.5501 |
| bm25@t5 | 0.2151 | 0.5194 | 0.5973 |
| bge-base@last | 0.3275 | 0.6558 | 0.7161 |
| bge-base@last2 | 0.2818 | 0.6200 | 0.6990 |
| bge-base@weighted | 0.2519 | 0.6025 | 0.6864 |
| bge-base@all | 0.1950 | 0.5254 | 0.6035 |
| bge-base@t5 | 0.3477 | 0.6863 | 0.7550 |

## Objective nDCG@5

- Tuned on all dev (in-sample, optimistic): **0.3985** with {bge-base@all×0.25, bge-base@last×3, bge-base@last2×1.5, bge-base@t5×3, bge-base@weighted×0.5, bm25@all×0.5, bm25@last×2, bm25@weighted×0.25} k=10 depth=50
- CV held-out estimate of the tuning procedure: **0.3875** (clapnq 0.462 cloud 0.368 fiqa 0.309 govt 0.399)
- Current hybrid5 (equal weights, k=60): 0.3206; dense@t5 alone: 0.3477
- CV held-out vs hybrid5: +0.0669 [+0.0503, +0.0841] p=0.000
- CV held-out vs dense@t5: +0.0398 [+0.0252, +0.0548] p=0.000
- Configs chosen per fold:
  - {bge-base@all×0.25, bge-base@last×3, bge-base@last2×2, bge-base@t5×3, bge-base@weighted×0.5, bm25@last×1.5, bm25@last2×1, bm25@t5×0.25, bm25@weighted×0.5} k=10 depth=100
  - {bge-base@all×0.25, bge-base@last×3, bge-base@last2×1.5, bge-base@t5×3, bge-base@weighted×0.5, bm25@last×2, bm25@last2×0.25, bm25@weighted×0.5} k=10 depth=50
  - {bge-base@last×1, bge-base@last2×0.5, bge-base@t5×1.5, bge-base@weighted×0.25, bm25@last×1} k=10 depth=100
  - {bge-base@last×3, bge-base@last2×1.5, bge-base@t5×3, bge-base@weighted×0.5, bm25@all×0.5, bm25@last×1, bm25@weighted×0.25} k=10 depth=100
  - {bge-base@all×1, bge-base@last×3, bge-base@last2×2, bge-base@t5×3, bge-base@weighted×1, bm25@last×3, bm25@last2×0.25, bm25@weighted×0.25} k=10 depth=100

## Check: the fusion setting chosen with nDCG@10

- Chosen with nDCG@10: nDCG@10 choice: nDCG@5 0.3938
- Best by nDCG@5: best: 0.3985
- Difference: -0.0047 [-0.0110, +0.0015] -> **tied with the best (95% CI includes 0)**
