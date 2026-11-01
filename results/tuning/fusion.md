# Fusion sweep (BM25 params: k1=2.2_b=0.5)

Search: coordinate ascent over weights [0, 0.25, 0.5, 1, 1.5, 2, 3] per source, RRF k [1, 2, 3, 5, 10, 20, 30, 60, 100], per-source depth [20, 30, 50, 100]; 4 starting points. Selection by 5-fold CV over conversations.

## Single sources (nDCG@10 / R@50 / R@100)

| source | nDCG@10 | R@50 | R@100 |
|---|---|---|---|
| bm25@last | 0.2337 | 0.4963 | 0.5637 |
| bm25@last2 | 0.2352 | 0.4953 | 0.5939 |
| bm25@weighted | 0.2535 | 0.5418 | 0.6263 |
| bm25@all | 0.1934 | 0.4536 | 0.5501 |
| bm25@t5 | 0.2459 | 0.5194 | 0.5973 |
| bge-base@last | 0.3705 | 0.6558 | 0.7161 |
| bge-base@last2 | 0.3192 | 0.6200 | 0.6990 |
| bge-base@weighted | 0.2884 | 0.6025 | 0.6864 |
| bge-base@all | 0.2266 | 0.5254 | 0.6035 |
| bge-base@t5 | 0.3932 | 0.6863 | 0.7550 |

## Objective nDCG@10

- Tuned on all dev (in-sample, optimistic): **0.4429** with {bge-base@last×1.5, bge-base@last2×0.5, bge-base@t5×2, bge-base@weighted×0.25, bm25@all×0.5, bm25@last×1, bm25@last2×0.25, bm25@weighted×0.25} k=5 depth=100
- CV held-out estimate of the tuning procedure: **0.4338** (clapnq 0.503 cloud 0.406 fiqa 0.359 govt 0.456)
- Current hybrid5 (equal weights, k=60): 0.3593; dense@t5 alone: 0.3932
- CV held-out vs hybrid5: +0.0746 [+0.0587, +0.0910] p=0.000
- CV held-out vs dense@t5: +0.0406 [+0.0276, +0.0541] p=0.000
- Configs chosen per fold:
  - {bge-base@all×0.25, bge-base@last×3, bge-base@last2×1, bge-base@t5×3, bge-base@weighted×1, bm25@last×2, bm25@last2×0.5, bm25@t5×0.25, bm25@weighted×0.25} k=10 depth=100
  - {bge-base@last×1, bge-base@last2×0.5, bge-base@t5×1, bge-base@weighted×0.25, bm25@last×0.5, bm25@last2×0.25} k=5 depth=100
  - {bge-base@all×0.25, bge-base@last×2, bge-base@last2×0.5, bge-base@t5×3, bge-base@weighted×0.5, bm25@last×2} k=10 depth=100
  - {bge-base@last×2, bge-base@last2×1, bge-base@t5×2, bge-base@weighted×0.5, bm25@all×0.25, bm25@last×1, bm25@t5×0.25, bm25@weighted×0.25} k=5 depth=100
  - {bge-base@all×1, bge-base@last×3, bge-base@last2×0.5, bge-base@t5×3, bge-base@weighted×1, bm25@all×0.5, bm25@last×1.5, bm25@last2×0.25, bm25@t5×0.5, bm25@weighted×0.5} k=10 depth=100

## Objective R@50

- Tuned on all dev (in-sample, optimistic): **0.7719** with {bge-base@all×1.5, bge-base@last×3, bge-base@last2×1, bge-base@t5×2, bge-base@weighted×1, bm25@all×1, bm25@last×0.25, bm25@last2×0.5, bm25@t5×1.5, bm25@weighted×1} k=20 depth=100
- CV held-out estimate of the tuning procedure: **0.7579** (clapnq 0.838 cloud 0.714 fiqa 0.667 govt 0.799)
- Current hybrid5 (equal weights, k=60): 0.7094; dense@t5 alone: 0.6863
- CV held-out vs hybrid5: +0.0485 [+0.0319, +0.0653] p=0.000
- CV held-out vs dense@t5: +0.0716 [+0.0524, +0.0912] p=0.000
- Configs chosen per fold:
  - {bge-base@all×0.5, bge-base@last×1, bge-base@last2×0.25, bge-base@t5×1, bge-base@weighted×0.5, bm25@all×0.25, bm25@last×0.5, bm25@last2×0.25, bm25@t5×0.5} k=30 depth=100
  - {bge-base@all×1, bge-base@last×3, bge-base@t5×1.5, bge-base@weighted×1, bm25@all×1, bm25@last×0.5, bm25@last2×0.25, bm25@t5×1, bm25@weighted×1} k=20 depth=100
  - {bge-base@all×1, bge-base@last×1, bge-base@last2×0.25, bge-base@t5×1.5, bge-base@weighted×0.5, bm25@all×0.25, bm25@last×0.25, bm25@last2×0.25, bm25@t5×0.5, bm25@weighted×0.25} k=10 depth=100
  - {bge-base@all×0.5, bge-base@last×1, bge-base@last2×0.5, bge-base@t5×1, bm25@all×0.25, bm25@last×0.25, bm25@t5×0.25, bm25@weighted×0.5} k=10 depth=100
  - {bge-base@all×1, bge-base@last×1, bge-base@t5×1, bm25@last×0.25, bm25@weighted×1} k=20 depth=100
