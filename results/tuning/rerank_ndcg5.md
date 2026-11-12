# Reranking sweep

Grid: 5 candidate pools × 42 rerankers/ensembles × depth × alpha (final score = alpha·minmax(CE) + (1−alpha)·minmax(fusion)); 6160 configs.

- Best on all dev (in-sample): **tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8** = 0.4530
- CV held-out estimate of the selection procedure: **0.4530** (clapnq 0.510 cloud 0.438 fiqa 0.352 govt 0.499)
- Current system (hybrid5_orig / MiniLM-L6-v2@t5 / depth=50 / alpha=1.0): 0.3867
- CV held-out vs current: +0.0663 [+0.0497, +0.0831] p=0.000
- Best vs current: +0.0663 [+0.0497, +0.0831] p=0.000
- Configs statistically tied with the best (95% CI of difference includes 0): 46 of 6160
- Chosen per fold:
  - tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8
  - tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8
  - tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8
  - tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8
  - tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8

## Best config per reranker

| reranker | best config | nDCG@5 | vs overall best |
|---|---|---|---|
| bge-reranker-base@t5 | tuned_ndcg / bge-reranker-base@t5 / depth=100 / alpha=0.3 | 0.4119 | -0.0412 [-0.0543, -0.0280] p=1.000 |
| bge-reranker-v2-m3@t5 | tuned_ndcg / bge-reranker-v2-m3@t5 / depth=50 / alpha=0.5 | 0.4220 | -0.0310 [-0.0421, -0.0194] p=1.000 |
| MiniLM-L12-v2@t5 | tuned_ndcg / MiniLM-L12-v2@t5 / depth=20 / alpha=0.6 | 0.4213 | -0.0317 [-0.0446, -0.0190] p=1.000 |
| MiniLM-L6-v2@last | tuned_ndcg / MiniLM-L6-v2@last / depth=10 / alpha=0.5 | 0.4247 | -0.0284 [-0.0415, -0.0156] p=1.000 |
| MiniLM-L6-v2@t5 | tuned_ndcg / MiniLM-L6-v2@t5 / depth=100 / alpha=0.7 | 0.4189 | -0.0341 [-0.0470, -0.0209] p=1.000 |
| MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L6-v2@weighted / depth=100 / alpha=0.3 | 0.4061 | -0.0469 [-0.0596, -0.0341] p=1.000 |
| bge-reranker-base@t5 + bge-reranker-v2-m3@t5 | tuned_ndcg / bge-reranker-base@t5 + bge-reranker-v2-m3@t5 / depth=20 / alpha=0.4 | 0.4210 | -0.0320 [-0.0440, -0.0199] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L12-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L12-v2@t5 / depth=100 / alpha=0.5 | 0.4167 | -0.0363 [-0.0489, -0.0237] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L6-v2@last | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@last / depth=10 / alpha=0.6 | 0.4332 | -0.0198 [-0.0318, -0.0080] p=0.999 |
| bge-reranker-base@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@t5 / depth=100 / alpha=0.4 | 0.4175 | -0.0355 [-0.0479, -0.0229] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@weighted / depth=100 / alpha=0.4 | 0.4197 | -0.0334 [-0.0452, -0.0214] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 / depth=50 / alpha=0.7 | 0.4331 | -0.0200 [-0.0317, -0.0080] p=0.999 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7 | 0.4513 | -0.0018 [-0.0100, +0.0066] p=0.661 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 / depth=50 / alpha=0.6 | 0.4323 | -0.0208 [-0.0317, -0.0098] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@weighted / depth=50 / alpha=0.7 | 0.4308 | -0.0222 [-0.0310, -0.0133] p=1.000 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@last | tuned_ndcg / MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=20 / alpha=0.7 | 0.4394 | -0.0137 [-0.0252, -0.0024] p=0.991 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 / depth=20 / alpha=0.6 | 0.4203 | -0.0327 [-0.0456, -0.0200] p=1.000 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted / depth=75 / alpha=0.5 | 0.4192 | -0.0338 [-0.0449, -0.0226] p=1.000 |
| MiniLM-L6-v2@last + MiniLM-L6-v2@t5 | tuned_ndcg / MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=100 / alpha=0.7 | 0.4432 | -0.0098 [-0.0210, +0.0011] p=0.960 |
| MiniLM-L6-v2@last + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=100 / alpha=0.7 | 0.4311 | -0.0219 [-0.0328, -0.0112] p=1.000 |
| MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted / depth=100 / alpha=0.6 | 0.4229 | -0.0301 [-0.0409, -0.0191] p=1.000 |
| bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 / depth=50 / alpha=0.6 | 0.4264 | -0.0266 [-0.0381, -0.0150] p=1.000 |
| bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last | tuned_ndcg / bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=20 / alpha=0.7 | 0.4420 | -0.0110 [-0.0216, -0.0004] p=0.979 |
| bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 / depth=20 / alpha=0.6 | 0.4252 | -0.0279 [-0.0395, -0.0161] p=1.000 |
| bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@weighted / depth=50 / alpha=0.6 | 0.4280 | -0.0251 [-0.0356, -0.0142] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=100 / alpha=0.6 | 0.4349 | -0.0181 [-0.0293, -0.0072] p=0.999 |
| bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 / depth=20 / alpha=0.6 | 0.4203 | -0.0327 [-0.0449, -0.0204] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted / depth=30 / alpha=0.6 | 0.4241 | -0.0289 [-0.0397, -0.0182] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=100 / alpha=0.6 | 0.4370 | -0.0160 [-0.0275, -0.0050] p=0.997 |
| bge-reranker-base@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=100 / alpha=0.6 | 0.4398 | -0.0132 [-0.0238, -0.0027] p=0.993 |
| bge-reranker-base@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted / depth=100 / alpha=0.7 | 0.4224 | -0.0306 [-0.0430, -0.0186] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7 | 0.4494 | -0.0036 [-0.0124, +0.0055] p=0.782 |
| bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 / depth=50 / alpha=0.8 | 0.4331 | -0.0200 [-0.0324, -0.0075] p=0.999 |
| bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted / depth=20 / alpha=0.7 | 0.4332 | -0.0198 [-0.0299, -0.0098] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=50 / alpha=0.7 | 0.4482 | -0.0048 [-0.0139, +0.0043] p=0.849 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8 | 0.4530 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8 | 0.4337 | -0.0193 [-0.0283, -0.0110] p=1.000 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 | tuned_ndcg / MiniLM-L12-v2@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=100 / alpha=0.8 | 0.4361 | -0.0169 [-0.0289, -0.0049] p=0.997 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L12-v2@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=100 / alpha=0.8 | 0.4420 | -0.0110 [-0.0209, -0.0013] p=0.987 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted / depth=20 / alpha=0.6 | 0.4226 | -0.0304 [-0.0412, -0.0195] p=1.000 |
| MiniLM-L6-v2@last + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L6-v2@last + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted / depth=100 / alpha=0.8 | 0.4400 | -0.0130 [-0.0230, -0.0031] p=0.995 |

Upper bound, reranking with the gold rewrite as query: tuned_ndcg / MiniLM-L6-v2@gold / depth=100 / alpha=0.7 = 0.4423 (vs best -0.0107 [-0.0259, +0.0039] p=0.924)

## Best config per candidate pool

| pool | best config | nDCG@5 |
|---|---|---|
| hybrid5 | hybrid5 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.9 | 0.4325 |
| dense_t5 | dense_t5 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7 | 0.4347 |
| tuned_ndcg | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8 | 0.4530 |
| tuned_r50 | tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.8 | 0.4482 |
| hybrid5_orig | hybrid5_orig / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.9 | 0.4348 |

## Top configs vs best (paired bootstrap)

| config | nDCG@5 | diff vs best [95% CI] | p(best not better) |
|---|---|---|---|
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8 | 0.4530 | +0.0000 [+0.0000, +0.0000] | 1.000 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7 | 0.4513 | -0.0018 [-0.0100, +0.0066] | 0.339 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.7 | 0.4501 | -0.0030 [-0.0088, +0.0030] | 0.158 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7 | 0.4494 | -0.0036 [-0.0124, +0.0055] | 0.218 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.7 | 0.4492 | -0.0039 [-0.0130, +0.0053] | 0.199 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=20 / alpha=0.7 | 0.4488 | -0.0042 [-0.0137, +0.0054] | 0.186 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.6 | 0.4485 | -0.0045 [-0.0138, +0.0048] | 0.165 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.8 | 0.4483 | -0.0047 [-0.0136, +0.0039] | 0.147 |
| tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.8 | 0.4482 | -0.0048 [-0.0141, +0.0044] | 0.151 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.8 | 0.4482 | -0.0048 [-0.0141, +0.0044] | 0.157 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=50 / alpha=0.7 | 0.4482 | -0.0048 [-0.0139, +0.0043] | 0.151 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=20 / alpha=0.8 | 0.4480 | -0.0050 [-0.0145, +0.0043] | 0.146 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=30 / alpha=0.8 | 0.4480 | -0.0050 [-0.0103, -0.0001] | 0.022 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.6 | 0.4479 | -0.0052 [-0.0150, +0.0048] | 0.143 |
| tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.9 | 0.4478 | -0.0052 [-0.0154, +0.0048] | 0.151 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.8 | 0.4476 | -0.0054 [-0.0147, +0.0037] | 0.125 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.9 | 0.4475 | -0.0055 [-0.0118, +0.0006] | 0.039 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.9 | 0.4472 | -0.0058 [-0.0165, +0.0047] | 0.138 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=20 / alpha=0.6 | 0.4471 | -0.0059 [-0.0161, +0.0042] | 0.119 |
| tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.8 | 0.4470 | -0.0060 [-0.0149, +0.0029] | 0.090 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=10 / alpha=0.8 | 0.4468 | -0.0062 [-0.0168, +0.0045] | 0.127 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=10 / alpha=0.7 | 0.4460 | -0.0070 [-0.0177, +0.0037] | 0.097 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=30 / alpha=0.7 | 0.4460 | -0.0070 [-0.0166, +0.0024] | 0.072 |
| tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7 | 0.4458 | -0.0072 [-0.0165, +0.0017] | 0.057 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.9 | 0.4457 | -0.0073 [-0.0178, +0.0030] | 0.083 |

## Best config, all metrics

| metric | value |
|---|---|
| nDCG@1 | 0.4466 |
| nDCG@3 | 0.4154 |
| nDCG@5 | 0.4530 |
| nDCG@10 | 0.4978 |
| R@10 | 0.5982 |
| R@50 | 0.7552 |
| R@100 | 0.7552 |

## Check: the reranking setup chosen with nDCG@10

- Chosen with nDCG@10: tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7: nDCG@5 0.4513
- Best by nDCG@5: tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8: 0.4530
- Difference: -0.0018 [-0.0100, +0.0066] -> **tied with the best (95% CI includes 0)**
