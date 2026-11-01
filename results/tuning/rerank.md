# Reranking sweep

Grid: 5 candidate pools × 42 rerankers/ensembles × depth × alpha (final score = alpha·minmax(CE) + (1−alpha)·minmax(fusion)); 6160 configs.

- Best on all dev (in-sample): **tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7** = 0.5021
- CV held-out estimate of the selection procedure: **0.5021** (clapnq 0.574 cloud 0.477 fiqa 0.412 govt 0.532)
- Current system (hybrid5_orig / MiniLM-L6-v2@t5 / depth=50 / alpha=1.0): 0.4321
- CV held-out vs current: +0.0700 [+0.0547, +0.0852] p=0.000
- Best vs current: +0.0700 [+0.0547, +0.0852] p=0.000
- Configs statistically tied with the best (95% CI of difference includes 0): 21 of 6160
- Chosen per fold:
  - tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7
  - tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7
  - tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7
  - tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7
  - tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7

## Best config per reranker

| reranker | best config | nDCG@10 | vs overall best |
|---|---|---|---|
| bge-reranker-base@t5 | tuned_ndcg / bge-reranker-base@t5 / depth=100 / alpha=0.3 | 0.4580 | -0.0441 [-0.0552, -0.0334] p=1.000 |
| bge-reranker-v2-m3@t5 | tuned_ndcg / bge-reranker-v2-m3@t5 / depth=50 / alpha=0.5 | 0.4746 | -0.0275 [-0.0356, -0.0195] p=1.000 |
| MiniLM-L12-v2@t5 | tuned_ndcg / MiniLM-L12-v2@t5 / depth=75 / alpha=0.7 | 0.4685 | -0.0337 [-0.0444, -0.0232] p=1.000 |
| MiniLM-L6-v2@last | tuned_r50 / MiniLM-L6-v2@last / depth=30 / alpha=0.6 | 0.4737 | -0.0285 [-0.0406, -0.0165] p=1.000 |
| MiniLM-L6-v2@t5 | tuned_ndcg / MiniLM-L6-v2@t5 / depth=100 / alpha=0.6 | 0.4677 | -0.0345 [-0.0447, -0.0242] p=1.000 |
| MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L6-v2@weighted / depth=30 / alpha=0.3 | 0.4554 | -0.0468 [-0.0579, -0.0354] p=1.000 |
| bge-reranker-base@t5 + bge-reranker-v2-m3@t5 | tuned_ndcg / bge-reranker-base@t5 + bge-reranker-v2-m3@t5 / depth=50 / alpha=0.5 | 0.4705 | -0.0316 [-0.0408, -0.0226] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L12-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L12-v2@t5 / depth=100 / alpha=0.5 | 0.4678 | -0.0343 [-0.0446, -0.0242] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L6-v2@last | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@last / depth=100 / alpha=0.6 | 0.4849 | -0.0172 [-0.0267, -0.0083] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@t5 / depth=50 / alpha=0.5 | 0.4680 | -0.0341 [-0.0439, -0.0243] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@weighted / depth=100 / alpha=0.4 | 0.4680 | -0.0342 [-0.0445, -0.0238] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 / depth=50 / alpha=0.6 | 0.4809 | -0.0213 [-0.0293, -0.0134] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7 | 0.5021 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 / depth=50 / alpha=0.6 | 0.4803 | -0.0218 [-0.0297, -0.0140] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@weighted / depth=50 / alpha=0.6 | 0.4812 | -0.0209 [-0.0295, -0.0125] p=1.000 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@last | tuned_r50 / MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=75 / alpha=0.8 | 0.4935 | -0.0086 [-0.0194, +0.0021] p=0.942 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 / depth=100 / alpha=0.6 | 0.4680 | -0.0341 [-0.0443, -0.0240] p=1.000 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted / depth=100 / alpha=0.6 | 0.4718 | -0.0303 [-0.0409, -0.0198] p=1.000 |
| MiniLM-L6-v2@last + MiniLM-L6-v2@t5 | tuned_ndcg / MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=100 / alpha=0.7 | 0.4933 | -0.0089 [-0.0175, -0.0005] p=0.980 |
| MiniLM-L6-v2@last + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.6 | 0.4802 | -0.0219 [-0.0322, -0.0120] p=1.000 |
| MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted / depth=100 / alpha=0.6 | 0.4744 | -0.0277 [-0.0384, -0.0171] p=1.000 |
| bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 / depth=50 / alpha=0.5 | 0.4749 | -0.0272 [-0.0357, -0.0188] p=1.000 |
| bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last | tuned_ndcg / bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.6 | 0.4906 | -0.0116 [-0.0180, -0.0051] p=1.000 |
| bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 / depth=50 / alpha=0.5 | 0.4767 | -0.0255 [-0.0339, -0.0170] p=1.000 |
| bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-base@t5 + bge-reranker-v2-m3@t5 + MiniLM-L6-v2@weighted / depth=30 / alpha=0.6 | 0.4788 | -0.0234 [-0.0324, -0.0144] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=100 / alpha=0.6 | 0.4896 | -0.0125 [-0.0209, -0.0043] p=0.999 |
| bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 / depth=100 / alpha=0.6 | 0.4733 | -0.0289 [-0.0390, -0.0190] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted / depth=75 / alpha=0.7 | 0.4775 | -0.0246 [-0.0354, -0.0142] p=1.000 |
| bge-reranker-base@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=100 / alpha=0.7 | 0.4890 | -0.0131 [-0.0225, -0.0041] p=0.997 |
| bge-reranker-base@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=100 / alpha=0.7 | 0.4920 | -0.0101 [-0.0206, -0.0002] p=0.977 |
| bge-reranker-base@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-base@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted / depth=100 / alpha=0.7 | 0.4761 | -0.0260 [-0.0377, -0.0148] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.8 | 0.4987 | -0.0034 [-0.0093, +0.0023] p=0.876 |
| bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 / depth=50 / alpha=0.6 | 0.4815 | -0.0207 [-0.0290, -0.0124] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@weighted / depth=50 / alpha=0.7 | 0.4842 | -0.0180 [-0.0267, -0.0097] p=1.000 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=50 / alpha=0.8 | 0.4969 | -0.0052 [-0.0113, +0.0007] p=0.957 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8 | 0.4978 | -0.0043 [-0.0112, +0.0024] p=0.896 |
| bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted / depth=50 / alpha=0.6 | 0.4817 | -0.0205 [-0.0286, -0.0123] p=1.000 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 | tuned_ndcg / MiniLM-L12-v2@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=100 / alpha=0.8 | 0.4907 | -0.0115 [-0.0214, -0.0015] p=0.988 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L12-v2@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=100 / alpha=0.8 | 0.4904 | -0.0117 [-0.0230, -0.0008] p=0.982 |
| MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L12-v2@t5 + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted / depth=100 / alpha=0.6 | 0.4732 | -0.0289 [-0.0388, -0.0192] p=1.000 |
| MiniLM-L6-v2@last + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted | tuned_ndcg / MiniLM-L6-v2@last + MiniLM-L6-v2@t5 + MiniLM-L6-v2@weighted / depth=100 / alpha=0.8 | 0.4885 | -0.0137 [-0.0251, -0.0025] p=0.992 |

Upper bound, reranking with the gold rewrite as query: tuned_ndcg / MiniLM-L6-v2@gold / depth=100 / alpha=0.6 = 0.4964 (vs best -0.0057 [-0.0188, +0.0069] p=0.807)

## Best config per candidate pool

| pool | best config | nDCG@10 |
|---|---|---|
| hybrid5 | hybrid5 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.9 | 0.4772 |
| dense_t5 | dense_t5 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.8 | 0.4792 |
| tuned_ndcg | tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7 | 0.5021 |
| tuned_r50 | tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.8 | 0.4968 |
| hybrid5_orig | hybrid5_orig / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=1.0 | 0.4771 |

## Top configs vs best (paired bootstrap)

| config | nDCG@10 | diff vs best [95% CI] | p(best not better) |
|---|---|---|---|
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7 | 0.5021 | +0.0000 [+0.0000, +0.0000] | 1.000 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.6 | 0.4989 | -0.0032 [-0.0076, +0.0011] | 0.069 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.8 | 0.4987 | -0.0034 [-0.0093, +0.0023] | 0.124 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.8 | 0.4978 | -0.0043 [-0.0112, +0.0024] | 0.104 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.7 | 0.4975 | -0.0046 [-0.0095, +0.0000] | 0.026 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=50 / alpha=0.7 | 0.4975 | -0.0046 [-0.0111, +0.0016] | 0.072 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.8 | 0.4975 | -0.0047 [-0.0096, -0.0000] | 0.025 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7 | 0.4974 | -0.0047 [-0.0100, +0.0001] | 0.029 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=50 / alpha=0.8 | 0.4969 | -0.0052 [-0.0113, +0.0007] | 0.043 |
| tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.8 | 0.4968 | -0.0053 [-0.0129, +0.0020] | 0.081 |
| tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.8 | 0.4967 | -0.0054 [-0.0130, +0.0020] | 0.077 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.8 | 0.4966 | -0.0056 [-0.0113, -0.0001] | 0.024 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.9 | 0.4960 | -0.0061 [-0.0133, +0.0009] | 0.043 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.6 | 0.4959 | -0.0062 [-0.0120, -0.0006] | 0.016 |
| tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.9 | 0.4959 | -0.0063 [-0.0143, +0.0017] | 0.062 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=50 / alpha=0.7 | 0.4958 | -0.0064 [-0.0116, -0.0012] | 0.006 |
| tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.8 | 0.4958 | -0.0064 [-0.0148, +0.0016] | 0.059 |
| tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.9 | 0.4948 | -0.0073 [-0.0158, +0.0012] | 0.046 |
| tuned_r50 / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.7 | 0.4947 | -0.0074 [-0.0146, -0.0004] | 0.019 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@weighted / depth=30 / alpha=0.8 | 0.4945 | -0.0076 [-0.0155, +0.0000] | 0.026 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last + MiniLM-L6-v2@t5 / depth=50 / alpha=0.6 | 0.4943 | -0.0078 [-0.0137, -0.0020] | 0.004 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.9 | 0.4941 | -0.0081 [-0.0164, +0.0002] | 0.027 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L6-v2@last / depth=30 / alpha=0.9 | 0.4938 | -0.0083 [-0.0159, -0.0009] | 0.012 |
| tuned_ndcg / bge-reranker-v2-m3@t5 + MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=50 / alpha=0.6 | 0.4937 | -0.0085 [-0.0140, -0.0031] | 0.001 |
| tuned_r50 / MiniLM-L12-v2@t5 + MiniLM-L6-v2@last / depth=75 / alpha=0.8 | 0.4935 | -0.0086 [-0.0194, +0.0021] | 0.058 |

## Best config, all metrics

| metric | value |
|---|---|
| nDCG@1 | 0.4556 |
| nDCG@3 | 0.4162 |
| nDCG@5 | 0.4513 |
| nDCG@10 | 0.5021 |
| R@10 | 0.6032 |
| R@50 | 0.7552 |
| R@100 | 0.7552 |
