
```
2026-11-06 23:20  {'preset': 'best', 'input': 'data/taskA_dev_input.jsonl', 'sources': ['bm25@last', 'bm25@last2', 'bm25@weighted', 'bm25@all', 'dense:BAAI/bge-base-en-v1.5@last', 'dense:BAAI/bge-base-en-v1.5@last2', 'dense:BAAI/bge-base-en-v1.5@weighted', 'dense:BAAI/bge-base-en-v1.5@file:data/rewrites_t5canard.jsonl'], 'weights': [1, 0.25, 0.25, 0.5, 1.5, 0.5, 0.25, 2], 'rrf_k': 5, 'depth': 100, 'bm25_k1': 2.2, 'bm25_b': 0.5, 'rerank': ['BAAI/bge-reranker-v2-m3@file:data/rewrites_t5canard.jsonl', 'cross-encoder/ms-marco-MiniLM-L6-v2@last'], 'rerank_query': None, 'rerank_depth': 50, 'rerank_batch': 16, 'alpha': 0.7, 'name': 'best', 'write': True, 'out': 'results/best_system', 'overwrite': False}
== best ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.5192   0.4831   0.5149   0.5737   0.2117   0.4446   0.5460   0.6890
cloud     188   0.4415   0.4000   0.4315   0.4770   0.2068   0.3779   0.4583   0.5621
fiqa      180   0.3778   0.3237   0.3602   0.4122   0.1602   0.2946   0.3878   0.5106
govt      201   0.4726   0.4448   0.4854   0.5321   0.2135   0.4141   0.5218   0.6357
all       777   0.4556   0.4162   0.4513   0.5021   0.1990   0.3858   0.4819   0.6032
```
