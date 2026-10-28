
```
2026-10-27 16:56  {'input': 'C:\\Users\\hp\\OneDrive\\Desktop\\IRE_PROJECT\\IRE_PROJECT\\data\\taskA_dev_input.jsonl', 'sources': ['bm25@last'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': None, 'write': False}
== bm25_last ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.2212   0.2085   0.2286   0.2686   0.0898   0.1944   0.2529   0.3508
cloud     188   0.1489   0.1552   0.1774   0.2037   0.0736   0.1553   0.2112   0.2725
fiqa      180   0.0833   0.0894   0.1046   0.1208   0.0368   0.0869   0.1209   0.1605
govt      201   0.2488   0.2318   0.2577   0.2877   0.1153   0.2119   0.2763   0.3449
all       777   0.1789   0.1740   0.1950   0.2236   0.0802   0.1646   0.2183   0.2862
```

```
2026-10-27 16:56  {'input': 'C:\\Users\\hp\\OneDrive\\Desktop\\IRE_PROJECT\\IRE_PROJECT\\data\\taskA_dev_input.jsonl', 'sources': ['bm25@all'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': None, 'write': False}
== bm25_all ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.2115   0.1869   0.2002   0.2356   0.0756   0.1691   0.2185   0.3063
cloud     188   0.1011   0.1086   0.1343   0.1452   0.0439   0.1134   0.1666   0.1938
fiqa      180   0.0833   0.0766   0.0878   0.0958   0.0375   0.0732   0.0996   0.1190
govt      201   0.1891   0.1900   0.2072   0.2488   0.0982   0.1869   0.2255   0.3194
all       777   0.1493   0.1432   0.1600   0.1848   0.0649   0.1380   0.1802   0.2391
```

```
2026-10-27 16:56  {'input': 'C:\\Users\\hp\\OneDrive\\Desktop\\IRE_PROJECT\\IRE_PROJECT\\data\\taskA_dev_input.jsonl', 'sources': ['bm25@last2'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': None, 'write': False}
== bm25_last2 ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.1923   0.2014   0.2165   0.2594   0.0731   0.1973   0.2469   0.3517
cloud     188   0.1649   0.1487   0.1732   0.1915   0.0771   0.1427   0.1963   0.2388
fiqa      180   0.1500   0.1116   0.1236   0.1397   0.0611   0.0968   0.1299   0.1699
govt      201   0.2537   0.2298   0.2423   0.2788   0.1140   0.2158   0.2563   0.3418
all       777   0.1918   0.1752   0.1912   0.2203   0.0819   0.1656   0.2100   0.2797
```

```
2026-10-27 16:57  {'input': 'C:\\Users\\hp\\OneDrive\\Desktop\\IRE_PROJECT\\IRE_PROJECT\\data\\taskA_dev_input.jsonl', 'sources': ['bm25@weighted'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': None, 'write': False}
== bm25_weighted ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.2404   0.2295   0.2461   0.2960   0.0897   0.2196   0.2713   0.3959
cloud     188   0.1330   0.1412   0.1702   0.1923   0.0605   0.1427   0.2071   0.2621
fiqa      180   0.1444   0.1151   0.1309   0.1466   0.0582   0.1024   0.1411   0.1801
govt      201   0.2587   0.2529   0.2838   0.3125   0.1200   0.2416   0.3169   0.3839
all       777   0.1969   0.1877   0.2108   0.2406   0.0832   0.1795   0.2374   0.3104
```

```
2026-10-27 16:57  {'input': 'C:\\Users\\hp\\OneDrive\\Desktop\\IRE_PROJECT\\IRE_PROJECT\\data\\taskA_dev_input.jsonl', 'sources': ['bm25@gold_rewrite'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': None, 'write': False}
== bm25_gold_rewrite ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.2692   0.2554   0.2701   0.3136   0.1068   0.2389   0.2903   0.3957
cloud     188   0.1330   0.1540   0.1713   0.2039   0.0691   0.1607   0.2028   0.2821
fiqa      180   0.1500   0.1275   0.1508   0.1720   0.0682   0.1156   0.1689   0.2191
govt      201   0.2488   0.2354   0.2606   0.3111   0.1137   0.2259   0.2890   0.4056
all       777   0.2033   0.1961   0.2161   0.2536   0.0905   0.1880   0.2406   0.3299
```

```
2026-10-27 17:39  {'input': 'C:\\Users\\hp\\OneDrive\\Desktop\\IRE_PROJECT\\IRE_PROJECT\\data\\taskA_dev_input.jsonl', 'sources': ['bm25@file:data/rewrites_t5canard.jsonl'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': None, 'write': False}
== bm25_rewrites_t5canard ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.2308   0.2223   0.2444   0.2829   0.0898   0.2071   0.2694   0.3605
cloud     188   0.1383   0.1545   0.1753   0.2004   0.0696   0.1595   0.2096   0.2687
fiqa      180   0.1278   0.1160   0.1323   0.1503   0.0585   0.1119   0.1492   0.1912
govt      201   0.2488   0.2344   0.2659   0.3030   0.1138   0.2212   0.2946   0.3808
all       777   0.1892   0.1844   0.2073   0.2374   0.0839   0.1772   0.2336   0.3043
```

```
2026-10-27 17:39  {'input': 'C:\\Users\\hp\\OneDrive\\Desktop\\IRE_PROJECT\\IRE_PROJECT\\data\\taskA_dev_input.jsonl', 'sources': ['bm25@file:data/rewrites_t5canard.jsonl', 'bm25@weighted'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': 'bm25_rrf_t5+weighted', 'write': False}
== bm25_rrf_t5+weighted ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.2356   0.2368   0.2676   0.3115   0.0866   0.2286   0.3099   0.4170
cloud     188   0.1543   0.1561   0.1790   0.2079   0.0758   0.1566   0.2089   0.2798
fiqa      180   0.1333   0.1118   0.1261   0.1466   0.0602   0.1026   0.1382   0.1876
govt      201   0.2587   0.2523   0.2766   0.3153   0.1153   0.2403   0.3015   0.3949
all       777   0.1982   0.1923   0.2157   0.2492   0.0853   0.1850   0.2435   0.3249
```

```
2026-10-27 17:39  {'input': 'C:\\Users\\hp\\OneDrive\\Desktop\\IRE_PROJECT\\IRE_PROJECT\\data\\taskA_dev_input.jsonl', 'sources': ['bm25@file:data/rewrites_t5canard.jsonl', 'bm25@weighted', 'bm25@last'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': 'bm25_rrf_t5+weighted+last', 'write': False}
== bm25_rrf_t5+weighted+last ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.2404   0.2333   0.2634   0.3052   0.0948   0.2227   0.2984   0.4035
cloud     188   0.1702   0.1749   0.2007   0.2245   0.0864   0.1772   0.2322   0.2882
fiqa      180   0.1278   0.1058   0.1240   0.1383   0.0525   0.0964   0.1394   0.1738
govt      201   0.2736   0.2416   0.2755   0.3174   0.1190   0.2216   0.3013   0.3992
all       777   0.2059   0.1918   0.2191   0.2502   0.0892   0.1822   0.2463   0.3213
```

```
2026-10-28 21:30  {'input': 'C:\\Users\\hp\\OneDrive\\Desktop\\IRE_PROJECT\\IRE_PROJECT\\data\\taskA_dev_input.jsonl', 'sources': ['bm25@file:data/rewrites_t5canard.jsonl', 'bm25@weighted', 'bm25@last'], 'weights': None, 'depth': 100, 'rerank': 'cross-encoder/ms-marco-MiniLM-L6-v2', 'rerank_query': 'file:data/rewrites_t5canard.jsonl', 'rerank_depth': 50, 'name': 'bm25rrf3__rr_minilm50', 'write': True}
== bm25rrf3__rr_minilm50 ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.4423   0.3727   0.4028   0.4469   0.1746   0.3303   0.4218   0.5288
cloud     188   0.3085   0.2951   0.3137   0.3323   0.1524   0.2801   0.3361   0.3813
fiqa      180   0.2278   0.2004   0.2212   0.2411   0.0963   0.1797   0.2379   0.2863
govt      201   0.3980   0.3674   0.3845   0.4239   0.1733   0.3459   0.3996   0.4902
all       777   0.3488   0.3126   0.3344   0.3656   0.1507   0.2873   0.3527   0.4270
```

## Runs on ADA (GPU)


```
2026-10-28 23:48  {'input': 'data/taskA_dev_input.jsonl', 'sources': ['bm25@last'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': 'bm25_last', 'write': False}
== bm25_last ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.2212   0.2085   0.2286   0.2686   0.0898   0.1944   0.2529   0.3508
cloud     188   0.1489   0.1552   0.1774   0.2028   0.0736   0.1553   0.2112   0.2699
fiqa      180   0.0833   0.0894   0.1046   0.1208   0.0368   0.0869   0.1209   0.1605
govt      201   0.2488   0.2318   0.2577   0.2877   0.1153   0.2119   0.2763   0.3449
all       777   0.1789   0.1740   0.1950   0.2234   0.0802   0.1646   0.2183   0.2856
```

```
2026-10-28 23:48  {'input': 'data/taskA_dev_input.jsonl', 'sources': ['dense:BAAI/bge-base-en-v1.5@last'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': 'dense_last', 'write': False}
== dense_last ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.3894   0.3680   0.3879   0.4376   0.1484   0.3448   0.4188   0.5368
cloud     188   0.2979   0.2986   0.3233   0.3561   0.1486   0.2923   0.3539   0.4301
fiqa      180   0.2833   0.2507   0.2762   0.3165   0.1196   0.2281   0.2969   0.3945
govt      201   0.3134   0.2937   0.3149   0.3629   0.1391   0.2792   0.3407   0.4522
all       777   0.3230   0.3048   0.3275   0.3705   0.1394   0.2881   0.3547   0.4561
```

```
2026-10-28 23:48  {'input': 'data/taskA_dev_input.jsonl', 'sources': ['dense:BAAI/bge-base-en-v1.5@file:data/rewrites_t5canard.jsonl'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': 'dense_t5', 'write': False}
== dense_t5 ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.4471   0.3946   0.4232   0.4764   0.1712   0.3638   0.4546   0.5816
cloud     188   0.3138   0.3005   0.3205   0.3629   0.1428   0.2929   0.3501   0.4532
fiqa      180   0.2722   0.2442   0.2790   0.3184   0.1186   0.2233   0.3058   0.3979
govt      201   0.3333   0.3365   0.3564   0.4025   0.1477   0.3331   0.3888   0.4945
all       777   0.3449   0.3220   0.3477   0.3932   0.1461   0.3062   0.3778   0.4854
```

```
2026-10-28 23:49  {'input': 'data/taskA_dev_input.jsonl', 'sources': ['dense:BAAI/bge-base-en-v1.5@weighted'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': 'dense_weighted', 'write': False}
== dense_weighted ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.2404   0.2559   0.2983   0.3500   0.0883   0.2484   0.3495   0.4694
cloud     188   0.2181   0.2096   0.2320   0.2612   0.0950   0.2027   0.2520   0.3223
fiqa      180   0.1944   0.1595   0.1836   0.2129   0.0832   0.1493   0.2019   0.2687
govt      201   0.2388   0.2504   0.2836   0.3176   0.1113   0.2500   0.3273   0.4046
all       777   0.2239   0.2209   0.2519   0.2884   0.0947   0.2148   0.2859   0.3706
```

```
2026-10-28 23:49  {'input': 'data/taskA_dev_input.jsonl', 'sources': ['dense:BAAI/bge-base-en-v1.5@file:data/rewrites_t5canard.jsonl', 'dense:BAAI/bge-base-en-v1.5@weighted', 'bm25@file:data/rewrites_t5canard.jsonl', 'bm25@weighted'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': 'hybrid4', 'write': False}
== hybrid4 ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.4038   0.3528   0.3910   0.4436   0.1577   0.3168   0.4207   0.5476
cloud     188   0.2713   0.2610   0.2876   0.3321   0.1234   0.2565   0.3257   0.4291
fiqa      180   0.2167   0.1926   0.2088   0.2580   0.0860   0.1875   0.2292   0.3512
govt      201   0.3234   0.3097   0.3467   0.3990   0.1430   0.2896   0.3845   0.5122
all       777   0.3076   0.2823   0.3123   0.3621   0.1290   0.2652   0.3440   0.4643
```

```
2026-10-29 00:21  {'input': 'data/taskA_dev_input.jsonl', 'sources': ['dense:BAAI/bge-base-en-v1.5@file:data/rewrites_t5canard.jsonl', 'dense:BAAI/bge-base-en-v1.5@weighted', 'bm25@file:data/rewrites_t5canard.jsonl', 'bm25@weighted'], 'weights': None, 'depth': 100, 'rerank': 'BAAI/bge-reranker-base', 'rerank_query': 'file:data/rewrites_t5canard.jsonl', 'rerank_depth': 100, 'name': 'hybrid4__rr', 'write': True}
== hybrid4__rr ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.3173   0.3224   0.3536   0.3969   0.1275   0.3029   0.3938   0.4949
cloud     188   0.2819   0.2904   0.3208   0.3602   0.1381   0.2924   0.3663   0.4589
fiqa      180   0.2167   0.2126   0.2301   0.2756   0.0933   0.2035   0.2526   0.3608
govt      201   0.3284   0.3157   0.3546   0.3981   0.1441   0.3051   0.3927   0.4961
all       777   0.2883   0.2875   0.3173   0.3602   0.1265   0.2779   0.3542   0.4554
```

```
2026-10-29 00:26  {'input': 'data/taskA_dev_input.jsonl', 'sources': ['dense:BAAI/bge-base-en-v1.5@file:data/rewrites_t5canard.jsonl'], 'weights': None, 'depth': 100, 'rerank': 'cross-encoder/ms-marco-MiniLM-L6-v2', 'rerank_query': 'file:data/rewrites_t5canard.jsonl', 'rerank_depth': 50, 'name': 'dense_t5__rr_minilm50', 'write': True}
== dense_t5__rr_minilm50 ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.4904   0.4165   0.4567   0.5108   0.1976   0.3725   0.4828   0.6147
cloud     188   0.3564   0.3379   0.3718   0.4018   0.1649   0.3187   0.4098   0.4783
fiqa      180   0.2500   0.2414   0.2785   0.3210   0.1052   0.2218   0.3143   0.4154
govt      201   0.3980   0.3910   0.4153   0.4643   0.1782   0.3841   0.4460   0.5621
all       777   0.3784   0.3503   0.3842   0.4285   0.1633   0.3276   0.4166   0.5219
```

```
2026-10-29 00:26  {'input': 'data/taskA_dev_input.jsonl', 'sources': ['dense:BAAI/bge-base-en-v1.5@file:data/rewrites_t5canard.jsonl', 'dense:BAAI/bge-base-en-v1.5@last', 'bm25@file:data/rewrites_t5canard.jsonl', 'bm25@weighted', 'bm25@last'], 'weights': None, 'depth': 100, 'rerank': None, 'rerank_query': None, 'rerank_depth': 50, 'name': 'hybrid5', 'write': False}
== hybrid5 ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.4135   0.3583   0.3902   0.4287   0.1647   0.3269   0.4173   0.5117
cloud     188   0.2713   0.2582   0.2895   0.3356   0.1274   0.2513   0.3279   0.4340
fiqa      180   0.2000   0.1797   0.2034   0.2400   0.0776   0.1682   0.2254   0.3148
govt      201   0.3234   0.3156   0.3499   0.3927   0.1391   0.2960   0.3864   0.4912
all       777   0.3063   0.2817   0.3121   0.3531   0.1289   0.2639   0.3432   0.4420
```

```
2026-10-29 00:29  {'input': 'data/taskA_dev_input.jsonl', 'sources': ['dense:BAAI/bge-base-en-v1.5@file:data/rewrites_t5canard.jsonl', 'dense:BAAI/bge-base-en-v1.5@last', 'bm25@file:data/rewrites_t5canard.jsonl', 'bm25@weighted', 'bm25@last'], 'weights': None, 'depth': 100, 'rerank': 'cross-encoder/ms-marco-MiniLM-L6-v2', 'rerank_query': 'file:data/rewrites_t5canard.jsonl', 'rerank_depth': 50, 'name': 'hybrid5__rr_minilm50', 'write': True}
== hybrid5__rr_minilm50 ==
domain      n   nDCG@1   nDCG@3   nDCG@5  nDCG@10      R@1      R@3      R@5     R@10
clapnq    208   0.5000   0.4153   0.4611   0.5155   0.2000   0.3641   0.4872   0.6178
cloud     188   0.3564   0.3409   0.3655   0.4025   0.1672   0.3239   0.3960   0.4779
fiqa      180   0.2778   0.2583   0.2827   0.3257   0.1163   0.2331   0.3067   0.4111
govt      201   0.4129   0.3893   0.4210   0.4665   0.1816   0.3706   0.4520   0.5602
all       777   0.3912   0.3542   0.3863   0.4315   0.1679   0.3257   0.4142   0.5212
```
