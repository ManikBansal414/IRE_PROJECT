# Task B/C API usage and cost

Every paid API call, logged by `taskb/llm.py` (`llm_usage.jsonl` here is a copy of `work/llm_usage.jsonl`).
Cost = tokens × list price per million tokens (gpt-4.1-mini $0.40 in / $1.60 out, gpt-4.1 $2.00 / $8.00,
Llama-3.3-70B-Instruct $0.71 / $0.71; Azure global standard). Cached answers cost nothing and are not logged.

**Total: $13.93** over 14,520 calls, 22,276,320 input and 2,872,869 output tokens (2026-11-06 to 2026-11-07).

| purpose | run | model | calls | input tokens | output tokens | cost |
|---|---|---|---|---|---|---|
| Task B: answers | mini_base | gpt-4.1-mini | 752 | 1,060,309 | 73,815 | $0.542 |
| Task B: answers | mini_concise | gpt-4.1-mini | 775 | 1,092,678 | 57,371 | $0.529 |
| Task B: answers | mini_extractive | gpt-4.1-mini | 775 | 1,106,628 | 79,494 | $0.570 |
| Task B: answers | mini_extractive_long_s150 | gpt-4.1-mini | 149 | 209,975 | 17,944 | $0.113 |
| Task B: answers | mini_extractive_partial_s150 | gpt-4.1-mini | 149 | 213,849 | 14,146 | $0.108 |
| Task B: answers | model_gpt-4.1_s150 | gpt-4.1 | 149 | 207,144 | 12,461 | $0.514 |
| Task B: answers | smoke_mini_s20 | gpt-4.1-mini | 23 | 35,469 | 2,204 | $0.018 |
| Task B: judging | mini_base | gpt-4.1-mini | 495 | 636,819 | 114,509 | $0.438 |
| Task B: judging | mini_concise | gpt-4.1-mini | 548 | 715,906 | 111,114 | $0.464 |
| Task B: judging | mini_extractive | gpt-4.1-mini | 2,979 | 4,013,734 | 721,010 | $2.759 |
| Task B: judging | mini_extractive | Llama-3.3-70B-Instruct | 102 | 125,176 | 32,293 | $0.112 |
| Task B: judging | mini_extractive_long_s150 | gpt-4.1-mini | 562 | 754,688 | 151,522 | $0.544 |
| Task B: judging | mini_extractive_partial_s150 | gpt-4.1-mini | 478 | 635,710 | 117,445 | $0.442 |
| Task B: judging | model_gpt-4.1_s150 | gpt-4.1-mini | 560 | 728,676 | 123,137 | $0.488 |
| Task B: judging | smoke_mini_s20 | gpt-4.1-mini | 89 | 124,714 | 22,121 | $0.085 |
| Task C: answers | c_mini_base | gpt-4.1-mini | 840 | 1,726,382 | 82,473 | $0.823 |
| Task C: answers | c_mini_extractive | gpt-4.1-mini | 840 | 1,738,142 | 86,475 | $0.834 |
| Task C: answers | c_mini_extractive_rag_a150 | gpt-4.1-mini | 151 | 316,031 | 15,928 | $0.152 |
| Task C: judging | c_mini_base | gpt-4.1-mini | 548 | 904,912 | 137,995 | $0.583 |
| Task C: judging | c_mini_extractive | gpt-4.1-mini | 3,045 | 5,086,961 | 763,798 | $3.257 |
| Task C: judging | c_mini_extractive_rag_a150 | gpt-4.1-mini | 509 | 842,305 | 135,591 | $0.554 |
| connection test | llama | Llama-3.3-70B-Instruct | 2 | 112 | 23 | $0.000 |

| purpose | cost |
|---|---|
| Task B: answers | $2.39 |
| Task B: judging | $5.33 |
| Task C: answers | $1.81 |
| Task C: judging | $4.39 |
| connection test | $0.00 |

Final benchmark runs: Task B `mini_extractive`, Task C `c_mini_extractive`. Runs with `_s150` / `.s150`
are on the 149-task tuning sample, `_a150` on the 151-task Task C sample. The Llama-3.3-70B calls are a
cross-check judge that was stopped early; its results are not used.
