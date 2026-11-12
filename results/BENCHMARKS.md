# MTRAGEval (SemEval-2026 Task 8): benchmark results

Dev set: MTRAG human data, 842 tasks over 110 conversations in 4 domains (ClapNQ, Cloud, FiQA, Govt).
Tasks and metrics follow https://ibm.github.io/mt-rag-benchmark/MTRAGEval/. Answerability, question type and
rewrites are removed from every system input. All prediction files pass the official `format_checker.py`.

| Task | Metric | Score |
|---|---|---|
| A: retrieval | nDCG@10 | **0.5021** |
| B: generation with reference passages | harmonic mean of RL_F, RB_llm, RB_alg | **0.7160** |
| C: generation with retrieved passages (top 5) | harmonic mean of RL_F, RB_llm, RB_alg | **0.6099** |

## Task A: retrieval

**System.** The query variants are:
- the last user question;
- the last two questions;
- a weighted history;
- all questions;
- a T5-CANARD rewrite (`castorini/t5-base-canard`).

These go to two retrievers:
- BM25 (`bm25s`, k1 = 2.2, b = 0.5);
- dense retrieval (`BAAI/bge-base-en-v1.5`, FAISS).

Eight of the resulting ten ranked lists (BM25 on the T5 rewrite and dense on all questions get weight 0) are fused
with weighted RRF (k = 5), dense weighted 2–3× over BM25. The top 50 are then
reranked by `bge-reranker-v2-m3` (T5-rewrite query) and `ms-marco-MiniLM-L6-v2` (last-question query), whose
min-max-normalised scores are averaged. Final score = 0.7 × reranker + 0.3 × fusion; top 10 returned.

| | nDCG@1 | nDCG@3 | nDCG@5 | **nDCG@10** | R@10 |
|---|---|---|---|---|---|
| ClapNQ (208) | 0.519 | 0.483 | 0.515 | **0.574** | 0.689 |
| Cloud (188) | 0.441 | 0.400 | 0.432 | **0.477** | 0.562 |
| FiQA (180) | 0.378 | 0.324 | 0.360 | **0.412** | 0.511 |
| Govt (201) | 0.473 | 0.445 | 0.485 | **0.532** | 0.636 |
| **All (777 scored)** | **0.456** | **0.416** | **0.451** | **0.502** | **0.603** |

**Tuning.** All parameters were tuned on dev; see `results/tuning/README.md`.
- **Gain over the untuned pipeline:** 0.432 → 0.502 nDCG@10, +0.070 [+0.055, +0.085] (95% CI from a paired bootstrap), p < 0.001.
- **Cross-validation:** 5-fold over conversations, held-out score also 0.502. All folds chose the same setup.
- **End-to-end check:** a full run on the GPU cluster reproduced the tuning result exactly.

## Task B: generation with reference passages

**System.**
- **Generator:** `gpt-4.1-mini` (Azure), temperature 0.
- **Prompt:** answer only from the passages, reuse their wording, 2–5 sentences, say what is missing when they
  answer only part of the question.
- **No passages given:** the fixed reply "I'm sorry, but I don't have the answer to your question." In the dev
  set these are exactly the 65 unanswerable and conversational tasks.

| | n | RL_F | RB_llm | RB_alg | **Harmonic mean** |
|---|---|---|---|---|---|
| Answerable | 709 | 0.902 | 0.802 | 0.519 | 0.701 |
| Partial | 68 | 0.693 | 0.702 | 0.386 | 0.550 |
| Unanswerable | 55 | 1.000 | 1.000 | 1.000 | 1.000 |
| Conversational | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| ClapNQ / Cloud / FiQA / Govt | 224 / 205 / 199 / 214 | 0.886 / 0.907 / 0.911 / 0.868 | 0.803 / 0.811 / 0.823 / 0.801 | 0.568 / 0.559 / 0.511 / 0.541 | 0.726 / 0.728 / 0.702 / 0.706 |
| **All** | **842** | **0.892** | **0.809** | **0.546** | **0.716** |

**Choice of prompt and model.** Measured on a fixed 149-task sample, plus judge-free RB_alg on the full dev set.

| Candidate (sample) | Harmonic mean | vs chosen |
|---|---|---|
| **gpt-4.1-mini, extractive (chosen)** | **0.686** | — |
| gpt-4.1-mini, extractive, 3–6 sentences | 0.695 | +0.009, not significant (p = 0.07) |
| gpt-4.1-mini, base prompt | 0.675 | −0.011 (p = 0.05) |
| gpt-4.1, base prompt | 0.667 | gpt-4.1 tied with gpt-4.1-mini on the same prompt (p = 0.17) |
| gpt-4.1-mini, concise (1–3 sentences) | 0.651 | −0.036 (p < 0.001) |

On the full dev set, RB_alg (no LLM judge) confirms the choice:
- extractive 0.557;
- base 0.537 (+0.020, p < 0.001);
- concise 0.528 (+0.029, p < 0.001).

## Task C: generation with retrieved passages

**System.** The top 5 passages from the Task A system (the page recommends at most 5), then the Task B
generator and prompt.

| | n | RL_F | RB_llm | RB_alg | **Harmonic mean** |
|---|---|---|---|---|---|
| Answerable | 709 | 0.912 | 0.759 | 0.457 | 0.652 |
| Partial | 68 | 0.728 | 0.652 | 0.314 | 0.492 |
| Unanswerable | 55 | 0.273 | 0.273 | 0.273 | 0.273 |
| Conversational | 10 | 0.100 | 0.100 | 0.100 | 0.100 |
| ClapNQ / Cloud / FiQA / Govt | 224 / 205 / 199 / 214 | 0.819 / 0.865 / 0.866 / 0.837 | 0.710 / 0.715 / 0.694 / 0.724 | 0.462 / 0.444 / 0.360 / 0.445 | 0.626 / 0.624 / 0.558 / 0.622 |
| **All** | **842** | **0.846** | **0.711** | **0.429** | **0.610** |

**Choice of prompt.** On a 151-task sample that includes unanswerable tasks:
- extractive beats base: 0.604 vs 0.582, +0.023, p = 0.001;
- on the full dev set, RB_alg agrees: 0.421 vs 0.410, p < 0.001;
- a prompt telling the model to say "I don't know" when the retrieved passages are off-topic scored lower (0.587).

The main weakness is unanswerable questions. With retrieved passages the model rarely declines to answer, and
only 27% of unanswerable tasks get the expected "I don't know" reply.

## How the scores are computed

**Official parts.**
- **Task A:** nDCG/Recall per domain, then averaged weighted by question count, as in `run_retrieval_eval.py`.
  Only the 777 questions with relevant passages are scored.
- **RB_alg:** the official `run_algorithmic.py`, unchanged: harmonic mean of BERTScore recall and ROUGE-L
  against the reference answer, and BERTScore precision against the passages (`deberta-xlarge-mnli`).
- **RB_llm and the IDK check:** the official RAD-Bench and IDK judge prompts from `judge_utils.py`.
- **IDK conditioning:** the official `get_idk_score` rule. On unanswerable and conversational tasks a score is 1
  only for an "I don't know" reply; an "I don't know" on an answerable task scores 0.
- **Ranking score:** harmonic mean of the three conditioned means.

**Differences from the official setup.**
- **Judge:** `gpt-4.1-mini`. The script's default `gpt-4o-mini` is no longer deployable on Azure, and the page
  does not name the leaderboard's judge. The judge is also the generator, so RL_F and RB_llm may be slightly
  optimistic. RB_alg involves no LLM.
- **RL_F:** faithfulness as defined by RAGAS (answer split into statements, each checked against the passages),
  re-implemented with the same judge rather than through the `ragas` library.
- **Content filter:** in Task C, 1 judge call of about 3,000 was refused by Azure's content filter. That task's
  RL_F is left out.
- **Dev set, not the official test set:** the official test set was given only to registered teams during the
  January 2026 evaluation phase.

## Reproduce

```bash
python -m taska.run --preset best --write                       # Task A (GPU; ada/best_run.sbatch on Ada)
python -m taskb.generate --prompt extractive --name mini_extractive                 # Task B
python -m taskb.generate --input results/best_system/best.predictions.jsonl --top_k 5 \
    --prompt extractive --name c_mini_extractive                                    # Task C
python -m taskb.evaluate --name mini_extractive --prepare_alg   # then sbatch ada/rb_alg.sbatch (GPU)
python -m taskb.evaluate --name mini_extractive                 # all metrics; same for c_mini_extractive
python -m taskb.llm --usage                                     # tokens and cost per run
```

Prediction files are in `results/best_system/` (A) and `results/taskb/` (B: `mini_extractive`, C: `c_mini_extractive`).
LLM cost of all Task B/C work, tuning included: $13.93 (Azure).
