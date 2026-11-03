"""Score Task B predictions like the official script (benchmark scripts/evaluation), with the judges on Groq.

  RB_alg  official code, run unchanged (run_algorithmic.py): harmonic mean of BERTScore recall and ROUGE-L against
          the reference answer and BERTScore precision against the passages
  RB_llm  official RAD-Bench judge prompt, rating 1-10 divided by 10
  RL_F    faithfulness as defined by RAGAS: the answer is split into statements, each is checked against the
          passages, score = share supported. Re-implemented here (the official script calls the ragas library).
  idk     official IDK judge prompt: does the answer say that it cannot answer?

Each metric is IDK-conditioned as in the official get_idk_score: on UNANSWERABLE/CONVERSATIONAL tasks it is 1 if
the answer is an IDK and 0 otherwise; on the other tasks a full IDK scores 0. The ranking score is the harmonic
mean of the three conditioned means. The judge should differ from the generator, so it does not grade itself.

  python -m taskb.evaluate --name base_s150                    # all metrics
  python -m taskb.evaluate --name base_s150 --no_llm           # RB_alg only, no API quota
  python -m taskb.evaluate --compare base_s150 extractive_s150 # paired bootstrap between two evaluated runs
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from taska.data import BENCH, ROOT, WORK, read_jsonl, write_jsonl
from .data import load_gold
from .llm import Client
from .prompts import IDK

EVAL = BENCH / "scripts" / "evaluation"
sys.path.insert(0, str(EVAL))
from judge_utils import (extract_conversation, extract_rating, first_token_idk,  # noqa: E402
                         format_conversation_radbench, format_idk_judge, split_conversation)

OUT = ROOT / "results" / "taskb"
TMP = WORK / "taskb_eval"
JUDGE = "openai/gpt-oss-120b"
METRICS = ("RL_F", "RB_llm", "RB_alg")
NONANSWERABLE = {"UNANSWERABLE", "CONVERSATIONAL"}
N_BOOT, SEED = 10000, 0

STATEMENTS = """Given a question and an answer, break each sentence of the answer down into one or more fully understandable statements. Do not use pronouns in the statements. Return JSON of the form {"statements": ["...", "..."]}.

Question:
QUESTION

Answer:
ANSWER"""

VERDICTS = """Judge the faithfulness of a series of statements based on the given context. For each statement return verdict 1 if it can be directly inferred from the context and 0 if it cannot. Return JSON of the form {"verdicts": [{"statement": "...", "reason": "...", "verdict": 1}]}.

Context:
CONTEXT

Statements:
STATEMENTS"""


def user(text):
    return [{"role": "user", "content": text}]


def rb_alg(name, rows, gold):
    """Official algorithmic metrics; the scored file is kept, so re-evaluating a run costs nothing."""
    scored = TMP / f"{name}.alg.jsonl"
    if not scored.exists():
        merged, part = TMP / f"{name}.merged.jsonl", TMP / f"{name}.alg.part.jsonl"
        write_jsonl(merged, [{**r, "targets": [{"speaker": "agent", "text": gold[r["task_id"]]["target"]}]}
                             for r in rows])
        subprocess.run([sys.executable, str(EVAL / "run_algorithmic.py"), "-i", str(merged), "-o", str(part),
                        "-e", str(ROOT / "taskb" / "alg_config.yaml")], check=True)
        part.rename(scored)
    return {r["task_id"]: r["metrics"]["RB_agg"][0] for r in read_jsonl(scored)}


def judge_prompts(rows, gold):
    """The official IDK and RAD-Bench prompts, built by the official helpers."""
    df = pd.DataFrame({"inquiry": [extract_conversation(r["input"]) for r in rows],
                       "response": [r["predictions"][0]["text"] for r in rows],
                       "document": [[c["text"] for c in r["contexts"]] for r in rows],
                       "reference_answer": [gold[r["task_id"]]["target"] for r in rows]})
    df["previous_conversation"], df["current_question"] = zip(*df["inquiry"].apply(split_conversation))
    return format_idk_judge(df), format_conversation_radbench(df)


def faithfulness(judge, question, answer, passages):
    """RAGAS-style faithfulness; None when the answer yields no statements (RAGAS returns NaN)."""
    try:
        statements = json.loads(judge.chat(user(STATEMENTS.replace("QUESTION", question).replace("ANSWER", answer)),
                                           json_mode=True))["statements"]
        if not statements:
            return None
        prompt = VERDICTS.replace("CONTEXT", "\n\n".join(passages)).replace(
            "STATEMENTS", "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1)))
        verdicts = json.loads(judge.chat(user(prompt), json_mode=True))["verdicts"]
    except (ValueError, KeyError, TypeError):
        return None
    return sum(str(v.get("verdict")) == "1" for v in verdicts) / len(verdicts) if verdicts else None


def condition(m, label, metric):
    """Official IDK conditioning (judge_wrapper.get_idk_score)."""
    if label in NONANSWERABLE:
        return 1.0 if m["idk"] == 1 else 0.0
    if m["idk"] == 1:
        return 0.0
    return m.get(metric)


def harmonic(xs):
    xs = [x for x in xs if x is not None]
    return len(xs) / sum(1 / x for x in xs) if len(xs) == len(METRICS) and all(x > 0 for x in xs) else None


def summarise(per_task, gold):
    def means(tids):
        out = {}
        for metric in METRICS:
            vals = [per_task[t]["cond"][metric] for t in tids if per_task[t]["cond"][metric] is not None]
            out[metric] = float(np.mean(vals)) if vals else None
        out["harmonic"] = harmonic([out[m] for m in METRICS])
        out["n"] = len(tids)
        return out

    tids = list(per_task)
    groups = {"all": tids}
    for key in ("answerability", "domain"):
        for value in sorted({gold[t][key] for t in tids}):
            groups[value] = [t for t in tids if gold[t][key] == value]
    return {g: means(ts) for g, ts in groups.items()}


def evaluate_run(name, judge_model, use_llm):
    rows = read_jsonl(OUT / f"{name}.predictions.jsonl")
    gold = load_gold()
    alg = rb_alg(name, rows, gold)
    if use_llm:
        judge = Client(judge_model, max_tokens=2048)
        idk_prompts, rad_prompts = judge_prompts(rows, gold)

    per_task = {}
    for i, r in enumerate(rows):
        tid, answer = r["task_id"], r["predictions"][0]["text"]
        label = gold[tid]["answerability"]
        m = {"RB_alg": alg[tid]}
        if use_llm:
            m["idk"] = 1 if answer == IDK else first_token_idk(judge.chat(user(idk_prompts[i])))
            # The conditioned score ignores RB_llm and RL_F on these tasks, so they are not judged (saves quota).
            if label not in NONANSWERABLE and m["idk"] != 1:
                m["RB_llm"] = extract_rating(judge.chat(user(rad_prompts[i])))
                m["RL_F"] = faithfulness(judge, extract_conversation(r["input"]), answer,
                                         [c["text"] for c in r["contexts"]])
        else:
            m["idk"] = 1 if answer == IDK else 0  # without the judge only the fixed reply counts as an IDK
        m["cond"] = {metric: condition(m, label, metric) for metric in METRICS}
        per_task[tid] = m
        if use_llm and (i + 1) % 25 == 0:
            print(f"judged {i + 1}/{len(rows)}  api calls {judge.calls}, cached {judge.cached}", flush=True)

    summary = summarise(per_task, gold)
    # RB_alg-only results go to their own file, so they never replace a full evaluation.
    (OUT / f"{name}.{'eval' if use_llm else 'eval_alg'}.json").write_text(json.dumps(
        {"judge": judge_model if use_llm else None, "summary": summary, "per_task": per_task}, indent=1))
    table = format_summary(name, summary, judge_model if use_llm else "none (RB_alg only)")
    print(table)
    with open(OUT / "results_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n{datetime.now():%Y-%m-%d %H:%M}\n\n{table}\n")


def fmt(x):
    return "-" if x is None else f"{x:.4f}"


def format_summary(name, summary, judge):
    lines = [f"### {name} (judge: {judge})", "", "| group | n | RL_F | RB_llm | RB_alg | harmonic mean |",
             "|---|---|---|---|---|---|"]
    for g, s in summary.items():
        lines.append(f"| {g} | {s['n']} | " + " | ".join(fmt(s[m]) for m in (*METRICS, "harmonic")) + " |")
    return "\n".join(lines)


def compare(a, b):
    """Paired bootstrap over the tasks both runs were evaluated on: b minus a, per metric and harmonic mean."""
    pa, pb = (json.loads((OUT / f"{n}.eval.json").read_text())["per_task"] for n in (a, b))
    tids = sorted(pa.keys() & pb.keys())
    arr = {n: np.array([[np.nan if p[t]["cond"][m] is None else p[t]["cond"][m] for m in METRICS] for t in tids])
           for n, p in ((a, pa), (b, pb))}
    idx = np.random.default_rng(SEED).integers(0, len(tids), (N_BOOT, len(tids)))

    def stats(x):  # (N_BOOT, metrics) bootstrap means and their harmonic mean
        mu = np.nanmean(x[idx], axis=1)
        return np.column_stack([mu, len(METRICS) / np.sum(1 / mu, axis=1)])

    def point(x):
        mu = np.nanmean(x, axis=0)
        return np.append(mu, len(METRICS) / np.sum(1 / mu))

    diff, d0 = stats(arr[b]) - stats(arr[a]), point(arr[b]) - point(arr[a])
    print(f"{b} minus {a}, {len(tids)} common tasks, paired bootstrap ({N_BOOT} resamples)")
    for j, m in enumerate((*METRICS, "harmonic")):
        lo, hi = np.nanpercentile(diff[:, j], [2.5, 97.5])
        p = np.mean(diff[:, j] <= 0) if d0[j] > 0 else np.mean(diff[:, j] >= 0)
        print(f"  {m:9s} {point(arr[a])[j]:.4f} -> {point(arr[b])[j]:.4f}  diff {d0[j]:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="run name, i.e. results/taskb/<name>.predictions.jsonl")
    ap.add_argument("--judge", default=JUDGE)
    ap.add_argument("--no_llm", action="store_true", help="RB_alg only")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        evaluate_run(args.name, args.judge, not args.no_llm)


if __name__ == "__main__":
    main()
