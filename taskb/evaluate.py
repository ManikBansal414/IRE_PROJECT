"""Score Task B predictions like the official script (benchmark scripts/evaluation), with an LLM judge
(gpt-4.1-mini on Azure by default; Groq models as a cross-check).

  RB_alg  official code, run unchanged (run_algorithmic.py): harmonic mean of BERTScore recall and ROUGE-L against
          the reference answer and BERTScore precision against the passages
  RB_llm  official RAD-Bench judge prompt, rating 1-10 divided by 10
  RL_F    faithfulness as defined by RAGAS: the answer is split into statements, each is checked against the
          passages, score = share supported. Re-implemented here (the official script calls the ragas library).
  idk     official IDK judge prompt: does the answer say that it cannot answer?

Each metric is IDK-conditioned as in the official get_idk_score: on UNANSWERABLE/CONVERSATIONAL tasks it is 1 if
the answer is an IDK and 0 otherwise; on the other tasks a full IDK scores 0. The ranking score is the harmonic
mean of the three conditioned means. The default judge (gpt-4.1-mini) is also the default generator; a second judge
from another model family (--judge_provider groq) cross-checks that this does not change any decision.

  python -m taskb.evaluate --name base                         # all metrics, judge gpt-4.1-mini on Azure
  python -m taskb.evaluate --name base --no_llm                # RB_alg only, no API cost
  python -m taskb.evaluate --name base --judge meta-llama/llama-4-scout-17b-16e-instruct --judge_provider groq \
      --sample 100                                              # cross-check judge (free tier) on part of a run
  python -m taskb.evaluate --compare base extractive [--suffix eval_alg]  # paired bootstrap between two runs
  python -m taskb.evaluate --compare mini_base.s150 mini_extractive.s150     # same, on sample results
"""
import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from taska.data import BENCH, ROOT, WORK, read_jsonl, write_jsonl
from .data import dev_sample, load_gold
from .llm import Client, ContentFiltered
from .prompts import IDK

EVAL = BENCH / "scripts" / "evaluation"
sys.path.insert(0, str(EVAL))
from judge_utils import (extract_conversation, extract_rating, first_token_idk,  # noqa: E402
                         format_conversation_radbench, format_idk_judge, split_conversation)

OUT = ROOT / "results" / "taskb"
TMP = WORK / "taskb_eval"
# The official metric code needs evaluate/bert_score, kept in their own environment (python -m venv .venv;
# .venv/bin/pip install evaluate==0.4.3 bert_score rouge-score beautifulsoup4 lxml pandas pyyaml "transformers<5").
EVAL_PYTHON = ROOT / ".venv" / "bin" / "python"
JUDGE, JUDGE_PROVIDER = "gpt-4.1-mini", "azure"
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


FILTERED = []  # (task_id, judge step) refused by Azure's content filter; that score is left out (None)


def ask(judge, prompt, tid, step, json_mode=False):
    try:
        return judge.chat(user(prompt), json_mode=json_mode)
    except ContentFiltered:
        FILTERED.append((tid, step))
        return None


def write_merged(name, rows, gold):
    """Input of the official run_algorithmic.py: predictions plus the reference answer."""
    merged = TMP / f"{name}.merged.jsonl"
    write_jsonl(merged, [{**r, "targets": [{"speaker": "agent", "text": gold[r["task_id"]]["target"]}]}
                         for r in rows])
    return merged


def rb_alg(name, rows, gold):
    """Official algorithmic metrics; the scored file is kept, so re-evaluating a run costs nothing.
    A <name>.alg.jsonl scored elsewhere (ada/rb_alg.sbatch on a GPU) is used as is."""
    scored = TMP / f"{name}.alg.jsonl"
    if not scored.exists():
        merged, part = write_merged(name, rows, gold), TMP / f"{name}.alg.part.jsonl"
        python = str(EVAL_PYTHON) if EVAL_PYTHON.exists() else sys.executable
        proc = subprocess.Popen([python, str(EVAL / "run_algorithmic.py"), "-i", str(merged), "-o", str(part),
                                 "-e", str(ROOT / "taskb" / "alg_config.yaml")])
        # The official script sometimes hangs on exit after writing every row, so a complete file is accepted.
        while proc.poll() is None and not (part.exists() and len(part.read_text().splitlines()) == len(rows)):
            time.sleep(5)
        if proc.poll() is None:
            time.sleep(10)
            proc.kill()
        elif proc.returncode:
            raise RuntimeError(f"run_algorithmic.py failed with exit code {proc.returncode}")
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


def parse_json(text):
    """JSON from a reply; models without JSON mode may wrap it in prose or a code fence."""
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def faithfulness(judge, question, answer, passages, tid=None):
    """RAGAS-style faithfulness; None when the answer yields no statements (RAGAS returns NaN)."""
    try:
        reply = ask(judge, STATEMENTS.replace("QUESTION", question).replace("ANSWER", answer), tid, "RL_F", True)
        if reply is None:
            return None
        statements = parse_json(reply)["statements"]
        if not statements:
            return None
        prompt = VERDICTS.replace("CONTEXT", "\n\n".join(passages)).replace(
            "STATEMENTS", "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1)))
        reply = ask(judge, prompt, tid, "RL_F", True)
        if reply is None:
            return None
        verdicts = parse_json(reply)["verdicts"]
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


def kind(use_llm, judge_model, use_alg=True):
    """Result file suffix: the default judge -> eval, another judge -> eval_<judge>, RB_alg only -> eval_alg,
    judge without RB_alg -> eval_noalg (or eval_noalg_<judge>)."""
    if not use_llm:
        return "eval_alg"
    return ("eval" if use_alg else "eval_noalg") + ("" if judge_model == JUDGE else "_" + judge_model.split("/")[-1])


def evaluate_run(name, judge_model, judge_provider, use_llm, sample=None, workers=8, use_alg=True,
                 sample_all=False, prepare_alg=False):
    rows = read_jsonl(OUT / f"{name}.predictions.jsonl")
    if sample:  # e.g. a cross-check with a second judge on part of a run
        keep = set(dev_sample(sample, everything=sample_all))
        rows = [r for r in rows if r["task_id"] in keep]
    gold = load_gold()
    # On a sample, only the sampled tasks are scored (the official RB_alg code takes ~20 s per task on CPU).
    tag = name if not sample else f"{name}.{'a' if sample_all else 's'}{sample}"  # a = unanswerable included
    if prepare_alg:  # only write the RB_alg input, to be scored on a GPU (ada/rb_alg.sbatch)
        print(write_merged(tag, rows, gold))
        return
    alg = rb_alg(tag, rows, gold) if use_alg else {}
    if use_llm:
        judge = Client(judge_model, judge_provider, max_tokens=2048, tag=f"judge:{name}")
        idk_prompts, rad_prompts = judge_prompts(rows, gold)

    def score(i):
        r = rows[i]
        tid, answer = r["task_id"], r["predictions"][0]["text"]
        label = gold[tid]["answerability"]
        m = {"RB_alg": alg.get(tid)}
        if use_llm:
            m["idk"] = 1 if answer == IDK else first_token_idk(ask(judge, idk_prompts[i], tid, "idk"))
            # The conditioned score ignores RB_llm and RL_F on these tasks, so they are not judged (saves cost).
            if label not in NONANSWERABLE and m["idk"] != 1:
                rating = ask(judge, rad_prompts[i], tid, "RB_llm")
                m["RB_llm"] = None if rating is None else extract_rating(rating)
                m["RL_F"] = faithfulness(judge, extract_conversation(r["input"]), answer,
                                         [c["text"] for c in r["contexts"]], tid)
        else:
            m["idk"] = 1 if answer == IDK else 0  # without the judge only the fixed reply counts as an IDK
        m["cond"] = {metric: condition(m, label, metric) for metric in METRICS}
        if not use_alg:
            m["cond"]["RB_alg"] = None  # not computed in this evaluation, so it must not count as 0
        if not use_llm:  # no judge: RL_F and RB_llm are unknown, and only the fixed reply counts as an IDK
            m["cond"]["RL_F"] = m["cond"]["RB_llm"] = None
        return tid, m

    per_task = {}
    with ThreadPoolExecutor(workers if use_llm and judge_provider == "azure" else 1) as pool:
        for i, (tid, m) in enumerate(pool.map(score, range(len(rows)))):
            per_task[tid] = m
            if use_llm and (i + 1) % 100 == 0:
                print(f"judged {i + 1}/{len(rows)}  {judge.status()}", flush=True)
    if use_llm:
        print(f"judge: {judge.status()}; content-filtered judge calls: {len(FILTERED)}")

    summary = summarise(per_task, gold)
    # Each judge (and RB_alg alone) has its own result file, so no evaluation replaces another.
    label = tag  # sample results never replace full-run results
    (OUT / f"{label}.{kind(use_llm, judge_model, use_alg)}.json").write_text(json.dumps(
        {"judge": judge_model if use_llm else None, "content_filtered": FILTERED, "summary": summary,
         "per_task": per_task}, indent=1))
    table = format_summary(label, summary, judge_model if use_llm else "none (RB_alg only)")
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


def compare(a, b, suffix="eval"):
    """Paired bootstrap over the tasks both runs were evaluated on: b minus a, per metric and harmonic mean."""
    pa, pb = (json.loads((OUT / f"{n}.{suffix}.json").read_text())["per_task"] for n in (a, b))
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
    ap.add_argument("--judge_provider", default=JUDGE_PROVIDER, choices=["azure", "groq"])
    ap.add_argument("--sample", type=int, default=None, help="score only the fixed dev sample of this size")
    ap.add_argument("--sample_all", action="store_true", help="the sample includes unanswerable tasks (Task C)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no_llm", action="store_true", help="RB_alg only")
    ap.add_argument("--no_alg", action="store_true", help="LLM judge only (RB_alg is slow on CPU)")
    ap.add_argument("--prepare_alg", action="store_true",
                    help="only write work/taskb_eval/<run>.merged.jsonl for RB_alg on a GPU")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--suffix", default="eval", help="which result files --compare reads (eval, eval_alg, ...)")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare, args.suffix)
    else:
        evaluate_run(args.name, args.judge, args.judge_provider, not args.no_llm, args.sample, args.workers,
                     not args.no_alg, args.sample_all, args.prepare_alg)


if __name__ == "__main__":
    main()
