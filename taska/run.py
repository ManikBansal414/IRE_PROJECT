"""Run a Task A retrieval pipeline, score it, and optionally write an official-format prediction file.

A pipeline is one or more sources "<retriever>@<query>", fused with RRF when there are several,
then optionally reranked with a cross-encoder.
  retriever: bm25 | dense:<sentence-transformers model>
  query:     last | all | last2 | weighted | file:<rewrites.jsonl> | gold_rewrite (upper bound only)

Examples:
  python -m taska.run --sources bm25@last
  python -m taska.run --sources bm25@file:data/rewrites_t5canard.jsonl dense:BAAI/bge-small-en-v1.5@file:data/rewrites_t5canard.jsonl \
      --rerank cross-encoder/ms-marco-MiniLM-L-6-v2 --rerank_query file:data/rewrites_t5canard.jsonl --write
"""
import argparse
import json
import re
from datetime import datetime

from . import bm25, queries
from .data import DOMAINS, ROOT, load_corpus, load_tasks, read_jsonl, write_jsonl
from .evaluate import evaluate_run, format_table
from .fusion import rrf

TOP_K = 10


def build_queries(tasks, strategy):
    if strategy == "gold_rewrite":
        gold = {r["task_id"]: r["rewrite"]
                for r in read_jsonl(ROOT / "data" / "gold_rewrites_UPPER_BOUND_ONLY.jsonl")}
        return {t["task_id"]: gold.get(t["task_id"]) or queries.last_turn(t) for t in tasks}
    if strategy.startswith("file:"):  # precomputed rewrites, e.g. from taska.rewrite
        rw = {r["task_id"]: r["rewrite"] for r in read_jsonl(ROOT / strategy[5:])}
        return {t["task_id"]: rw[t["task_id"]] for t in tasks}
    fn = queries.STRATEGIES[strategy]
    return {t["task_id"]: fn(t) for t in tasks}


def retrieve(tasks, qtext, retriever, depth):
    run = {}
    for domain in DOMAINS:
        dtasks = [t for t in tasks if t["domain"] == domain]
        if not dtasks:
            continue
        qs = [qtext[t["task_id"]] for t in dtasks]
        if retriever == "bm25":
            hits = bm25.search(domain, qs, k=depth)
        elif retriever.startswith("dense:"):
            from . import dense
            hits = dense.search(retriever[6:], domain, qs, k=depth)
        else:
            raise ValueError(retriever)
        for t, h in zip(dtasks, hits):
            run[t["task_id"]] = h
    return run


def corpus_texts():
    texts = {}
    for domain in DOMAINS:
        ids, txt = load_corpus(domain)
        texts.update(zip(ids, txt))
    return texts


def write_predictions(path, tasks, run, texts):
    rows = []
    for t in tasks:
        row = {k: t[k] for k in ("conversation_id", "task_id", "Collection", "input")}
        row["contexts"] = [{"document_id": pid, "text": texts[pid], "score": round(s, 6)}
                           for pid, s in run[t["task_id"]][:TOP_K]]
        rows.append(row)
    write_jsonl(path, rows)


def short(s):
    s = s.replace("file:data/", "").replace(".jsonl", "").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "taskA_dev_input.jsonl"))
    ap.add_argument("--sources", nargs="+", default=["bm25@last"])
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="RRF weight per source")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--rerank", default=None, help="cross-encoder model name")
    ap.add_argument("--rerank_query", default=None, help="query strategy for the reranker")
    ap.add_argument("--rerank_depth", type=int, default=50)
    ap.add_argument("--name", default=None)
    ap.add_argument("--write", action="store_true", help="write the official-format prediction file")
    args = ap.parse_args()

    tasks = load_tasks(args.input)
    runs = []
    for src in args.sources:
        retriever, strategy = src.split("@", 1)
        runs.append(retrieve(tasks, build_queries(tasks, strategy), retriever, args.depth))
    if len(runs) == 1:
        run = runs[0]
    else:
        run = {tid: rrf([r[tid] for r in runs], weights=args.weights) for tid in runs[0]}

    texts = None
    if args.rerank:
        from .rerank import rerank
        texts = corpus_texts()
        rq = build_queries(tasks, args.rerank_query or args.sources[0].split("@", 1)[1])
        for i, t in enumerate(tasks):
            tid = t["task_id"]
            head = rerank(args.rerank, rq[tid], run[tid], texts, top_n=args.rerank_depth)
            run[tid] = head + run[tid][len(head):]
            if (i + 1) % 100 == 0:
                print(f"reranked {i + 1}/{len(tasks)}", flush=True)

    name = args.name or "+".join(short(s) for s in args.sources) + (
        f"__rr_{short(args.rerank)}" if args.rerank else "")
    results = evaluate_run({k: v[:TOP_K] for k, v in run.items()}, tasks)
    table = format_table(results, f"== {name} ==")
    print(table)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "results_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n```\n{datetime.now():%Y-%m-%d %H:%M}  {' '.join(vars(args).__repr__().split())}\n{table}\n```\n")
    with open(out / f"{name}.metrics.json", "w") as f:
        json.dump(results, f, indent=1)
    (ROOT / "work" / "runs").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "work" / "runs" / f"{name}.json", "w") as f:
        json.dump(run, f)
    if args.write:
        write_predictions(out / f"{name}.predictions.jsonl", tasks, run, texts or corpus_texts())
        print("predictions:", out / f"{name}.predictions.jsonl")


if __name__ == "__main__":
    main()
