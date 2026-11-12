"""Run a Task A retrieval pipeline, score it, and optionally write an official-format prediction file.

A pipeline is one or more sources "<retriever>@<query>", fused with (weighted) RRF when there are several,
then optionally reranked on the top --rerank_depth with one or more cross-encoders "<model>[@<query>]".
With several cross-encoders their min-max-normalised scores are averaged; the final score is
alpha * CE + (1 - alpha) * min-max(fusion score), so alpha=1 ranks by the cross-encoder alone.
  retriever: bm25 | dense:<sentence-transformers model>
  query:     last | all | last2 | weighted | file:<rewrites.jsonl> | gold_rewrite (upper bound only)

Examples:
  python -m taska.run --preset best --write      # tuned system, see results/tuning/README.md
  python -m taska.run --sources bm25@last
  python -m taska.run --sources bm25@file:data/rewrites_t5canard.jsonl dense:BAAI/bge-small-en-v1.5@file:data/rewrites_t5canard.jsonl \
      --rerank cross-encoder/ms-marco-MiniLM-L6-v2 --rerank_query file:data/rewrites_t5canard.jsonl --write
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
T5 = "file:data/rewrites_t5canard.jsonl"
BGE = "dense:BAAI/bge-base-en-v1.5"
# Best dev configuration of the tuning sweep (results/tuning/README.md): nDCG@10 0.502, CV held-out 0.502.
PRESETS = {
    "best": dict(
        sources=["bm25@last", "bm25@last2", "bm25@weighted", "bm25@all",
                 f"{BGE}@last", f"{BGE}@last2", f"{BGE}@weighted", f"{BGE}@{T5}"],
        weights=[1, 0.25, 0.25, 0.5, 1.5, 0.5, 0.25, 2], rrf_k=5, bm25_k1=2.2, bm25_b=0.5,
        rerank=[f"BAAI/bge-reranker-v2-m3@{T5}", "cross-encoder/ms-marco-MiniLM-L6-v2@last"],
        rerank_depth=50, alpha=0.7, rerank_batch=16, name="best", out="results/best_system"),
}


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


def retrieve(tasks, qtext, retriever, depth, k1=bm25.K1, b=bm25.B):
    run = {}
    for domain in DOMAINS:
        dtasks = [t for t in tasks if t["domain"] == domain]
        if not dtasks:
            continue
        qs = [qtext[t["task_id"]] for t in dtasks]
        if retriever == "bm25":
            hits = bm25.search(domain, qs, k=depth, k1=k1, b=b)
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


def minmax(xs):
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]


def interpolate(head, ce_scores, alpha):
    """head: fused [(pid, score), ...]; ce_scores: one score list per cross-encoder.
    Returns head re-sorted by alpha * mean(minmax(CE)) + (1 - alpha) * minmax(fusion score)."""
    ce = [sum(col) / len(col) for col in zip(*map(minmax, ce_scores))]
    fused = minmax([s for _, s in head])
    final = [alpha * c + (1 - alpha) * f for c, f in zip(ce, fused)]
    return sorted(((pid, s) for (pid, _), s in zip(head, final)), key=lambda x: -x[1])


def source_run(tasks, src, args):
    retriever, strategy = src.split("@", 1)
    return retrieve(tasks, build_queries(tasks, strategy), retriever, args.depth, args.bm25_k1, args.bm25_b)


def pipeline(tasks, args):
    """Retrieve, fuse and rerank; returns ({task_id: [(pid, score), ...]}, corpus texts or None)."""
    runs = [source_run(tasks, src, args) for src in args.sources]
    if len(runs) == 1:
        run = runs[0]
    else:
        run = {tid: rrf([r[tid] for r in runs], k=args.rrf_k, weights=args.weights) for tid in runs[0]}

    texts = None
    if args.rerank:
        from . import rerank
        texts = corpus_texts()
        default_query = args.rerank_query or args.sources[0].split("@", 1)[1]
        models = [(m.split("@", 1) + [default_query])[:2] for m in args.rerank]
        rq = {query: build_queries(tasks, query) for _, query in models}
        for i, t in enumerate(tasks):
            tid = t["task_id"]
            head = run[tid][:args.rerank_depth]
            pids = [pid for pid, _ in head]
            ce = [rerank.score(m, rq[query][tid], pids, texts, args.rerank_batch) for m, query in models]
            if head:
                run[tid] = interpolate(head, ce, args.alpha) + run[tid][len(head):]
            if (i + 1) % 100 == 0:
                print(f"reranked {i + 1}/{len(tasks)}", flush=True)
    return run, texts


def short(s):
    s = s.replace("file:data/", "").replace(".jsonl", "").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=list(PRESETS))
    preset = pre.parse_known_args()[0].preset
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--input", default=str(ROOT / "data" / "taskA_dev_input.jsonl"))
    ap.add_argument("--sources", nargs="+", default=["bm25@last"])
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="RRF weight per source")
    ap.add_argument("--rrf_k", type=float, default=60)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--bm25_k1", type=float, default=bm25.K1)
    ap.add_argument("--bm25_b", type=float, default=bm25.B)
    ap.add_argument("--rerank", nargs="+", default=None,
                    help="cross-encoder(s) as <model>[@<query>]; scores are averaged after min-max")
    ap.add_argument("--rerank_query", default=None, help="query strategy for rerankers given without @<query>")
    ap.add_argument("--rerank_depth", type=int, default=50)
    ap.add_argument("--rerank_batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the cross-encoder vs the fusion score")
    ap.add_argument("--name", default=None)
    ap.add_argument("--write", action="store_true", help="write the official-format prediction file")
    ap.add_argument("--out", default=None,
                    help="directory for the log, metrics, run and predictions (default: results/ and work/runs/)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing existing result files")
    if preset:
        ap.set_defaults(**PRESETS[preset])  # explicit command-line flags still override the preset
    args = ap.parse_args()

    name = args.name or "+".join(short(s) for s in args.sources) + (
        "__rr_" + "+".join(short(m.split("@", 1)[0]) for m in args.rerank) if args.rerank else "")
    out = ROOT / (args.out or "results")
    run_dir = out if args.out else ROOT / "work" / "runs"
    targets = [out / f"{name}.metrics.json", run_dir / f"{name}.json"] + (
        [out / f"{name}.predictions.jsonl"] if args.write else [])
    existing = [str(p) for p in targets if p.exists()]
    if existing and not args.overwrite:  # checked before any retrieval, so no GPU time is wasted
        raise SystemExit(f"refusing to overwrite {existing}; use another --name/--out or --overwrite")

    tasks = load_tasks(args.input)
    run, texts = pipeline(tasks, args)

    results = evaluate_run({k: v[:TOP_K] for k, v in run.items()}, tasks)
    table = format_table(results, f"== {name} ==")
    print(table)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n```\n{datetime.now():%Y-%m-%d %H:%M}  {' '.join(vars(args).__repr__().split())}\n{table}\n```\n")
    with open(out / f"{name}.metrics.json", "w") as f:
        json.dump(results, f, indent=1)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / f"{name}.json", "w") as f:
        json.dump(run, f)
    if args.write:
        write_predictions(out / f"{name}.predictions.jsonl", tasks, run, texts or corpus_texts())
        print("predictions:", out / f"{name}.predictions.jsonl")


if __name__ == "__main__":
    main()
