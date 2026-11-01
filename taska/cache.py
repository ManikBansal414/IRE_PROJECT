"""Cache retrieval runs and cross-encoder scores so parameter sweeps need no GPU.

  python -m taska.cache runs                       # top-100 per (retriever, query variant)
  python -m taska.cache bm25grid                   # BM25 k1/b grid, built in memory (not saved)
  python -m taska.cache rerank --pool work/cache/pool.json --model <cross-encoder> --query t5

Outputs go to work/cache/: runs/<name>.json and bm25grid/<name>.json hold {task_id: [[pid, score], ...]},
rerank/<model>@<query>/part_*.json hold {task_id: {pid: score}}. Every step skips files that already exist.
"""
import argparse
import json

import bm25s

from . import bm25
from .data import DOMAINS, ROOT, WORK, load_corpus, load_tasks
from .run import build_queries, corpus_texts, retrieve

CACHE = WORK / "cache"
DENSE = "dense:BAAI/bge-base-en-v1.5"
T5 = "file:data/rewrites_t5canard.jsonl"
STRATEGIES = {"last": "last", "last2": "last2", "weighted": "weighted", "all": "all", "t5": T5,
              "gold": "gold_rewrite"}  # gold = upper bound only, never used for the submission
RETRIEVERS = {"bm25": "bm25", "bge-base": DENSE}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.rename(path)


def cache_runs(tasks, depth):
    for rname, retriever in RETRIEVERS.items():
        for sname, strategy in STRATEGIES.items():
            out = CACHE / "runs" / f"{rname}@{sname}.json"
            if out.exists():
                continue
            save(out, retrieve(tasks, build_queries(tasks, strategy), retriever, depth))
            print("wrote", out, flush=True)


def cache_bm25_grid(tasks, depth, k1s, bs, strategies):
    qtexts = {s: build_queries(tasks, STRATEGIES[s]) for s in strategies}
    runs = {}
    for domain in DOMAINS:
        dtasks = [t for t in tasks if t["domain"] == domain]
        ids, texts = load_corpus(domain)
        corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=bm25._stemmer, show_progress=False)
        for k1 in k1s:
            for b in bs:
                retriever = bm25s.BM25(k1=k1, b=b)
                retriever.index(corpus_tokens, show_progress=False)
                for s in strategies:
                    qs = bm25s.tokenize([qtexts[s][t["task_id"]] for t in dtasks], stopwords="en",
                                        stemmer=bm25._stemmer, show_progress=False)
                    docs, scores = retriever.retrieve(qs, k=depth, show_progress=False, n_threads=8)
                    run = runs.setdefault((k1, b, s), {})
                    for t, drow, srow in zip(dtasks, docs, scores):
                        run[t["task_id"]] = [(ids[d], float(x)) for d, x in zip(drow, srow) if x > 0]
        print(f"bm25grid: {domain} done", flush=True)
    for (k1, b, s), run in runs.items():
        save(CACHE / "bm25grid" / f"k1={k1}_b={b}@{s}.json", run)


def cache_rerank(tasks, pool_path, model, strategy, chunk, batch_size):
    from .rerank import get_model
    pool = json.loads((ROOT / pool_path).read_text())
    qtext = build_queries(tasks, STRATEGIES[strategy])
    out_dir = CACHE / "rerank" / f"{model.replace('/', '__')}@{strategy}"
    tids = [t["task_id"] for t in tasks if t["task_id"] in pool]
    texts = corpus_texts()
    ce = get_model(model)
    for c in range(0, len(tids), chunk):
        out = out_dir / f"part_{c // chunk:03d}.json"
        if out.exists():
            continue
        pairs = [(tid, pid) for tid in tids[c:c + chunk] for pid in pool[tid]]
        # Length-sorted batches pad less, which matters for the larger cross-encoders.
        order = sorted(range(len(pairs)), key=lambda i: len(texts[pairs[i][1]]))
        scores = ce.predict([(qtext[pairs[i][0]], texts[pairs[i][1]]) for i in order],
                            batch_size=batch_size, show_progress_bar=False)
        res = {}
        for i, s in zip(order, scores):
            tid, pid = pairs[i]
            res.setdefault(tid, {})[pid] = float(s)
        save(out, res)
        print(f"{out_dir.name}: {min(c + chunk, len(tids))}/{len(tids)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["runs", "bm25grid", "rerank"])
    ap.add_argument("--input", default=str(ROOT / "data" / "taskA_dev_input.jsonl"))
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--k1", nargs="+", type=float, default=[0.6, 0.9, 1.2, 1.5])
    ap.add_argument("--b", nargs="+", type=float, default=[0.3, 0.4, 0.6, 0.75])
    ap.add_argument("--grid_queries", nargs="+", default=["last", "last2", "weighted", "all", "t5"])
    ap.add_argument("--pool", default="work/cache/pool.json")
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--query", default="t5", choices=list(STRATEGIES))
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()

    tasks = load_tasks(a.input)
    if a.step == "runs":
        cache_runs(tasks, a.depth)
    elif a.step == "bm25grid":
        cache_bm25_grid(tasks, a.depth, a.k1, a.b, a.grid_queries)
    else:
        cache_rerank(tasks, a.pool, a.model, a.query, a.chunk, a.batch_size)


if __name__ == "__main__":
    main()
