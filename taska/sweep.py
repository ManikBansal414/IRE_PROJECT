"""Parameter sweeps on cached runs (see taska/cache.py), with cross-validation and significance tests.

Every selection is made by 5-fold cross-validation over conversations (all turns of a conversation stay in
the same fold): parameters are chosen on 4 folds and scored on the held-out fold, so the reported CV score
is an unbiased estimate of the tuning procedure. Differences are tested with a paired bootstrap over questions.

  python -m taska.sweep bm25      # BM25 k1/b grid
  python -m taska.sweep fusion    # source weights, RRF k, per-source depth
  python -m taska.sweep pool      # candidate pools for the reranker job -> work/cache/pool{50,100}.json
  python -m taska.sweep rerank    # reranker, query, depth, score interpolation
  python -m taska.sweep fusion --metric nDCG@5   # same search with another selection metric

Results are written to results/tuning/<step>.md and <step>.json (<step>_ndcg5.* etc. for --metric nDCG@5).
With another metric, each step also tests whether the configuration chosen with nDCG@10 is tied with the best.
"""
import argparse
import itertools
import json
from collections import defaultdict

import numpy as np

from .data import DOMAINS, ROOT, WORK, load_qrels, load_tasks, norm_qid

CACHE = WORK / "cache"  # written by taska/cache.py
OUT = ROOT / "results" / "tuning"
N_FOLDS, SEED, N_BOOT = 5, 0, 10000
DISC = 1.0 / np.log2(np.arange(2, 102))
BIG = np.float32(1e9)

# Sources available at test time (gold rewrites excluded).
QUERIES = ["last", "last2", "weighted", "all", "t5"]
SOURCES = [f"{r}@{q}" for r in ("bm25", "bge-base") for q in QUERIES]
# The submitted system (README row 15) and the best single run.
HYBRID5 = {"w": {"bge-base@t5": 1, "bge-base@last": 1, "bm25@t5": 1, "bm25@weighted": 1, "bm25@last": 1},
           "k": 60, "depth": 100}
DENSE_T5 = {"w": {"bge-base@t5": 1}, "k": 60, "depth": 100}
DEFAULT_BM25 = "k1=0.9_b=0.4"
WEIGHTS = [0, 0.25, 0.5, 1, 1.5, 2, 3]
RRF_K = [1, 2, 3, 5, 10, 20, 30, 60, 100]
DEPTHS = [20, 30, 50, 100]
OBJ = "nDCG@10"  # selection metric; set with --metric
SUFFIX = ""      # result file suffix for another selection metric, e.g. "_ndcg5"


# ---------------------------------------------------------------- evaluation
class Bench:
    """Scored dev questions (those with qrels), their conversation folds and relevance labels."""

    def __init__(self, input_path):
        tasks = load_tasks(input_path)
        qrels = {}
        for d in DOMAINS:
            qrels.update(load_qrels(d))
        self.tasks = [t for t in tasks if norm_qid(t["task_id"]) in qrels]
        self.tids = [t["task_id"] for t in self.tasks]
        self.domain = np.array([t["domain"] for t in self.tasks])
        self.rel = [qrels[norm_qid(t)] for t in self.tids]
        self.nrel = np.array([sum(1 for v in r.values() if v > 0) for r in self.rel], dtype=float)
        self.idcg = {k: np.array([(np.sort(np.array(list(r.values()), float))[::-1][:k] * DISC[:min(k, len(r))]).sum()
                                  for r in self.rel]) for k in (1, 3, 5, 10)}
        convs = sorted({t["conversation_id"] for t in self.tasks})
        rng = np.random.default_rng(SEED)
        fold_of = {c: i % N_FOLDS for i, c in enumerate(rng.permutation(convs))}
        self.fold = np.array([fold_of[t["conversation_id"]] for t in self.tasks])

    def metrics(self, ranked):
        """ranked: one list of pids per scored question -> {metric: per-question array}."""
        gains = np.zeros((len(ranked), 100))
        for i, (pids, rel) in enumerate(zip(ranked, self.rel)):
            for j, p in enumerate(pids[:100]):
                gains[i, j] = rel.get(p, 0)
        return self._from_gains(gains)

    def _from_gains(self, gains):
        m = {}
        for k in (1, 3, 5, 10):
            m[f"nDCG@{k}"] = (gains[:, :k] * DISC[:k]).sum(1) / self.idcg[k]
        for k in (10, 50, 100):
            m[f"R@{k}"] = (gains[:, :k] > 0).sum(1) / self.nrel
        return m


def load_run(path):
    return {tid: [p for p, _ in lst] for tid, lst in json.loads(path.read_text()).items()}


class Fusion:
    """Vectorised weighted RRF over cached runs: rank tensor of shape (sources, questions, candidates)."""

    def __init__(self, bench, runs):
        self.bench, self.names = bench, list(runs)
        unions = []
        for tid in bench.tids:
            u = {}
            for name in self.names:
                for p in runs[name].get(tid, []):
                    u.setdefault(p, len(u))
            unions.append(u)
        self.pids = [list(u) for u in unions]
        width = max(len(u) for u in unions)
        self.rank = np.full((len(self.names), len(unions), width), BIG, np.float32)
        self.gain = np.zeros((len(unions), width))
        for q, (tid, u, rel) in enumerate(zip(bench.tids, unions, bench.rel)):
            for s, name in enumerate(self.names):
                for r, p in enumerate(runs[name].get(tid, [])):
                    self.rank[s, q, u[p]] = r
            for p, j in u.items():
                self.gain[q, j] = rel.get(p, 0)

    def _contrib(self, k, depth):
        # Coordinate ascent mostly varies weights, so caching the last (k, depth) saves most of the work.
        if getattr(self, "_key", None) != (k, depth):
            inside = self.rank < depth
            self._inside = inside.astype(np.float32)
            self._c = np.where(inside, 1.0 / (k + self.rank + 1), 0).astype(np.float32)
            self._key = (k, depth)
        return self._c, self._inside

    def scores(self, cfg):
        w = np.array([cfg["w"].get(n, 0) for n in self.names], np.float32)
        contrib, inside = self._contrib(cfg["k"], cfg["depth"])
        s = np.tensordot(w, contrib, 1)
        present = np.tensordot((w > 0).astype(np.float32), inside, 1) > 0
        return np.where(present, s, -np.inf)

    def top(self, cfg, k=100):
        s = self.scores(cfg)
        order = np.argsort(-s, axis=1, kind="stable")[:, :k]
        return order, np.take_along_axis(s, order, 1)

    def metrics(self, cfg):
        order, s = self.top(cfg)
        gains = np.take_along_axis(self.gain, order, 1) * np.isfinite(s)
        if gains.shape[1] < 100:
            gains = np.pad(gains, ((0, 0), (0, 100 - gains.shape[1])))
        return self.bench._from_gains(gains)

    def ranked(self, cfg, k=100):
        order, s = self.top(cfg, k)
        return [[(self.pids[q][j], float(x)) for j, x in zip(order[q], s[q]) if np.isfinite(x)]
                for q in range(len(self.pids))]


# ---------------------------------------------------------------- statistics
def bootstrap(a, b):
    """Paired bootstrap of mean(a - b): (mean diff, 95% CI low, high, P(diff <= 0))."""
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(SEED).integers(0, len(d), (N_BOOT, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means <= 0).mean())


def fmt_boot(a, b):
    m, lo, hi, p = bootstrap(a, b)
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3f}"


def cv_select(bench, per_query):
    """per_query: {config_name: per-question scores}. Choose on 4 folds, score the 5th.
    Returns held-out per-question scores and the config chosen in each fold."""
    names = list(per_query)
    mat = np.stack([per_query[n] for n in names])
    held, chosen = np.zeros(mat.shape[1]), []
    for f in range(N_FOLDS):
        train, test = bench.fold != f, bench.fold == f
        best = int(np.argmax(mat[:, train].mean(1)))
        held[test] = mat[best, test]
        chosen.append(names[best])
    return held, chosen


def near_best(per_query, metric_name, top=15):
    """Configs ranked by full-dev score, each tested against the best one."""
    ranked = sorted(per_query, key=lambda n: -per_query[n].mean())
    best = per_query[ranked[0]]
    rows = [f"| config | {metric_name} | diff vs best [95% CI] | p(best not better) |", "|---|---|---|---|"]
    for n in ranked[:top]:
        m, lo, hi, p = bootstrap(per_query[n], best)
        rows.append(f"| {n} | {per_query[n].mean():.4f} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | {1 - p if n != ranked[0] else 1:.3f} |")
    # Same resamples as bootstrap(), as a count matrix, so all configs are tested in a few matrix products.
    idx = np.random.default_rng(SEED).integers(0, len(best), (N_BOOT, len(best)))
    counts = np.zeros((N_BOOT, len(best)), np.float32)
    np.add.at(counts, (np.arange(N_BOOT)[:, None], idx), 1)
    n_tied = 0
    for i in range(0, len(ranked), 500):
        diffs = np.stack([best - per_query[n] for n in ranked[i:i + 500]]).astype(np.float32)
        lows = np.percentile(counts @ diffs.T / len(best), 2.5, axis=0)
        n_tied += int((lows <= 0).sum())
    return ranked, rows, n_tied


def write(step, lines, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{step}{SUFFIX}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / f"{step}{SUFFIX}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("\n".join(lines))


def vs_ndcg10_choice(chosen10, scores, best, label):
    """With another selection metric: is the nDCG@10 choice tied with the best under that metric?"""
    if not SUFFIX:
        return []
    m, lo, hi, p = bootstrap(scores[chosen10], scores[best])
    verdict = "tied with the best (95% CI includes 0)" if lo <= 0 <= hi or chosen10 == best else "worse than the best"
    return ["", f"## Check: the {label} chosen with nDCG@10", "",
            f"- Chosen with nDCG@10: {chosen10}: {OBJ} {scores[chosen10].mean():.4f}",
            f"- Best by {OBJ}: {best}: {scores[best].mean():.4f}",
            f"- Difference: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] -> **{verdict}**"]


def by_domain(bench, scores):
    return " ".join(f"{d} {scores[bench.domain == d].mean():.3f}" for d in DOMAINS)


# ---------------------------------------------------------------- steps
def step_bm25(bench):
    grid = defaultdict(dict)
    for f in sorted((CACHE / "bm25grid").glob("*.json")):
        params, q = f.stem.split("@")
        run = load_run(f)
        grid[params][q] = bench.metrics([run.get(t, []) for t in bench.tids])
    qs = sorted(next(iter(grid.values())))
    # Objective: mean nDCG over the query variants, so one (k1, b) serves every BM25 source.
    obj = {p: np.mean([grid[p][q][OBJ] for q in qs], 0) for p in grid}
    held, chosen = cv_select(bench, obj)
    ranked, rows, n_tied = near_best(obj, f"mean {OBJ}")
    default = DEFAULT_BM25
    lines = ["# BM25 k1/b grid", "",
             f"Query variants: {', '.join(qs)}. Objective: mean {OBJ} over these variants.", "",
             "| k1, b | " + " | ".join(f"{q} {OBJ}" for q in qs) + " | " + " | ".join(f"{q} R@100" for q in qs) + " |",
             "|---|" + "---|" * (2 * len(qs))]
    for p in ranked:
        lines.append(f"| {p} | " + " | ".join(f"{grid[p][q][OBJ].mean():.4f}" for q in qs) + " | "
                     + " | ".join(f"{grid[p][q]['R@100'].mean():.4f}" for q in qs) + " |")
    lines += ["", "## Best configs vs the best one (paired bootstrap)", ""] + rows + [
        "", f"Configs statistically tied with the best: {n_tied} of {len(obj)}.",
        f"CV (choose on 4 folds, score 5th): held-out {held.mean():.4f}; chosen per fold: {chosen}.",
        f"Default {default}: {obj[default].mean():.4f}; best {ranked[0]} vs default: {fmt_boot(obj[ranked[0]], obj[default])}."]
    if SUFFIX:
        lines += vs_ndcg10_choice(json.loads((OUT / "bm25.json").read_text())["best"], obj, ranked[0], "BM25 setting")
    write("bm25", lines, {"best": ranked[0], "cv_heldout": held.mean(), "chosen_per_fold": chosen,
                          "scores": {p: float(v.mean()) for p, v in obj.items()}})


def bm25_runs(bm25_params):
    """BM25 sources from the grid at the given params (falls back to the default-parameter runs)."""
    runs = {}
    for q in QUERIES:
        f = CACHE / "bm25grid" / f"{bm25_params}@{q}.json"
        runs[f"bm25@{q}"] = load_run(f if f.exists() else CACHE / "runs" / f"bm25@{q}.json")
    return runs


def load_fusion(bench, bm25_params):
    runs = bm25_runs(bm25_params)
    for q in QUERIES:
        runs[f"bge-base@{q}"] = load_run(CACHE / "runs" / f"bge-base@{q}.json")
    return Fusion(bench, runs)


def cfg_name(cfg):
    w = ", ".join(f"{n}×{v:g}" for n, v in sorted(cfg["w"].items()) if v)
    return f"{{{w}}} k={cfg['k']} depth={cfg['depth']}"


def ascent(fusion, mask, metric, starts):
    """Coordinate ascent over source weights, RRF k and depth, maximising mean `metric` on `mask`."""
    def score(c):
        return fusion.metrics(c)[metric][mask].mean()

    best_cfg, best = None, -1
    for start in starts:
        cfg = {"w": dict(start["w"]), "k": start["k"], "depth": start["depth"]}
        cur = score(cfg)
        for _ in range(8):
            improved = False
            for name in fusion.names:
                for v in WEIGHTS:
                    c2 = {**cfg, "w": {**cfg["w"], name: v}}
                    if not any(c2["w"].values()):
                        continue
                    s = score(c2)
                    if s > cur + 1e-9:
                        cfg, cur, improved = c2, s, True
            for key, values in (("k", RRF_K), ("depth", DEPTHS)):
                for v in values:
                    s = score({**cfg, key: v})
                    if s > cur + 1e-9:
                        cfg, cur, improved = {**cfg, key: v}, s, True
            if not improved:
                break
        if cur > best:
            best_cfg, best = cfg, cur
    best_cfg["w"] = {n: v for n, v in best_cfg["w"].items() if v}
    return best_cfg


def step_fusion(bench, bm25_params):
    fusion = load_fusion(bench, bm25_params)
    starts = [HYBRID5, DENSE_T5, {"w": {n: 1 for n in SOURCES}, "k": 60, "depth": 100},
              {"w": {"bge-base@t5": 1, "bm25@t5": 1}, "k": 60, "depth": 100}]
    singles = {n: {"w": {n: 1}, "k": 60, "depth": 100} for n in SOURCES}
    lines = [f"# Fusion sweep (BM25 params: {bm25_params})", "",
             f"Search: coordinate ascent over weights {WEIGHTS} per source, RRF k {RRF_K}, per-source depth {DEPTHS}; "
             f"{len(starts)} starting points. Selection by {N_FOLDS}-fold CV over conversations.", "",
             f"## Single sources ({OBJ} / R@50 / R@100)", "", f"| source | {OBJ} | R@50 | R@100 |", "|---|---|---|---|"]
    for n, c in singles.items():
        m = fusion.metrics(c)
        lines.append(f"| {n} | {m[OBJ].mean():.4f} | {m['R@50'].mean():.4f} | {m['R@100'].mean():.4f} |")
    data = {"bm25_params": bm25_params}
    for metric in ((OBJ, "R@50") if not SUFFIX else (OBJ,)):
        held = np.zeros(len(bench.tids))
        chosen = []
        for f in range(N_FOLDS):
            cfg = ascent(fusion, bench.fold != f, metric, starts)
            held[bench.fold == f] = fusion.metrics(cfg)[metric][bench.fold == f]
            chosen.append(cfg_name(cfg))
        final = ascent(fusion, np.ones(len(bench.tids), bool), metric, starts)
        m_final, m_h5, m_d = (fusion.metrics(c)[metric] for c in (final, HYBRID5, DENSE_T5))
        lines += ["", f"## Objective {metric}", "",
                  f"- Tuned on all dev (in-sample, optimistic): **{m_final.mean():.4f}** with {cfg_name(final)}",
                  f"- CV held-out estimate of the tuning procedure: **{held.mean():.4f}** ({by_domain(bench, held)})",
                  f"- Current hybrid5 (equal weights, k=60): {m_h5.mean():.4f}; dense@t5 alone: {m_d.mean():.4f}",
                  f"- CV held-out vs hybrid5: {fmt_boot(held, m_h5)}",
                  f"- CV held-out vs dense@t5: {fmt_boot(held, m_d)}",
                  "- Configs chosen per fold:"] + [f"  - {c}" for c in chosen]
        data[metric] = {"final": final, "final_score": m_final.mean(), "cv_heldout": held.mean(),
                        "hybrid5": m_h5.mean(), "dense_t5": m_d.mean(), "chosen_per_fold": chosen}
        if SUFFIX:
            c10 = json.loads((OUT / "fusion.json").read_text())["nDCG@10"]["final"]
            lines += vs_ndcg10_choice("nDCG@10 choice", {"nDCG@10 choice": fusion.metrics(c10)[metric],
                                                         "best": m_final}, "best", "fusion setting")
    write("fusion", lines, data)


def pool_lists(bench, bm25_params, configs):
    """Fused top-100 per config; a config may pin its own BM25 params with the key 'bm25'."""
    fusions = {}
    out = {}
    for n, c in configs.items():
        p = c.get("bm25", bm25_params)
        if p not in fusions:
            fusions[p] = load_fusion(bench, p)
        out[n] = fusions[p].ranked(c)
    return out


def step_pool(bench, bm25_params):
    """Candidate pools for reranking: union of the top-k of hybrid5, dense@t5 and both tuned fusions."""
    tuned = json.loads((OUT / "fusion.json").read_text())
    configs = {"hybrid5": HYBRID5, "dense_t5": DENSE_T5,
               "tuned_ndcg": tuned["nDCG@10"]["final"], "tuned_r50": tuned["R@50"]["final"],
               "hybrid5_orig": {**HYBRID5, "bm25": DEFAULT_BM25}}  # the submitted system's pool, as it was
    ranked = pool_lists(bench, bm25_params, configs)
    for k in (50, 100):
        pool = {}
        for q, tid in enumerate(bench.tids):
            u = {}
            for r in ranked.values():
                for p, _ in r[q][:k]:
                    u.setdefault(p, None)
            pool[tid] = list(u)
        path = CACHE / f"pool{k}.json"
        path.write_text(json.dumps(pool))
        print(f"{path}: {sum(map(len, pool.values()))} pairs, {np.mean([len(v) for v in pool.values()]):.1f} per question")
    (CACHE / "pool_configs.json").write_text(json.dumps({"bm25_params": bm25_params, "configs": configs}, indent=1))


def load_ce(name):
    scores = {}
    for f in sorted((CACHE / "rerank" / name).glob("part_*.json")):
        scores.update(json.loads(f.read_text()))
    return scores


def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def step_rerank(bench):
    pc = json.loads((CACHE / "pool_configs.json").read_text())
    pools = pool_lists(bench, pc["bm25_params"], pc["configs"])
    ce = {d.name: load_ce(d.name) for d in sorted((CACHE / "rerank").iterdir()) if d.is_dir()}

    def covered(plists, m):
        """Deepest prefix of every question's candidate list that has cross-encoder scores."""
        depth = 100
        for q, tid in enumerate(bench.tids):
            s = ce[m].get(tid, {})
            depth = min(depth, next((i for i, (p, _) in enumerate(plists[q]) if p not in s), len(plists[q])))
        return depth

    def short(n):
        return n.split("__")[-1].replace("ms-marco-", "")

    rerankers = {short(n): [n] for n in ce}
    usable = [n for n in ce if not n.endswith("@gold")]
    for r in (2, 3):
        for combo in itertools.combinations(usable, r):
            rerankers[" + ".join(map(short, combo))] = list(combo)

    per_query, upper = {}, {}
    for pname, plists in pools.items():
        for rname, models in rerankers.items():
            if len(models) == 3 and not pname.startswith("tuned"):
                continue  # 3-way ensembles only on the tuned pools, which are best for every reranker
            max_depth = min(covered(plists, m) for m in models)  # v2-m3 is cached on the top-50 pool only
            for depth in [d for d in (10, 20, 30, 50, 75, 100) if d <= max_depth]:
                for alpha in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
                    ranked = []
                    for q, tid in enumerate(bench.tids):
                        head = plists[q][:depth]
                        pids = [p for p, _ in head]
                        s_ce = np.mean([minmax(np.array([ce[m][tid][p] for p in pids])) for m in models], 0)
                        s = alpha * s_ce + (1 - alpha) * minmax(np.array([x for _, x in head]))
                        ranked.append([pids[i] for i in np.argsort(-s, kind="stable")])
                    # Gold-rewrite queries are not available at test time: upper bound only, never selected.
                    target = upper if any(m.endswith("@gold") for m in models) else per_query
                    target[f"{pname} / {rname} / depth={depth} / alpha={alpha}"] = bench.metrics(ranked)
    ndcg = {n: m[OBJ] for n, m in per_query.items()}
    held, chosen = cv_select(bench, ndcg)
    ranked_cfgs, rows, n_tied = near_best(ndcg, OBJ, top=25)
    current = "hybrid5_orig / MiniLM-L6-v2@t5 / depth=50 / alpha=1.0"
    best = ranked_cfgs[0]
    lines = ["# Reranking sweep", "",
             f"Grid: {len(pools)} candidate pools × {len(rerankers)} rerankers/ensembles × depth × alpha "
             f"(final score = alpha·minmax(CE) + (1−alpha)·minmax(fusion)); {len(ndcg)} configs.", "",
             f"- Best on all dev (in-sample): **{best}** = {ndcg[best].mean():.4f}",
             f"- CV held-out estimate of the selection procedure: **{held.mean():.4f}** ({by_domain(bench, held)})",
             f"- Current system ({current}): {ndcg[current].mean():.4f}" if current in ndcg else "- Current system not in grid",
             f"- CV held-out vs current: {fmt_boot(held, ndcg[current])}" if current in ndcg else "",
             f"- Best vs current: {fmt_boot(ndcg[best], ndcg[current])}" if current in ndcg else "",
             f"- Configs statistically tied with the best (95% CI of difference includes 0): {n_tied} of {len(ndcg)}",
             "- Chosen per fold:"] + [f"  - {c}" for c in chosen]

    def best_where(pred):
        cands = [n for n in ndcg if pred(n)]
        return max(cands, key=lambda n: ndcg[n].mean()) if cands else None

    lines += ["", "## Best config per reranker", "", f"| reranker | best config | {OBJ} | vs overall best |", "|---|---|---|---|"]
    for rname in rerankers:
        n = best_where(lambda x: x.split(" / ")[1] == rname)
        if n is None:  # gold-query reranker: reported as the upper bound below
            continue
        lines.append(f"| {rname} | {n} | {ndcg[n].mean():.4f} | {fmt_boot(ndcg[n], ndcg[best])} |")
    if upper:
        n = max(upper, key=lambda x: upper[x][OBJ].mean())
        lines += ["", f"Upper bound, reranking with the gold rewrite as query: {n} = {upper[n][OBJ].mean():.4f} "
                      f"(vs best {fmt_boot(upper[n][OBJ], ndcg[best])})"]
    lines += ["", "## Best config per candidate pool", "", f"| pool | best config | {OBJ} |", "|---|---|---|"]
    for pname in pools:
        n = best_where(lambda x: x.split(" / ")[0] == pname)
        lines.append(f"| {pname} | {n} | {ndcg[n].mean():.4f} |")
    lines += ["", "## Top configs vs best (paired bootstrap)", ""] + rows
    lines += ["", "## Best config, all metrics", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v.mean():.4f} |" for k, v in per_query[best].items()]
    if SUFFIX:  # the candidate pools (and their cached reranker scores) are those built from the nDCG@10 fusion
        lines += vs_ndcg10_choice(json.loads((OUT / "rerank.json").read_text())["best"], ndcg, best, "reranking setup")
    write("rerank", lines, {"best": best, "best_score": ndcg[best].mean(), "cv_heldout": held.mean(),
                            "chosen_per_fold": chosen, "n_tied": n_tied,
                            "scores": {n: float(v.mean()) for n, v in ndcg.items()}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["bm25", "fusion", "pool", "rerank"])
    ap.add_argument("--input", default=str(ROOT / "data" / "taskA_dev_input.jsonl"))
    ap.add_argument("--bm25_params", default=None, help="e.g. k1=0.9_b=0.4 (default: best from the bm25 step)")
    ap.add_argument("--metric", default="nDCG@10", choices=["nDCG@1", "nDCG@3", "nDCG@5", "nDCG@10"],
                    help="selection metric; results of another metric go to <step>_ndcg<k>.*")
    a = ap.parse_args()
    global OBJ, SUFFIX
    OBJ, SUFFIX = a.metric, "" if a.metric == "nDCG@10" else "_" + a.metric.replace("@", "").lower()
    if SUFFIX and a.step == "pool":
        raise SystemExit("pool builds the reranker job's input; run it with nDCG@10 only")
    bench = Bench(a.input)
    params = a.bm25_params
    if params is None and a.step in ("fusion", "pool"):
        f = OUT / f"bm25{SUFFIX}.json"
        params = json.loads(f.read_text())["best"] if f.exists() else DEFAULT_BM25
    {"bm25": lambda: step_bm25(bench), "fusion": lambda: step_fusion(bench, params),
     "pool": lambda: step_pool(bench, params), "rerank": lambda: step_rerank(bench)}[a.step]()


if __name__ == "__main__":
    main()
