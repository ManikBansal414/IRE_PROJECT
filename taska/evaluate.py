"""nDCG / Recall @ 1,3,5,10 with pytrec_eval, matching the official run_retrieval_eval.py
(per-domain mean, then a query-count-weighted average across domains)."""
import pytrec_eval

from .data import DOMAINS, load_qrels, norm_qid

KS = (1, 3, 5, 10)


def evaluate_run(run, tasks):
    """run: {task_id: [(pid, score), ...]}; tasks: Task A rows. Returns {domain|'all': {metric: value}}."""
    by_domain = {}
    for t in tasks:
        by_domain.setdefault(t["domain"], []).append(t["task_id"])

    out, total = {}, {}
    for domain in DOMAINS:
        if domain not in by_domain:
            continue
        qrels = load_qrels(domain)
        res = {}
        for tid in by_domain[domain]:
            qid = norm_qid(tid)
            if qid in qrels:  # only answerable questions are scored
                res[qid] = {pid: float(s) for pid, s in run.get(tid, [])}
        evaluator = pytrec_eval.RelevanceEvaluator(
            {q: qrels[q] for q in res},
            {"ndcg_cut." + ",".join(map(str, KS)), "recall." + ",".join(map(str, KS))})
        scores = evaluator.evaluate(res)
        m = {}
        for k in KS:
            m[f"nDCG@{k}"] = sum(s[f"ndcg_cut_{k}"] for s in scores.values()) / len(scores)
            m[f"R@{k}"] = sum(s[f"recall_{k}"] for s in scores.values()) / len(scores)
        m["n"] = len(scores)
        out[domain] = m
        for key, v in m.items():
            total[key] = total.get(key, 0) + (v * m["n"] if key != "n" else v)
    out["all"] = {k: (v / total["n"] if k != "n" else v) for k, v in total.items()}
    return out


def format_table(results, title=""):
    cols = [f"nDCG@{k}" for k in KS] + [f"R@{k}" for k in KS]
    lines = [title, f"{'domain':8} {'n':>4} " + " ".join(f"{c:>8}" for c in cols)]
    for d, m in results.items():
        lines.append(f"{d:8} {m['n']:>4} " + " ".join(f"{m[c]:8.4f}" for c in cols))
    return "\n".join(lines)
