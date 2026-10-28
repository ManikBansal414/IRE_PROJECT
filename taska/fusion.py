"""Reciprocal Rank Fusion of several ranked lists."""


def rrf(ranked_lists, k=60, weights=None):
    """ranked_lists: list of [(pid, score), ...]. Returns one fused [(pid, rrf_score), ...]."""
    weights = weights or [1.0] * len(ranked_lists)
    fused = {}
    for w, lst in zip(weights, ranked_lists):
        for rank, (pid, _) in enumerate(lst):
            fused[pid] = fused.get(pid, 0.0) + w / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])
