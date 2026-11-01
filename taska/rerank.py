"""Cross-encoder reranking of the top candidates."""
import torch
from sentence_transformers import CrossEncoder

_models = {}


def get_model(name, max_len=512):
    if name not in _models:
        _models[name] = CrossEncoder(name, max_length=max_len,
                                     device="cuda" if torch.cuda.is_available() else "cpu")
    return _models[name]


def score(name, query, pids, texts, batch_size=32):
    """Raw cross-encoder score of (query, passage) for each pid."""
    if not pids:
        return []
    return [float(s) for s in get_model(name).predict([(query, texts[pid]) for pid in pids],
                                                       batch_size=batch_size, show_progress_bar=False)]


def rerank(name, query, candidates, texts, top_n=50, batch_size=32):
    """candidates: [(pid, score), ...]; texts: {pid: text}. Returns reranked [(pid, ce_score), ...]."""
    pids = [pid for pid, _ in candidates[:top_n]]
    return sorted(zip(pids, score(name, query, pids, texts, batch_size)), key=lambda x: -x[1])
