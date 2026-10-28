"""Cross-encoder reranking of the top candidates."""
import torch
from sentence_transformers import CrossEncoder

_models = {}


def get_model(name, max_len=512):
    if name not in _models:
        _models[name] = CrossEncoder(name, max_length=max_len,
                                     device="cuda" if torch.cuda.is_available() else "cpu")
    return _models[name]


def rerank(name, query, candidates, texts, top_n=50, batch_size=32):
    """candidates: [(pid, score), ...]; texts: {pid: text}. Returns reranked [(pid, ce_score), ...]."""
    cands = candidates[:top_n]
    if not cands:
        return []
    scores = get_model(name).predict([(query, texts[pid]) for pid, _ in cands],
                                     batch_size=batch_size, show_progress_bar=False)
    return sorted(((pid, float(s)) for (pid, _), s in zip(cands, scores)), key=lambda x: -x[1])
