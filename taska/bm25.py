"""BM25 (bm25s + Snowball stemmer), one index per domain and (k1, b), cached under work/bm25/."""
import pickle

import bm25s
import Stemmer

from .data import WORK, load_corpus

K1, B = 0.9, 0.4  # Anserini defaults (bm25s itself defaults to 1.5, 0.75); tuned on dev: k1=2.2, b=0.5 (results/tuning/bm25.md)
_stemmer = Stemmer.Stemmer("english")
_cache = {}


def get_index(domain, k1=K1, b=B):
    if (domain, k1, b) in _cache:
        return _cache[(domain, k1, b)]
    # Default-parameter indexes keep their original path, so existing caches stay valid.
    path = WORK / "bm25" / (domain if (k1, b) == (K1, B) else f"{domain}_k1={k1}_b={b}")
    if (path / "ids.pkl").exists():
        retriever = bm25s.BM25.load(str(path))
        ids = pickle.loads((path / "ids.pkl").read_bytes())
    else:
        ids, texts = load_corpus(domain)
        retriever = bm25s.BM25(k1=k1, b=b)
        retriever.index(bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer, show_progress=False),
                        show_progress=False)
        path.mkdir(parents=True, exist_ok=True)
        retriever.save(str(path))
        (path / "ids.pkl").write_bytes(pickle.dumps(ids))
    _cache[(domain, k1, b)] = (retriever, ids)
    return _cache[(domain, k1, b)]


def search(domain, queries, k=100, k1=K1, b=B):
    """Returns one [(pid, score), ...] list per query."""
    retriever, ids = get_index(domain, k1, b)
    tokens = bm25s.tokenize(queries, stopwords="en", stemmer=_stemmer, show_progress=False)
    docs, scores = retriever.retrieve(tokens, k=k, show_progress=False, n_threads=4)
    return [[(ids[d], float(s)) for d, s in zip(drow, srow) if s > 0]
            for drow, srow in zip(docs, scores)]
