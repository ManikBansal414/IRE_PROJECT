"""BM25 (bm25s + Snowball stemmer), one index per domain, cached under work/bm25/."""
import pickle

import bm25s
import Stemmer

from .data import WORK, load_corpus

_stemmer = Stemmer.Stemmer("english")
_cache = {}


def get_index(domain):
    if domain in _cache:
        return _cache[domain]
    path = WORK / "bm25" / domain
    if (path / "ids.pkl").exists():
        retriever = bm25s.BM25.load(str(path))
        ids = pickle.loads((path / "ids.pkl").read_bytes())
    else:
        ids, texts = load_corpus(domain)
        retriever = bm25s.BM25(k1=0.9, b=0.4)
        retriever.index(bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer, show_progress=False),
                        show_progress=False)
        path.mkdir(parents=True, exist_ok=True)
        retriever.save(str(path))
        (path / "ids.pkl").write_bytes(pickle.dumps(ids))
    _cache[domain] = (retriever, ids)
    return _cache[domain]


def search(domain, queries, k=100):
    """Returns one [(pid, score), ...] list per query."""
    retriever, ids = get_index(domain)
    tokens = bm25s.tokenize(queries, stopwords="en", stemmer=_stemmer, show_progress=False)
    docs, scores = retriever.retrieve(tokens, k=k, show_progress=False, n_threads=4)
    return [[(ids[d], float(s)) for d, s in zip(drow, srow) if s > 0]
            for drow, srow in zip(docs, scores)]
