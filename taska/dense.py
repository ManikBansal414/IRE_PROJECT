"""Dense retrieval with sentence-transformers + exact FAISS inner-product search.

Corpus embeddings are encoded in shards and cached under work/dense/<model>/<domain>/,
so a long CPU encoding run can be stopped and resumed. On a GPU machine (e.g. ADA)
the same code runs unchanged, just faster.
"""
import argparse
import pickle

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .data import DOMAINS, WORK, load_corpus

QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-base-v2": "query: ",
}
DOC_PREFIX = {"intfloat/e5-base-v2": "passage: "}
SHARD = 4096

_models, _indexes = {}, {}


def get_model(name, max_len=512):
    if name not in _models:
        m = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")
        m.max_seq_length = max_len
        _models[name] = m
    return _models[name]


def model_dir(name, domain):
    return WORK / "dense" / name.replace("/", "__") / domain


def encode_corpus(name, domain, batch_size=32, max_len=512):
    out = model_dir(name, domain)
    out.mkdir(parents=True, exist_ok=True)
    ids, texts = load_corpus(domain)
    (out / "ids.pkl").write_bytes(pickle.dumps(ids))
    # Encode longest-first within the whole corpus would pad less, but shards keep resume simple.
    prefix = DOC_PREFIX.get(name, "")
    model = get_model(name, max_len)
    n_shards = (len(texts) + SHARD - 1) // SHARD
    for s in range(n_shards):
        f = out / f"shard_{s:04d}.npy"
        if f.exists():
            continue
        chunk = [prefix + t for t in texts[s * SHARD:(s + 1) * SHARD]]
        emb = model.encode(chunk, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        np.save(f, emb.astype(np.float16))
        print(f"{domain}: shard {s + 1}/{n_shards} done", flush=True)


def get_index(name, domain):
    key = (name, domain)
    if key not in _indexes:
        d = model_dir(name, domain)
        ids = pickle.loads((d / "ids.pkl").read_bytes())
        emb = np.concatenate([np.load(f) for f in sorted(d.glob("shard_*.npy"))]).astype(np.float32)
        assert len(emb) == len(ids), f"{domain}: encoding incomplete ({len(emb)}/{len(ids)})"
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        _indexes[key] = (index, ids)
    return _indexes[key]


def search(name, domain, queries, k=100):
    index, ids = get_index(name, domain)
    model = get_model(name)
    q = model.encode([QUERY_PREFIX.get(name, "") + x for x in queries], batch_size=32,
                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    scores, idx = index.search(q.astype(np.float32), k)
    return [[(ids[i], float(s)) for i, s in zip(irow, srow) if i >= 0] for irow, srow in zip(idx, scores)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Encode corpora (resumable).")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    for dom in a.domains:
        encode_corpus(a.model, dom, a.batch_size, a.max_len)
