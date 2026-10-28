"""Loading of MTRAG corpora, Task A inputs and qrels."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "external" / "mt-rag-benchmark"
CORPUS_DIR = BENCH / "corpora" / "passage_level"
WORK = ROOT / "work"  # indexes, embeddings, runs

COLLECTION_TO_DOMAIN = {
    "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
    "mt-rag-govt-elser-512-100-20240611": "govt",
    "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
    "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
}
DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]

# Fields present in the benchmark files that will NOT be given at evaluation time.
GOLD_FIELDS = {"contexts", "targets", "Question Type", "Multi-Turn", "No. References",
               "rewritten_query", "Answerability", "predictions", "enrichments"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_corpus(domain):
    """Returns (ids, texts) with the title prepended when it is not already the text's first line."""
    ids, texts = [], []
    with open(CORPUS_DIR / f"{domain}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = " ".join(d["text"].split())
            title = " ".join((d.get("title") or "").split())
            if title and not text.startswith(title):
                text = f"{title}. {text}"
            ids.append(d["_id"])
            texts.append(text)
    return ids, texts


def load_tasks(path):
    """Task A input rows (official format). Gold fields are dropped so they cannot leak."""
    rows = read_jsonl(path)
    for r in rows:
        for k in GOLD_FIELDS & r.keys():
            del r[k]
        r["domain"] = COLLECTION_TO_DOMAIN[r["Collection"]]
    return rows


def load_qrels(domain):
    """{query_id: {passage_id: rel}}, query ids use the '<::>' separator."""
    qrels = {}
    path = BENCH / "mtrag-human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for qid, pid, score in reader:
            qrels.setdefault(qid, {})[pid] = int(score)
    return qrels


def norm_qid(task_id):
    return task_id.replace("<::>", "::").replace("::", "<::>")
