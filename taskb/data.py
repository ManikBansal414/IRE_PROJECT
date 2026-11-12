"""Task B data: generation with reference passages (benchmark file generation_tasks/reference.jsonl, 842 dev tasks)."""
import random

from taska.data import BENCH, COLLECTION_TO_DOMAIN, read_jsonl

GEN = BENCH / "mtrag-human" / "generation_tasks"
REFERENCE = GEN / "reference.jsonl"
# Fields of the benchmark file that are not given at evaluation time.
GOLD_FIELDS = {"targets", "Answerability", "answerability", "Question Type", "Multi-Turn", "enrichments", "predictions",
               "Standalone Type", "Ambiguity Type", "Validity"}
CONTEXT_FIELDS = ("document_id", "text", "title", "score")  # drops the human relevance feedback and query metadata


def strip(task):
    t = {k: v for k, v in task.items() if k not in GOLD_FIELDS}
    t["input"] = [{"speaker": x["speaker"], "text": x["text"]} for x in task["input"]]
    t["contexts"] = [{k: c[k] for k in CONTEXT_FIELDS if k in c} for c in task.get("contexts", [])]
    return t


def load_tasks(path=REFERENCE):
    """Task B input rows with gold fields removed, so they cannot leak into generation."""
    return [strip(t) for t in read_jsonl(path)]


def answerability(task):
    return (task.get("Answerability") or task.get("answerability"))[0]


def load_gold(path=REFERENCE):
    """{task_id: {"target": reference answer, "answerability": label, "domain": domain}}"""
    return {t["task_id"]: {"target": t["targets"][0]["text"], "answerability": answerability(t),
                           "domain": COLLECTION_TO_DOMAIN[t["Collection"]]} for t in read_jsonl(path)}


def dev_sample(n, seed=0, path=REFERENCE, everything=False):
    """Fixed tuning sample stratified by domain and answerability (dev set only). By default only tasks that have
    reference passages (Task B: the others get the fixed IDK reply); everything=True also draws the UNANSWERABLE
    and CONVERSATIONAL tasks (Task C, where retrieval always returns passages)."""
    strata = {}
    for t in read_jsonl(path):
        if t["contexts"] or everything:
            strata.setdefault((t["Collection"], answerability(t)), []).append(t["task_id"])
    total = sum(map(len, strata.values()))
    rng = random.Random(seed)
    out = []
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        out += ids[:max(1, round(n * len(ids) / total))]
    return sorted(out)
