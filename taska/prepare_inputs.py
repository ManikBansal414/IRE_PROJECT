"""Build the Task A dev input file (official evaluation format, no gold fields)
from the benchmark's generation_tasks/RAG.jsonl. Gold rewrites are saved separately,
for use only as an upper-bound reference."""
from .data import BENCH, ROOT, read_jsonl, write_jsonl

KEEP = ("conversation_id", "task_id", "turn", "Collection", "input")


def main():
    rows = read_jsonl(BENCH / "mtrag-human" / "generation_tasks" / "RAG.jsonl")
    write_jsonl(ROOT / "data" / "taskA_dev_input.jsonl", [{k: r[k] for k in KEEP} for r in rows])
    write_jsonl(ROOT / "data" / "gold_rewrites_UPPER_BOUND_ONLY.jsonl",
                [{"task_id": r["task_id"], "rewrite": r["rewritten_query"]}
                 for r in rows if r.get("rewritten_query")])
    print(f"wrote {len(rows)} tasks")


if __name__ == "__main__":
    main()
