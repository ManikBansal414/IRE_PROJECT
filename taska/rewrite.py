"""Conversational query rewriting with a T5 model fine-tuned on CANARD
(castorini/t5-base-canard): turns the last question into a standalone question.

Input format expected by the model: previous utterances and the current question
joined by " ||| ", current question last.

  python -m taska.rewrite --out data/rewrites_t5canard.jsonl
"""
import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import ROOT, load_tasks, write_jsonl

MODEL = "castorini/t5-base-canard"


def history_string(task, max_turns=6, max_agent_words=40):
    """Recent turns (user + shortened agent answers), current question last."""
    parts = []
    for t in task["input"][-max_turns - 1:]:
        text = " ".join(t["text"].split())
        if t["speaker"] == "agent":
            text = " ".join(text.split()[:max_agent_words])
        parts.append(text)
    return " ||| ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "taskA_dev_input.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "rewrites_t5canard.jsonl"))
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device).eval()

    tasks = load_tasks(args.input)
    rows = []
    for i in range(0, len(tasks), args.batch_size):
        batch = tasks[i:i + args.batch_size]
        enc = tok([history_string(t) for t in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=64, num_beams=4)
        for t, o in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
            last = [x["text"] for x in t["input"] if x["speaker"] == "user"][-1]
            rows.append({"task_id": t["task_id"], "rewrite": o.strip() or last})
        print(f"{min(i + args.batch_size, len(tasks))}/{len(tasks)}", flush=True)
    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()
