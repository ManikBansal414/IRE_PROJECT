"""Task B: generate a grounded answer for every task and write an official-format prediction file.

  python -m taskb.generate --prompt base --sample 150 --name base_s150    # fixed dev tuning sample
  python -m taskb.generate --prompt base --name base                      # full dev set
  python -m taskb.generate --input <test.jsonl> --prompt base --name test_base

Tasks without passages get the fixed "I don't know" reply (no API call). Output:
results/taskb/<name>.predictions.jsonl and <name>.config.json; existing files are never overwritten.
"""
import argparse
import json

from taska.data import ROOT, write_jsonl
from .data import REFERENCE, dev_sample, load_tasks
from .llm import Client
from .prompts import IDK, VARIANTS, build_messages

OUT = ROOT / "results" / "taskb"
MODEL = "llama-3.3-70b-versatile"


def generate(tasks, client, variant):
    rows = []
    for i, t in enumerate(tasks):
        text = client.chat(build_messages(t, variant)).strip() if t["contexts"] else IDK
        rows.append({**t, "predictions": [{"text": text}]})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(tasks)}  api calls {client.calls}, cached {client.cached}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REFERENCE))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt", default="base", choices=list(VARIANTS))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--sample", type=int, default=None, help="only the fixed stratified dev sample of this size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = OUT / f"{args.name}.predictions.jsonl"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out}; use another --name or --overwrite")

    tasks = load_tasks(args.input)
    if args.sample:
        keep = set(dev_sample(args.sample, args.seed))
        tasks = [t for t in tasks if t["task_id"] in keep]
    client = Client(args.model, args.temperature, args.max_tokens)
    rows = generate(tasks, client, args.prompt)

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    (OUT / f"{args.name}.config.json").write_text(json.dumps(vars(args), indent=1))
    print(f"{out}: {len(rows)} tasks ({client.calls} api calls, {client.cached} cached)")


if __name__ == "__main__":
    main()
