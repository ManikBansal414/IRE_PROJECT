"""Task B: generate a grounded answer for every task and write an official-format prediction file.

  python -m taskb.generate --prompt base --sample 150 --name base_s150    # fixed dev tuning sample
  python -m taskb.generate --model gpt-4.1 --prompt base --sample 150 --name gpt41_base_s150
  python -m taskb.generate --prompt base --name base                      # full dev set
  python -m taskb.generate --input <test.jsonl> --prompt base --name test_base
  python -m taskb.generate --input results/best_system/best.predictions.jsonl --top_k 5 --name c_base   # Task C

Task C uses the same generator on the top --top_k passages of our Task A system (its prediction file has the
same format and task ids); the reference answers are identical to Task B's, so taskb.evaluate scores both.

Tasks without passages get the fixed "I don't know" reply (no API call). Output:
results/taskb/<name>.predictions.jsonl and <name>.config.json; existing files are never overwritten.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from taska.data import ROOT, write_jsonl
from .data import REFERENCE, dev_sample, load_tasks
from .llm import Client
from .prompts import IDK, VARIANTS, build_messages

OUT = ROOT / "results" / "taskb"
MODEL, PROVIDER = "gpt-4.1-mini", "azure"


def generate(tasks, client, variant, workers=8):
    def answer(t):
        return client.chat(build_messages(t, variant)).strip() if t["contexts"] else IDK

    texts = []
    with ThreadPoolExecutor(workers) as pool:
        for i, text in enumerate(pool.map(answer, tasks)):
            texts.append(text)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(tasks)}  {client.status()}", flush=True)
    return [{**t, "predictions": [{"text": x}]} for t, x in zip(tasks, texts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REFERENCE))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--provider", default=PROVIDER, choices=["azure", "groq"])
    ap.add_argument("--workers", type=int, default=8, help="parallel requests")
    ap.add_argument("--prompt", default="base", choices=list(VARIANTS))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--top_k", type=int, default=None, help="keep only the first k passages (Task C)")
    ap.add_argument("--sample", type=int, default=None, help="only the fixed stratified dev sample of this size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_all", action="store_true",
                    help="sample over all tasks, unanswerable ones included (Task C)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = OUT / f"{args.name}.predictions.jsonl"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out}; use another --name or --overwrite")

    tasks = load_tasks(args.input)
    if args.top_k:
        tasks = [{**t, "contexts": t["contexts"][:args.top_k]} for t in tasks]
    if args.sample:
        keep = set(dev_sample(args.sample, args.seed, everything=args.sample_all))
        tasks = [t for t in tasks if t["task_id"] in keep]
    client = Client(args.model, args.provider, args.temperature, args.max_tokens, tag=f"generate:{args.name}")
    rows = generate(tasks, client, args.prompt, args.workers if args.provider == "azure" else 1)

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    (OUT / f"{args.name}.config.json").write_text(json.dumps(vars(args), indent=1))
    print(f"{out}: {len(rows)} tasks; {client.status()}")


if __name__ == "__main__":
    main()
