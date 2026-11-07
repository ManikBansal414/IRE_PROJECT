"""Chat completions through OpenAI-compatible APIs (Azure OpenAI, Groq), with an on-disk cache, a token and
cost ledger, a hard spending cap and rate-limit handling.

Keys are never logged. Azure: $AZURE_OPENAI_API_KEY and $AZURE_OPENAI_ENDPOINT, else the git-ignored file
azure_api.txt. Groq: $GROQ_API_KEY, else the git-ignored file api.txt.
Every response is cached by its exact request under work/llm_cache/, so re-running a configuration costs nothing
and an interrupted run resumes. Every paid call is appended to work/llm_usage.jsonl; a new call is refused once
the ledger total reaches the budget ($LLM_BUDGET_USD, default 30).

  python -m taskb.llm --usage                    # tokens and cost so far, per model and per run
  python -m taskb.llm --models --provider groq   # model ids available to a key
"""
import argparse
import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime

import requests

from taska.data import ROOT, WORK

CACHE = WORK / "llm_cache"
LEDGER = WORK / "llm_usage.jsonl"
BUDGET = float(os.environ.get("LLM_BUDGET_USD", 30))
MAX_WAIT = 600  # a longer Retry-After means a daily quota is used up
# USD per million (input, output) tokens; Azure global standard list prices. Groq's free tier costs nothing.
PRICES = {"gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
          "Llama-3.3-70B-Instruct": (0.71, 0.71)}
# Seconds between requests for deployments with low per-minute limits (Llama on Azure: 20K tokens/min).
PACE = {"Llama-3.3-70B-Instruct": 4.0}

_lock = threading.Lock()


class ContentFiltered(Exception):
    """Azure's content filter refused the request (HTTP 400, code content_filter)."""
_spent = None  # ledger total, loaded on first use


def azure_settings():
    key, endpoint = os.environ.get("AZURE_OPENAI_API_KEY"), os.environ.get("AZURE_OPENAI_ENDPOINT")
    path = ROOT / "azure_api.txt"
    if (not key or not endpoint) and path.exists():
        text = path.read_text()
        key = key or re.findall(r"\b[A-Za-z0-9]{40,}\b", text)[0]
        endpoint = endpoint or re.search(r"https://[\w.-]+\.azure\.com", text).group(0) + "/openai/v1"
    if not key or not endpoint:
        raise SystemExit("set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (or fill azure_api.txt)")
    return endpoint, {"api-key": key}


def groq_settings():
    key = os.environ.get("GROQ_API_KEY")
    if not key and (ROOT / "api.txt").exists():
        key = (ROOT / "api.txt").read_text().strip()
    if not key:
        raise SystemExit("set GROQ_API_KEY (or put the key in api.txt)")
    return "https://api.groq.com/openai/v1", {"Authorization": f"Bearer {key}"}


PROVIDERS = {"azure": (azure_settings, 0.0), "groq": (groq_settings, 2.0)}  # (settings, seconds between requests)


def cost(model, usage):
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (usage.get("prompt_tokens", 0) * p_in + usage.get("completion_tokens", 0) * p_out) / 1e6


def spent():
    global _spent
    if _spent is None:
        _spent = sum(json.loads(l)["cost"] for l in LEDGER.read_text().splitlines()) if LEDGER.exists() else 0.0
    return _spent


class Client:
    def __init__(self, model, provider="azure", temperature=0.0, max_tokens=512, tag=""):
        """tag: name of the run, recorded in the ledger."""
        self.model, self.provider, self.temperature, self.max_tokens, self.tag = (
            model, provider, temperature, max_tokens, tag)
        settings, self.min_interval = PROVIDERS[provider]
        self.min_interval = PACE.get(model, self.min_interval)
        self.url, headers = settings()
        self.session = requests.Session()
        self.session.headers.update(headers)
        self.calls = self.cached = self.tokens_in = self.tokens_out = 0
        self.cost = 0.0
        self._last = 0.0
        self._pace = threading.Lock()

    def chat(self, messages, json_mode=False):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        key = hashlib.sha256(json.dumps({"provider": self.provider, **body}, sort_keys=True).encode()).hexdigest()
        path = CACHE / key[:2] / f"{key}.json"
        if path.exists():
            with _lock:
                self.cached += 1
            return json.loads(path.read_text())["content"]
        if spent() >= BUDGET:
            raise SystemExit(f"budget of ${BUDGET:.2f} reached (ledger total ${spent():.2f}); "
                             "raise LLM_BUDGET_USD to continue")

        for attempt in range(10):
            if self.min_interval:
                with self._pace:
                    time.sleep(max(0.0, self.min_interval - (time.time() - self._last)))
                    self._last = time.time()
            try:
                r = self.session.post(f"{self.url}/chat/completions", json=body, timeout=180)
            except requests.RequestException:
                time.sleep(5 * 2 ** min(attempt, 4))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("retry-after") or 5 * 2 ** min(attempt, 4))
                if wait > MAX_WAIT:
                    raise SystemExit(f"{self.model}: quota exhausted (retry after {wait / 3600:.1f} h). "
                                     "Answers so far are cached; re-run the same command later to continue.")
                time.sleep(wait)
                continue
            if r.status_code == 400 and "response_format" in body and "response_format" in r.text:
                # Some deployments (e.g. Llama on Azure) reject JSON mode; the prompts ask for JSON anyway.
                body = {k: v for k, v in body.items() if k != "response_format"}
                continue
            if r.status_code == 400 and "content_filter" in r.text:
                raise ContentFiltered(self.model)
            if r.status_code >= 400:
                raise RuntimeError(f"{self.model}: HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            # Reasoning models may inline their reasoning; only the final answer is kept.
            content = re.sub(r"<think>.*?</think>", "", data["choices"][0]["message"]["content"] or "", flags=re.S)
            usage = data.get("usage") or {}
            c = cost(self.model, usage)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"request": body, "content": content, "usage": usage}))
            self._record(usage, c)
            return content
        raise RuntimeError(f"{self.model}: no response after 10 attempts")

    def _record(self, usage, c):
        global _spent
        row = {"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "provider": self.provider, "model": self.model,
               "tag": self.tag, "prompt_tokens": usage.get("prompt_tokens", 0),
               "completion_tokens": usage.get("completion_tokens", 0), "cost": c}
        with _lock:
            spent()
            _spent += c
            self.calls += 1
            self.tokens_in += row["prompt_tokens"]
            self.tokens_out += row["completion_tokens"]
            self.cost += c
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with open(LEDGER, "a") as f:
                f.write(json.dumps(row) + "\n")

    def status(self):
        return (f"api calls {self.calls}, cached {self.cached}, tokens {self.tokens_in:,} in / "
                f"{self.tokens_out:,} out, ${self.cost:.3f} (ledger total ${spent():.2f} of ${BUDGET:.0f})")


def usage_report():
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    lines = ["| model | run | calls | input tokens | output tokens | cost |", "|---|---|---|---|---|---|"]
    groups = defaultdict(lambda: [0, 0, 0, 0.0])
    for r in rows:
        g = groups[(r["model"], r["tag"])]
        g[0] += 1
        g[1] += r["prompt_tokens"]
        g[2] += r["completion_tokens"]
        g[3] += r["cost"]
    for (model, tag), (n, i, o, c) in sorted(groups.items()):
        lines.append(f"| {model} | {tag} | {n} | {i:,} | {o:,} | ${c:.3f} |")
    total = sum(r["cost"] for r in rows)
    lines.append(f"| **total** | | {len(rows)} | {sum(r['prompt_tokens'] for r in rows):,} | "
                 f"{sum(r['completion_tokens'] for r in rows):,} | **${total:.3f}** |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", action="store_true", help="tokens and cost so far, per model and run")
    ap.add_argument("--models", action="store_true", help="list the model ids available to the key")
    ap.add_argument("--provider", default="groq", choices=list(PROVIDERS))
    args = ap.parse_args()
    if args.usage:
        print(usage_report())
    elif args.models:
        url, headers = PROVIDERS[args.provider][0]()
        r = requests.get(f"{url}/models", headers=headers, timeout=60)
        r.raise_for_status()
        for m in sorted(r.json()["data"], key=lambda m: m["id"]):
            print(m["id"])


if __name__ == "__main__":
    main()
