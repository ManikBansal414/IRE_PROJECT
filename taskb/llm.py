"""Chat completions through Groq's OpenAI-compatible API, with an on-disk cache and rate-limit handling.

The key is read from $GROQ_API_KEY, else from the git-ignored file api.txt in the project root; it is never logged.
Every response is cached by its exact request under work/llm_cache/, so re-running a configuration costs no
quota and an interrupted run resumes where it stopped.

  python -m taskb.llm --models        # model ids available to the key
"""
import argparse
import hashlib
import json
import os
import re
import time

import requests

from taska.data import ROOT, WORK

URL = "https://api.groq.com/openai/v1"
CACHE = WORK / "llm_cache"
MAX_WAIT = 600  # a longer Retry-After means the daily quota is used up


def api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key and (ROOT / "api.txt").exists():
        key = (ROOT / "api.txt").read_text().strip()
    if not key:
        raise SystemExit("set GROQ_API_KEY (or put the key in api.txt)")
    return key


def session():
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {api_key()}"
    return s


class Client:
    def __init__(self, model, temperature=0.0, max_tokens=512, min_interval=2.0):
        """min_interval: seconds between requests (2 s = the free tier's 30 requests/minute)."""
        self.model, self.temperature, self.max_tokens, self.min_interval = model, temperature, max_tokens, min_interval
        self.session = session()
        self.calls = self.cached = 0
        self._last = 0.0

    def chat(self, messages, json_mode=False):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path = CACHE / key[:2] / f"{key}.json"
        if path.exists():
            self.cached += 1
            return json.loads(path.read_text())["content"]

        for attempt in range(10):
            time.sleep(max(0.0, self.min_interval - (time.time() - self._last)))
            self._last = time.time()
            r = self.session.post(f"{URL}/chat/completions", json=body, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("retry-after") or 5 * 2 ** attempt)
                if wait > MAX_WAIT:
                    raise SystemExit(f"{self.model}: quota exhausted (retry after {wait / 3600:.1f} h). "
                                     "Answers so far are cached; re-run the same command later to continue.")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            # Reasoning models may inline their reasoning; only the final answer is kept.
            content = re.sub(r"<think>.*?</think>", "", data["choices"][0]["message"]["content"] or "", flags=re.S)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"request": body, "content": content, "usage": data.get("usage")}))
            self.calls += 1
            return content
        raise RuntimeError(f"{self.model}: no response after 10 attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true", help="list the model ids available to the key")
    if ap.parse_args().models:
        r = session().get(f"{URL}/models", timeout=60)
        r.raise_for_status()
        for m in sorted(r.json()["data"], key=lambda m: m["id"]):
            print(m["id"], m.get("context_window", ""))


if __name__ == "__main__":
    main()
