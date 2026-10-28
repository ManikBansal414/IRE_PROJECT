"""Download a large file over a flaky link using parallel HTTP range requests.

Usage: python chunked_download.py URL OUT_PATH SIZE_BYTES [N_CHUNKS]
Each chunk is retried independently and resumed from where it stopped.
"""
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def fetch_range(url, part_path, start, end):
    expected = end - start + 1
    for _ in range(50):
        have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if have >= expected:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(part_path, "ab") as f:
                while chunk := resp.read(1 << 16):
                    f.write(chunk)
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"gave up on {part_path}")


def main():
    url, out, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    step = -(-size // n)
    ranges = [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(n)]
    with ThreadPoolExecutor(n) as pool:
        list(pool.map(lambda r: fetch_range(url, f"{out}.part{r[0]}", r[1], r[2]), ranges))
    with open(out, "wb") as f:
        for i, _, _ in ranges:
            with open(f"{out}.part{i}", "rb") as p:
                f.write(p.read())
            os.remove(f"{out}.part{i}")
    assert os.path.getsize(out) == size, "size mismatch"
    print("OK", out)


if __name__ == "__main__":
    main()
