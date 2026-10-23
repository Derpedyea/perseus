#!/usr/bin/env python3
"""Client for the kit's llama-server steps (run.sh step_serve). Standard library only.

  serve_bench.py --port N [--copy FILE] [--free TEXT] [--n TOKENS] [--parallel 1,2,4]

Waits for the server's /health, then sends greedy /completion requests and prints one JSON line
per case, with the server's own timings (prompt and generated tokens per second, and for
speculative decoding the drafted and accepted token counts):
  copy:     FILE, then the instruction to write it out again: the answer repeats the prompt,
            as a code edit does (n-gram speculation's best case)
  free:     TEXT as is (an answer that repeats nothing)
  parallel: K copies of the free request at once, for each K: total generated tokens over the
            wall time of the batch
Exits non-zero if the server never came up or a request failed.
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

# A server that never opens its port has failed; one that loads a 75 GB model takes minutes.
REFUSED_LIMIT_S = 120
LOADING_LIMIT_S = 900


def wait_ready(base):
    t0 = time.time()
    seen_up = False
    while True:
        try:
            with urllib.request.urlopen(base + "/health", timeout=10) as r:
                if r.status == 200:
                    return time.time() - t0
        except urllib.error.HTTPError as e:  # 503 while the model loads
            seen_up = True
            if e.code != 503:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        waited = time.time() - t0
        if not seen_up and waited > REFUSED_LIMIT_S:
            sys.exit(f"serve_bench: nothing listening at {base} after {REFUSED_LIMIT_S} s (see server.txt)")
        if waited > LOADING_LIMIT_S:
            sys.exit(f"serve_bench: server still loading after {LOADING_LIMIT_S} s")
        time.sleep(2)


def complete(base, prompt, n):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 42,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(base + "/completion", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    out["wall_s"] = time.time() - t0
    return out


def row(case, out):
    t = out.get("timings", {})
    keep = ["prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted"]
    r = {"case": case, **{k: t[k] for k in keep if k in t}, "wall_s": round(out["wall_s"], 2)}
    r["text_head"] = out.get("content", "")[:120]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--copy")
    ap.add_argument("--free")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--parallel")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"

    print(json.dumps({"case": "ready", "load_s": round(wait_ready(base), 1)}), flush=True)
    if a.copy:
        src = open(a.copy).read()[:4000]  # ~1.2k tokens: the prompt read stays short at ~40 tok/s
        prompt = f"Here is a C source file:\n\n{src}\n\nThe same file again, unchanged:\n\n"
        print(json.dumps(row("copy", complete(base, prompt, a.n))), flush=True)
    if a.free and not a.parallel:
        print(json.dumps(row("free", complete(base, a.free, a.n))), flush=True)
    for k in [int(x) for x in a.parallel.split(",")] if a.parallel else []:
        results, errors = [None] * k, []

        def one(i):
            try:
                results[i] = complete(base, a.free, a.n)
            except Exception as e:  # reported below; the batch's number would be wrong
                errors.append(repr(e))

        t0 = time.time()
        threads = [threading.Thread(target=one, args=(i,)) for i in range(k)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall = time.time() - t0
        if errors:
            sys.exit(f"serve_bench: {len(errors)} of {k} parallel requests failed: {errors[0]}")
        total = sum(r["timings"]["predicted_n"] for r in results)
        print(json.dumps({"case": "parallel", "k": k, "total_predicted": total, "wall_s": round(wall, 2),
                          "total_tok_s": round(total / wall, 2),
                          "per_request_tok_s": [round(r["timings"]["predicted_per_second"], 2) for r in results]}),
              flush=True)


if __name__ == "__main__":
    main()
