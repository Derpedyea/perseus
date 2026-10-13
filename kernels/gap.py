#!/usr/bin/env python3
"""Where does decode time go? Compares a lab run's measured decode speed with its two ceilings.

  bandwidth ceiling = memory bandwidth (membw step) / bytes of weights read per token
  compute ceiling   = sum over quant formats of (weights / 256) * ns per 256 weights (kernel
                      benches, steps 2-00c and 2-00g, one thread) / cores used

Whichever ceiling is lower is the bottleneck; the gap between it and the measured speed is
overhead (other ops, sync, compute and memory not overlapping). The compute ceiling assumes the
one-thread bench speed holds with every core busy, so it is optimistic.

usage: gap.py RESULTS_DIR MODEL.gguf BENCH_STEP [MEMBW_LABEL]
  e.g. gap.py results/macpro-2026... models/Underdog-Saluki-27B-1.0-IQ2-mix.gguf 5-07-saluki-all-sockets
GGUF_PY=path/to/llama.cpp/gguf-py (the model file is only read for tensor types and sizes), so
it runs under python3 -I.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.environ["GGUF_PY"])
from gguf import GGUFReader  # noqa: E402


def main():
    run, model, step = sys.argv[1:4]
    label = sys.argv[4] if len(sys.argv) > 4 else None

    # ns per 256 weights, one column, from the kernel benches: "bench iq2_xxs_q8_K nc=1 single 12.9 ..."
    # (their "generic" rows, llama.cpp's scalar code, are not what the build runs)
    # a "repacked" row (the load-time layout llama.cpp then runs, see repack.cpp) replaces the
    # format's "single" row, and its bigger blocks count in the bytes read per token
    ns, repacked = {}, set()
    for f in glob.glob(os.path.join(run, "steps", "*-kernel-bench", "stdout.txt")):
        for line in open(f):
            m = re.match(r"bench (\w+?)_q8_\w+ +nc=1 (single|repacked) +([0-9.]+)", line)
            if m and (m.group(2) == "repacked" or m.group(1).lower() not in repacked):
                ns[m.group(1).lower()] = float(m.group(3))
                if m.group(2) == "repacked":
                    repacked.add(m.group(1).lower())
    if not ns:
        sys.exit("no kernel bench (steps 2-00c, 2-00g) in this run")

    # bandwidth: the named membw label, else the best all-core result
    gbps = {}
    for f in glob.glob(os.path.join(run, "steps", "*membw*", "stdout.txt")):
        for line in open(f):
            if '"mode":"read"' in line:
                r = json.loads(line)
                gbps[r["label"]] = r["gbps_best"]
    if label:
        bw_label, bw = label, gbps[label]
    else:
        bw_label, bw = max(((k, v) for k, v in gbps.items() if "all" in k), key=lambda kv: kv[1])

    # measured decode speed and thread count from the llama-bench step
    tg = threads = None
    for line in open(os.path.join(run, "steps", step, "stdout.txt")):
        if '"avg_ts"' in line:
            r = json.loads(line)
            if r.get("n_gen", 0) > 0:
                tg, threads = r["avg_ts"], r["n_threads"]
    if tg is None:
        sys.exit(f"no decode (tg) result in {step}")

    # weights and bytes per quant format; formats without a bench row (f32 norms etc.) are tiny
    repack_growth = {"iq3_s": 144 / 110}  # block_iq3s_p vs block_iq3_s
    elems, nbytes, missing = {}, 0, {}
    for t in GGUFReader(model).tensors:
        name = t.tensor_type.name.lower()
        grow = repack_growth.get(name, 1) if name in repacked and t.name != "token_embd.weight" else 1
        nbytes += t.n_bytes * grow
        if t.name == "token_embd.weight":
            continue  # a row lookup per token, not a matmul
        if name in ns:
            elems[name] = elems.get(name, 0) + t.n_elements
        else:
            missing[name] = missing.get(name, 0) + t.n_bytes

    compute_s = sum(e / 256 * ns[k] * 1e-9 for k, e in elems.items()) / threads
    bw_s = nbytes / (bw * 1e9)
    meas_s = 1 / tg
    print(f"model {os.path.basename(model)}: {nbytes/1e9:.2f} GB, {threads} threads")
    print(f"measured            {tg:6.2f} tok/s  ({meas_s*1e3:6.0f} ms/token)")
    print(f"bandwidth ceiling   {1/bw_s:6.2f} tok/s  ({bw_s*1e3:6.0f} ms/token at {bw:.1f} GB/s, {bw_label})")
    print(f"compute ceiling     {1/compute_s:6.2f} tok/s  ({compute_s*1e3:6.0f} ms/token, 1-thread bench x {threads})")
    floor = max(bw_s, compute_s)
    print(f"bound by {'compute (kernels)' if compute_s > bw_s else 'memory bandwidth'}; "
          f"overhead above that floor {100*(meas_s-floor)/meas_s:.0f}% of the measured time")
    print("kernel time by format (share of the compute estimate):")
    for k, e in sorted(elems.items(), key=lambda kv: -kv[1] / 256 * ns[kv[0]]):
        s = e / 256 * ns[k] * 1e-9 / threads
        print(f"  {k:8} {100*s/compute_s:5.1f}%  {ns[k]:6.1f} ns/256 weights{'  (repacked)' if k in repacked else ''}")
    if missing:
        print("no bench for: " + ", ".join(f"{k} ({v/1e6:.0f} MB)" for k, v in missing.items()))


if __name__ == "__main__":
    main()
