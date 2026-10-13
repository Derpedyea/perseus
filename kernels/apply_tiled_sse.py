#!/usr/bin/env python3
"""Turns on llama.cpp's tiled batched matmul (ggml-cpu/tiled) for pre-AVX x86 with SSE4.1.

Upstream gates the tiled path on AVX. Its AVX1 microkernel only uses 128-bit integer
intrinsics, the newest being _mm_mullo_epi32 (SSE4.1), so Penryn/Nehalem/Westmere can run it
unchanged; the unpackers fall back to their scalar versions. The tiled path handles batches of
8+ tokens (prompt processing) for the K-quants and every IQ format in Underdog Saluki 27B,
decoding each weight block once per tile instead of once per token.

usage: apply_tiled_sse.py path/to/llama.cpp   (edits in place; refuses to run twice)
"""
import os
import sys

MARK = "/* perseus: tiled on SSE4.1 */"

# (file, old, new, expected count)
EDITS = [
    ("tiled.cpp",
     "#if !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)\n",
     "#if !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__) && !defined(__SSE4_1__) " + MARK + "\n",
     1),
    ("tiled-kernel.cpp",
     "#if defined(__AVX512VNNI__) || defined(__AVX2__) || defined(__AVX__)\n#include <immintrin.h>",
     "#if defined(__AVX512VNNI__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__) " + MARK + "\n#include <immintrin.h>",
     1),
    ("tiled-kernel.cpp",
     "#if defined(__AVX__) && !defined(__AVX2__)\n",
     "#if (defined(__AVX__) || defined(__SSE4_1__)) && !defined(__AVX2__) " + MARK + "\n",
     1),
    ("tiled-kernel.cpp",
     "#elif defined(__AVX__)\n    if (standard) {\n        tiled_run_microtile_avx<",
     "#elif defined(__AVX__) || defined(__SSE4_1__) " + MARK + "\n    if (standard) {\n        tiled_run_microtile_avx<",
     1),
]


def main():
    tiled = os.path.join(sys.argv[1], "ggml", "src", "ggml-cpu", "tiled")
    srcs = {f: open(os.path.join(tiled, f)).read() for f in {e[0] for e in EDITS}}
    if any(MARK in s for s in srcs.values()):
        sys.exit(f"{tiled} already has the perseus SSE4.1 tiled patch")
    for f, old, new, count in EDITS:
        assert srcs[f].count(old) == count, f"{f}: anchor changed upstream: {old!r}"
        srcs[f] = srcs[f].replace(old, new)
    for f, s in srcs.items():
        open(os.path.join(tiled, f), "w").write(s)
    print("enabled the tiled matmul for SSE4.1")


if __name__ == "__main__":
    main()
