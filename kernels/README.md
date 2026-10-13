# Kernels for pre-AVX x86

llama.cpp's x86 dot-product kernels for most quant formats exist only for AVX/AVX2; on
Penryn/Nehalem/Westmere (e.g. the 2012 Mac Pro) they fall back to generic scalar code. Three patches,
all applied by `kit/build/build-kit.sh binaries` to a clean checkout of the pinned llama.cpp:

- `apply_ssse3.py` (quants.c, quants.h, ggml-cpu.c, repack.cpp)
  - SSSE3 vec_dot for q8_0, q2_K, q4_K, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s, iq1_m,
    iq4_xs: the integer math of the AVX paths (already 128-bit), each block summed exactly
    before one float multiply. iq1_s/iq1_m go further: weights 8*grid +- 1 become unsigned
    u = 8*(grid+1) + {0,2} (table generated from iq1s_grid) and the -9*sum(q8) correction comes
    from bsums, so no sign ops and no separate delta pass.
  - Multi-column kernels for 2..16 tokens (MTP verification, short batches): each 32-weight
    sub-block is decoded once into registers and dotted with up to 4 columns. Bit-identical to
    the single-column kernels. `PERSEUS_MC=0` turns them off.
  - IQ3_S repacked at load (llama.cpp's CPU_REPACK mechanism): every weight is grid (1..15 odd)
    times a sign, so one signed 4-bit code; two pshufb decode 32 weights instead of 8 scalar grid
    lookups. The activations are biased to unsigned once per matmul, so no sign op either; the
    bias correction is precomputed per block. 144 instead of 110 bytes per 256 weights.
    `PERSEUS_REPACK=0` (or llama.cpp's `-nr`) keeps the file layout.
- `apply_ssse3_fn.py` (quants.c, arch-fallback.h; runs after `apply_ssse3.py`, uses its helpers):
  SSSE3 vec_dot for the formats Qwen3.8-Flash-Next GGUFs read every token: q6_K (output layer in
  nearly every build, also Qwen3.8-27B Q4_0's), q5_K (dense weights of the low-bit builds, ~1 GB
  of Qwen3.8-27B Q4_0), iq4_nl (expert down projection: row length 640 rules out block-256
  formats) and q2_0 (lookup-free 2-bit experts; llama.cpp had no x86 kernel at all).
- `apply_tiled_sse.py` (ggml-cpu/tiled): turns on llama.cpp's tiled matmul (batches of 8+, i.e.
  prompt processing) for SSE4.1. Upstream gates it on AVX, but its AVX1 microkernel is 128-bit
  integer code whose newest instruction is SSE4.1. `GGML_CPU_TILED_MM=0` turns it off.

Tests (kit steps 2-00, 2-00b, 2-00d, 2-00f on the lab box; `build-kit.sh test` under emulated Penryn/Nehalem):
- `test_ssse3_kernels.c`: vec_dot vs llama.cpp's generic code on random blocks (9 formats bit
  for bit; q4_K, iq4_xs within 5e-8 relative), every multi-column result vs the single-column
  kernel, and the repacked IQ3_S vs the original, bit for bit. Mutation-checked (an off-by-one in
  a correction term fails it), and it refuses vacuous passes (all-zero results, insensitive to
  inputs, multi-column path off).
- `test_fn_kernels.c`: the `apply_ssse3_fn.py` kernels vs the generic code the same way (iq4_nl
  and q2_0 bit for bit; q6_K, q5_K within 1e-5 of sum|x*y|, 3e-8 measured, as upstream's AVX
  paths). Each kernel mutation-checked. End to end, a random-weight Flash-Next-architecture model
  (Lynxpda/micro-qwen4exp) with q2_0/iq4_nl experts gives 200 identical greedy tokens with and
  without the patch.
- `test_repack_mulmat.c`: the repack through ggml's graph API (buffer, traits, threads, tiles,
  3-D activations) vs llama.cpp's own path: 396 cases bit-identical. Fails if the repack is off.
- llama.cpp's `tests/test-tiled-mulmat.cpp`: tiled vs vec_dot. 88/88 with `--fuzz` on SSE. It also
  passes if the tiled path is off, so engagement is shown by timing.
- End to end: Saluki's output is byte-identical with the multi-column kernels on and off, and
  with the repack on and off (tiled off, so both sides take the bit-exact path).

`test-ssse3-kernels bench` (kit step 2-00c): ns per 256 weights, one thread, weights in L2,
single-column, multi-column at 1/2/4/8 tokens and repacked IQ3_S. `test-fn-kernels bench` (2-00g):
the Flash-Next formats next to llama.cpp's generic code, plus q4_0 and q8_0. `gap.py RESULTS_DIR MODEL STEP`
turns a lab run into the decode budget: measured tok/s vs the bandwidth ceiling (membw / bytes
per token, repacked bytes included) and the kernel ceiling (bench x weights per format / cores).

Saluki 27B, Ryzen 7700X, 8 threads, old-CPU (nehalem) build, tok/s:

| | before | after |
|---|---|---|
| generate | 0.58 (generic) | 4.17 best of 9 (AVX2 build 5.0); 3.77 with the 10:17 kernels |
| prompt, 128 tokens | 4.65 | 10.6 (tiled; repack on or off) |
| MTP draft 3, 82% accepted | 3.24 | 5.15 (multi-column in registers; 3.83 plain decoding) |

ns per 256 weights, 1 thread: iq1_s 18.7 -> 12.5, iq1_m 28.0 -> 25.5, iq3_s 26.6 -> 8.1
(repacked); 4 tokens at once: iq2_xxs 51.6 (4x single) -> 25.4, iq3_s 107 -> 17.9.

Flash-Next formats, ns per 256 weights, 1 thread, generic -> SSSE3: q6_K 30.2 -> 10.9, q5_K 28.4 ->
13.5, iq4_nl 86.5 -> 15.3, q2_0 88.7 -> 16.4 (q4_0: 12.2). q2_0 is still slower than q4_0 at half
the bytes; a load-time repack would fix that if Westmere is compute-bound (2-00g decides).

Measured and not kept:
- Lookup-table kernels (T-MAC / bitnet.cpp style, `lut_bench.c`, kit step 2-00e): 2-bit weights,
  multiply kernel 5.08 ns vs exact tables 6.40 vs lossy int8 tables 6.68 (and 4.7% error). A
  16-entry pshufb table cannot hold exact sums of 4 activations, so it takes two lookups and a
  widen. Kept as a lab bench in case Westmere's ports disagree.
- Software prefetch (4 blocks ahead): no measurable gain at 8 threads, and even the off switch
  cost iq2_xxs 5% (12.7 -> 13.3 ns). Removed.
- IQ2_XXS / IQ1_M / IQ2_S repacks: these already cost ~12 instructions per 16 weights; a 4-bit
  layout would add 56-150% bytes for a smaller compute saving, a loss once memory is near its limit.
- Fusing non-matmul ops: one thread runs within 5% of the kernel-only prediction, so the other
  ops are not where time goes. At 8 threads the loss (~36% under background load) is contention:
  kernel demand is ~49 GB/s against 57 GB/s here.
