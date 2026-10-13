# Kernels for pre-AVX x86

llama.cpp's x86 dot-product kernels for most quant formats exist only for AVX/AVX2; on
Penryn/Nehalem/Westmere (e.g. the 2012 Mac Pro) they fall back to generic scalar code. Two patches,
both applied by `kit/build/build-kit.sh binaries` to a clean checkout of the pinned llama.cpp:

- `apply_ssse3.py` (quants.c, quants.h, ggml-cpu.c)
  - SSSE3 vec_dot for q8_0, q2_K, q4_K, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s, iq1_m,
    iq4_xs: the integer math of the AVX paths (already 128-bit), each block summed exactly
    before one float multiply. iq1_s/iq1_m go further: weights 8*grid +- 1 become unsigned
    u = 8*(grid+1) + {0,2} (table generated from iq1s_grid) and the -9*sum(q8) correction comes
    from bsums, so no sign ops and no separate delta pass (iq1_s 18.7 -> 12.3 ns, iq1_m 28.0 ->
    25.3 ns per 256 weights; Saluki decode +4.3%).
  - Multi-column kernels for 2..16 tokens (MTP verification, short batches): each weight block
    is decoded once and dotted with every column. Bit-identical to the single-column kernels.
    `PERSEUS_MC=0` turns them off for A/B runs.
- `apply_tiled_sse.py` (ggml-cpu/tiled): turns on llama.cpp's tiled matmul (batches of 8+, i.e.
  prompt processing) for SSE4.1. Upstream gates it on AVX, but its AVX1 microkernel is 128-bit
  integer code whose newest instruction is SSE4.1. `GGML_CPU_TILED_MM=0` turns it off.

Tests (kit step 2-00 / 2-00b on the lab box; `build-kit.sh test` under emulated Penryn/Nehalem):
- `test_ssse3_kernels.c`: vec_dot vs llama.cpp's generic code on random blocks (9 formats bit
  for bit; q4_K, iq4_xs within 5e-8 relative), then every multi-column result vs the
  single-column kernel, bit for bit. A broken kernel fails it (mutation-checked), and it refuses
  vacuous passes (all-zero results, insensitive to inputs, multi-column path switched off).
- llama.cpp's `tests/test-tiled-mulmat.cpp`: tiled vs vec_dot. 88/88 with `--fuzz` on SSE. It
  also passes if the tiled path is off, so engagement is shown by timing (`GGML_CPU_TILED_MM=0`
  vs default: Qwen3.5-0.8B Q4_K_M pp128 194 -> 427 tok/s on the penryn build).

`gap.py RESULTS_DIR MODEL STEP` turns a lab run into the decode budget: measured tok/s vs the
bandwidth ceiling (membw / bytes per token) and the kernel compute ceiling (bench x weights per
format / cores), plus each format's share of kernel time. That picks the next target.

`test-ssse3-kernels bench` (kit step 2-00c) times every kernel on one thread with the weights in
L2: ns per 256 weights, single-column and multi-column at 1/2/4/8 tokens. On the lab box this says
which formats are compute-bound there and what each extra MTP token costs.

Saluki 27B, Ryzen 7700X, 8 threads, old-CPU (nehalem) build, tok/s:

| | before | after |
|---|---|---|
| generate | 0.58 (generic) | 3.83 (SSSE3 vec_dot; AVX2 build 3.97) |
| prompt, 32 tokens | 0.63 | 8.05 (tiled; 4.76 without) |
| prompt, 128 tokens | | 10.6 (tiled; 4.65 without) |
| 4-token batch (= one MTP check) | | 6.18 (multi-column; 4.54 without) |
| MTP draft 3, 82% accepted | | 4.24 (multi-column; 3.24 without, slower than no MTP) |

Where the multi-column time goes (Ryzen, 1 thread, ns per 256 weights): iq2_xxs single 12.9;
multi-column ~16 to decode a block into its buffer plus ~6.8 per token. Storing and reloading
the decoded block costs about as much as the whole single-column kernel, so 4 tokens cost ~3.3x
one. q2_K's single-column kernel is so cheap (8.0) that multi-column only breaks even at 4.
Next, once the lab numbers say it matters: keep the decoded chunk in registers for up to 4
columns and use the tiled kernels' activation-bias trick (biased activations shared across rows,
signed weights, per-block correction): ~4 instead of ~6 instructions per 16 weights per token.
