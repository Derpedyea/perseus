// Checks the kernels kernels/apply_ssse3_fn.py adds (q6_K, q5_K, iq4_nl, q4_1, q2_0) against
// llama.cpp's generic (scalar) reference on random blocks, the same way test_ssse3_kernels.c
// checks the others. iq4_nl, q4_1 and q2_0 keep the generic float order and must match bit for
// bit on SSSE3 builds; q6_K and q5_K must agree within 1e-5 of sum|x*y|.
//
// Guards against a vacuous pass: ggml_cpu_init() runs first (without it every fp16 scale reads
// as 0), most results must be non-zero, and a control run with one activation nudged must
// change the result.
//
// 'bench' instead of a trial count times them (one thread, weights in L2): ns per 256 weights,
// next to q4_0 and q8_0, the rest of Qwen3.8-27B Q4_0, so kernels/gap.py can cost all of it.
//
// Build: kernels/build-test.sh. Exit code 0 = all pass.
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

static uint64_t rng = 0xC2B2AE3D27D4EB4Full;
static uint64_t next_u64(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static float rand_uniform(void) { return (float)((next_u64() >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; }

typedef struct {
    const char * name;
    enum ggml_type type;
    vec_dot_fn fast, generic;
    int exact;          // SSSE3 path keeps the generic float order
    const int * sizes;  // row lengths to test (multiples of the block size)
} kernel_t;

// 640 and 2560 are Flash-Next's expert and hidden widths; 17408 is a long row
static const int SIZES_K[]  = { 256, 512, 2560, 6144, 17408, 0 };
static const int SIZES_32[] = { 32, 64, 96, 640, 2560, 17408, 0 };
static const int SIZES_64[] = { 64, 128, 640, 2560, 17408, 0 };

#define K(t, T, ex, sz) { #t, GGML_TYPE_##T, ggml_vec_dot_##t, ggml_vec_dot_##t##_generic, ex, sz }
static const kernel_t KERNELS[] = {
    K(q6_K_q8_K,   Q6_K,   0, SIZES_K),
    K(q5_K_q8_K,   Q5_K,   0, SIZES_K),
    K(iq4_nl_q8_0, IQ4_NL, 1, SIZES_32),
    K(q4_1_q8_1,   Q4_1,   1, SIZES_32),
    K(q2_0_q8_0,   Q2_0,   1, SIZES_64),
};

// Random weight blocks with sane fp16 scales (random bits there could be NaN/Inf).
static void fill_weights(enum ggml_type type, void * data, int nblocks) {
    const size_t bsize = ggml_type_size(type);
    uint8_t * p = data;
    for (size_t i = 0; i < bsize * nblocks; ++i) p[i] = (uint8_t)next_u64();
    for (int b = 0; b < nblocks; ++b) {
        void * blk = p + b * bsize;
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.02f * rand_uniform());
        const ggml_fp16_t dmin = ggml_fp32_to_fp16(0.02f * rand_uniform());
        switch (type) {
            case GGML_TYPE_Q4_0:   ((block_q4_0 *)blk)->d = d; break;
            case GGML_TYPE_Q4_1:   ((block_q4_1 *)blk)->d = d; ((block_q4_1 *)blk)->m = dmin; break;
            case GGML_TYPE_Q8_0:   ((block_q8_0 *)blk)->d = d; break;
            case GGML_TYPE_Q6_K:   ((block_q6_K *)blk)->d = d; break;
            case GGML_TYPE_Q5_K:   ((block_q5_K *)blk)->d = d; ((block_q5_K *)blk)->dmin = dmin; break;
            case GGML_TYPE_IQ4_NL: ((block_iq4_nl *)blk)->d = d; break;
            case GGML_TYPE_Q2_0:   ((block_q2_0 *)blk)->d = d; break;
            default: abort();
        }
    }
}

// Random activations in the full int8 range the quantizers produce ([-127, 127]),
// with q8_K block sums kept consistent (q6_K and q5_K use them).
static void fill_activations(enum ggml_type vec_type, void * data, int n) {
    if (vec_type == GGML_TYPE_Q8_0) {
        block_q8_0 * y = data;
        for (int b = 0; b < n / QK8_0; ++b) {
            y[b].d = ggml_fp32_to_fp16(0.05f * rand_uniform());
            for (int j = 0; j < QK8_0; ++j) y[b].qs[j] = (int8_t)((int)(next_u64() % 255) - 127);
        }
        return;
    }
    if (vec_type == GGML_TYPE_Q8_1) {  // q4_1's activations: s = d * sum(qs)
        block_q8_1 * y = data;
        for (int b = 0; b < n / QK8_1; ++b) {
            const float d = 0.05f * rand_uniform();
            int sum = 0;
            for (int j = 0; j < QK8_1; ++j) sum += y[b].qs[j] = (int8_t)((int)(next_u64() % 255) - 127);
            y[b].d = ggml_fp32_to_fp16(d);
            y[b].s = ggml_fp32_to_fp16(d * sum);
        }
        return;
    }
    block_q8_K * y = data;
    for (int b = 0; b < n / QK_K; ++b) {
        y[b].d = 0.05f * rand_uniform();
        for (int j = 0; j < QK_K; ++j) y[b].qs[j] = (int8_t)((int)(next_u64() % 255) - 127);
        for (int g = 0; g < QK_K/16; ++g) {
            int s = 0;
            for (int j = 0; j < 16; ++j) s += y[b].qs[16*g + j];
            y[b].bsums[g] = (int16_t)s;
        }
    }
}

// sum |x_i * y_i| in double from dequantized values: the scale the reorder tolerance uses
static double abs_dot(enum ggml_type type, enum ggml_type vec_type, const void * x, const void * y, int n) {
    float * xf = malloc(sizeof(float) * n);
    float * yf = malloc(sizeof(float) * n);
    ggml_get_type_traits(type)->to_float(x, xf, n);
    if (vec_type == GGML_TYPE_Q8_0) {
        const block_q8_0 * yb = y;
        for (int i = 0; i < n; ++i) yf[i] = ggml_fp16_to_fp32(yb[i / QK8_0].d) * yb[i / QK8_0].qs[i % QK8_0];
    } else if (vec_type == GGML_TYPE_Q8_1) {
        const block_q8_1 * yb = y;
        for (int i = 0; i < n; ++i) yf[i] = ggml_fp16_to_fp32(yb[i / QK8_1].d) * yb[i / QK8_1].qs[i % QK8_1];
    } else {
        const block_q8_K * yb = y;
        for (int i = 0; i < n; ++i) yf[i] = yb[i / QK_K].d * yb[i / QK_K].qs[i % QK_K];
    }
    double s = 0;
    for (int i = 0; i < n; ++i) s += fabs((double)xf[i] * yf[i]);
    free(xf); free(yf);
    return s;
}

static enum ggml_type vec_type_of(enum ggml_type type) {
    return ggml_get_type_traits_cpu(type)->vec_dot_type;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// ns per 256 weights: best of 5, 64 rows x 5120 weights (L2-sized), one activation row.
static void bench_one(const char * name, const char * which, enum ggml_type type, vec_dot_fn fn) {
    enum { N = 5120, R = 64, ITERS = 400 };
    const enum ggml_type vt = vec_type_of(type);
    const size_t rs = ggml_row_size(type, N);
    char * x = malloc(rs * R);
    void * y = malloc(ggml_row_size(vt, N));
    for (int r = 0; r < R; ++r) fill_weights(type, x + r*rs, N / (int)ggml_blck_size(type));
    fill_activations(vt, y, N);
    double best = 1e9;
    float out = 0;
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now_s();
        for (int it = 0; it < ITERS; ++it)
            for (int r = 0; r < R; ++r) fn(N, &out, 0, x + r*rs, 0, y, 0, 1);
        const double t = now_s() - t0;
        if (t < best) best = t;
    }
    printf("bench %-14s nc=1 %-7s %7.1f\n", name, which, best * 1e9 / ((double)ITERS * R * (N / 256)));
    free(x); free(y);
}

static void bench(void) {
    printf("bench: ns per 256 weights, 1 thread, weights in L2\n");
    for (size_t k = 0; k < sizeof KERNELS / sizeof KERNELS[0]; ++k) {
        bench_one(KERNELS[k].name, "single", KERNELS[k].type, KERNELS[k].fast);
        bench_one(KERNELS[k].name, "generic", KERNELS[k].type, KERNELS[k].generic);
    }
    bench_one("q4_0_q8_0", "single", GGML_TYPE_Q4_0, ggml_vec_dot_q4_0_q8_0);
    bench_one("q8_0_q8_0", "single", GGML_TYPE_Q8_0, ggml_vec_dot_q8_0_q8_0);
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        bench();
        return 0;
    }
    const int trials = argc > 1 ? atoi(argv[1]) : 200;
#if defined(__AVX2__)
    const char * path = "AVX2";
#elif defined(__AVX__)
    const char * path = "AVX";
#elif defined(__SSSE3__)
    const char * path = "SSSE3";
#else
    const char * path = "scalar";
#endif
    const int ssse3 = !strcmp(path, "SSSE3");
    printf("kernel path compiled in: %s, %d trials per size\n", path, trials);

    int failures = 0;
    for (size_t k = 0; k < sizeof KERNELS / sizeof KERNELS[0]; ++k) {
        const kernel_t * kt = &KERNELS[k];
        const enum ggml_type vt = vec_type_of(kt->type);
        int exact = 0, total = 0, bad = 0, nonzero = 0, control_diff = 0;
        double worst = 0;
        for (const int * pn = kt->sizes; *pn; ++pn) {
            const int n = *pn;
            void * x = malloc(ggml_row_size(kt->type, n));
            void * y = malloc(ggml_row_size(vt, n));
            for (int t = 0; t < trials; ++t) {
                fill_weights(kt->type, x, n / (int)ggml_blck_size(kt->type));
                fill_activations(vt, y, n);
                float a = 0, b = 0;
                kt->fast(n, &a, 0, x, 0, y, 0, 1);
                kt->generic(n, &b, 0, x, 0, y, 0, 1);
                ++total;
                if (a != 0.0f) ++nonzero;
                {   // control: nudge one activation; the fast result must move
                    int8_t * q = vt == GGML_TYPE_Q8_0 ? ((block_q8_0 *)y)->qs
                               : vt == GGML_TYPE_Q8_1 ? ((block_q8_1 *)y)->qs : ((block_q8_K *)y)->qs;
                    const int8_t saved = q[0];
                    q[0] = (int8_t)(saved > 0 ? saved - 1 : saved + 1);
                    if (vt == GGML_TYPE_Q8_K) ((block_q8_K *)y)->bsums[0] += q[0] - saved;
                    float c = 0;
                    kt->fast(n, &c, 0, x, 0, y, 0, 1);
                    if (c != a) ++control_diff;
                    if (vt == GGML_TYPE_Q8_K) ((block_q8_K *)y)->bsums[0] -= q[0] - saved;
                    q[0] = saved;
                }
                if (memcmp(&a, &b, sizeof a) == 0) { ++exact; continue; }
                const double scale = abs_dot(kt->type, vt, x, y, n);
                const double err = fabs((double)a - b) / (scale > 0 ? scale : 1);
                if (err > worst) worst = err;
                const int ok = !(ssse3 && kt->exact) && isfinite(a) && err <= 1e-5;
                if (!ok) {
                    if (bad < 3) printf("  MISMATCH %s n=%d trial=%d fast=%.9g generic=%.9g rel=%.3g\n", kt->name, n, t, a, b, err);
                    ++bad;
                }
            }
            free(x); free(y);
        }
        // a vacuous comparison (all zeros, or insensitive to its inputs) is a failure too.
        // A weight code of 1 (q2_0) or a zero grid value contributes 0, so "moved" is not 100%.
        const int vacuous = nonzero < total * 9 / 10 || control_diff < total / 2;
        printf("%-14s %s  %5d/%-5d bit-exact  worst rel err %.2g  nonzero %d  control moved %d%s\n",
               kt->name, bad || vacuous ? "FAIL" : "ok  ", exact, total, worst, nonzero, control_diff,
               ssse3 && kt->exact ? "  (must be bit-exact)" : "");
        failures += bad + vacuous;
    }
    printf(failures ? "FAILED: %d mismatches\n" : "all kernels agree with the generic reference\n", failures);
    return failures != 0;
}
