// Checks llama.cpp's x86 dot-product kernels against its generic (scalar) reference on
// random blocks, so every grid index, sign pattern and scale value gets exercised, not just
// the ones a quantizer happens to produce.
//
// On a build without AVX the kernels under test are the perseus SSSE3 paths. Types whose
// SSSE3 path keeps the generic float order must match bit for bit; the rest must agree within
// 1e-5 of sum|x*y| (computed in double from the dequantized values).
//
// Guards against a vacuous pass: ggml_cpu_init() must run first (without it the CPU
// backend's fp16->fp32 table is all zeros and every dot product is 0), most results must be
// non-zero, and a control run with one activation changed must disagree.
//
// On SSSE3 builds it also checks the perseus multi-column kernels (one weight row against
// 1..16 activation columns): every column must match the single-column kernel bit for bit.
//
// 'bench' instead of a trial count times the kernels instead (one thread, weights in L2):
// ns per 256-weight block, single-column and multi-column, to see where a CPU spends its time.
//
// Build: see kernels/build-test.sh. Exit code 0 = all pass.
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

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static float rand_uniform(void) { return (float)((next_u64() >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; }

typedef struct {
    const char * name;
    enum ggml_type type;
    vec_dot_fn fast, generic;
    int exact;  // SSSE3 path keeps the generic float order
    int mc;     // has a perseus multi-column kernel (SSSE3 builds)
} kernel_t;

#define K(t, T, ex, mc) { #t, GGML_TYPE_##T, ggml_vec_dot_##t, ggml_vec_dot_##t##_generic, ex, mc }
static const kernel_t KERNELS[] = {
    K(q8_0_q8_0,    Q8_0,    1, 0),
    K(q2_K_q8_K,    Q2_K,    1, 1),
    K(q4_K_q8_K,    Q4_K,    0, 0),
    K(iq2_xxs_q8_K, IQ2_XXS, 1, 1),
    K(iq2_xs_q8_K,  IQ2_XS,  1, 1),
    K(iq2_s_q8_K,   IQ2_S,   1, 1),
    K(iq3_xxs_q8_K, IQ3_XXS, 1, 1),
    K(iq3_s_q8_K,   IQ3_S,   1, 1),
    K(iq1_s_q8_K,   IQ1_S,   1, 1),
    K(iq1_m_q8_K,   IQ1_M,   1, 1),
    K(iq4_xs_q8_K,  IQ4_XS,  0, 1),
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
            case GGML_TYPE_Q8_0:    ((block_q8_0 *)blk)->d = d; break;
            case GGML_TYPE_Q2_K:    ((block_q2_K *)blk)->d = d; ((block_q2_K *)blk)->dmin = dmin; break;
            case GGML_TYPE_Q4_K:    ((block_q4_K *)blk)->d = d; ((block_q4_K *)blk)->dmin = dmin; break;
            case GGML_TYPE_IQ2_XXS: ((block_iq2_xxs *)blk)->d = d; break;
            case GGML_TYPE_IQ2_XS:  ((block_iq2_xs *)blk)->d = d; break;
            case GGML_TYPE_IQ2_S:   ((block_iq2_s *)blk)->d = d; break;
            case GGML_TYPE_IQ3_XXS: ((block_iq3_xxs *)blk)->d = d; break;
            case GGML_TYPE_IQ3_S:   ((block_iq3_s *)blk)->d = d; break;
            case GGML_TYPE_IQ1_S:   ((block_iq1_s *)blk)->d = d; break;
            case GGML_TYPE_IQ4_XS:  ((block_iq4_xs *)blk)->d = d; break;
            case GGML_TYPE_IQ1_M: {
                // the block scale is an fp16 spread over the top nibble of the four scale words
                uint16_t sc[4];
                memcpy(sc, ((block_iq1_m *)blk)->scales, sizeof sc);
                for (int k = 0; k < 4; ++k) sc[k] = (uint16_t)((sc[k] & 0x0fff) | (((d >> (4*k)) & 0xf) << 12));
                memcpy(((block_iq1_m *)blk)->scales, sc, sizeof sc);
                break;
            }
            default: abort();
        }
    }
}

// Random activations in the full int8 range the quantizers produce ([-127, 127]),
// with block sums kept consistent (several kernels use them).
static void fill_activations(enum ggml_type vec_type, void * data, int n) {
    if (vec_type == GGML_TYPE_Q8_0) {
        block_q8_0 * y = data;
        for (int b = 0; b < n / QK8_0; ++b) {
            y[b].d = ggml_fp32_to_fp16(0.05f * rand_uniform());
            for (int j = 0; j < QK8_0; ++j) y[b].qs[j] = (int8_t)((int)(next_u64() % 255) - 127);
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
    } else {
        const block_q8_K * yb = y;
        for (int i = 0; i < n; ++i) yf[i] = yb[i / QK_K].d * yb[i / QK_K].qs[i % QK_K];
    }
    double s = 0;
    for (int i = 0; i < n; ++i) s += fabs((double)xf[i] * yf[i]);
    free(xf); free(yf);
    return s;
}

#if !defined(__AVX__) && defined(__SSSE3__)
// Each column of the multi-column kernel against the single-column kernel, which the main
// test has already checked against the generic reference. Returns the number of failures.
static int test_mc(const kernel_t * kt, int trials) {
    static const int sizes[] = { 256, 1024, 5120 };
    if (!perseus_mc_supported(kt->type)) {
        printf("%-14s FAIL  no multi-column kernel (or PERSEUS_MC=0 set)\n", kt->name);
        return 1;
    }
    const size_t by = ggml_row_size(GGML_TYPE_Q8_K, 5120);
    void * x = malloc(ggml_row_size(kt->type, 5120));
    char * y = malloc(16 * by);
    int total = 0, bad = 0;
    for (size_t si = 0; si < sizeof sizes / sizeof *sizes; ++si) {
        const int n = sizes[si];
        for (int t = 0; t < trials; ++t) {
            const int nc = 1 + (int)(next_u64() % 16);
            fill_weights(kt->type, x, n / QK_K);
            for (int c = 0; c < nc; ++c) fill_activations(GGML_TYPE_Q8_K, y + c*by, n);
            float out[16];
            perseus_vec_dot_mc(kt->type, n, out, x, y, by, nc);
            for (int c = 0; c < nc; ++c) {
                float ref = 0;
                kt->fast(n, &ref, 0, x, 0, y + c*by, 0, 1);
                ++total;
                if (memcmp(&ref, &out[c], sizeof ref) != 0) {
                    if (bad < 3) printf("  MISMATCH %s multi-column n=%d nc=%d col=%d mc=%.9g single=%.9g\n", kt->name, n, nc, c, out[c], ref);
                    ++bad;
                }
            }
        }
    }
    free(x); free(y);
    printf("%-14s %s  %5d/%-5d columns bit-identical (multi-column vs single-column)\n", kt->name, bad ? "FAIL" : "ok  ", total - bad, total);
    return bad;
}

// biased activations (q8 ^ 0x80) of nc q8_K columns, as repack.cpp prepares them
static void bias_columns(const char * y, size_t by, uint8_t * qb, int n, int nc) {
    for (int c = 0; c < nc; ++c)
        for (int j = 0; j < n; ++j) qb[(size_t)c*n + j] = (uint8_t)(((const block_q8_K *)(y + c*by))[j / QK_K].qs[j % QK_K] ^ 0x80);
}

// IQ3_S repacked at load (signed 4-bit codes): every column vs the original iq3_s kernel
static int test_repack_iq3s(int trials) {
    static const int sizes[] = { 256, 1024, 5120 };
    const size_t by = ggml_row_size(GGML_TYPE_Q8_K, 5120);
    void * x = malloc(ggml_row_size(GGML_TYPE_IQ3_S, 5120));
    block_iq3s_p * xp = malloc(5120 / QK_K * sizeof(block_iq3s_p));
    char * y = malloc(16 * by);
    uint8_t * qb = malloc(16 * 5120);
    int total = 0, bad = 0, packed = 1;
    for (size_t si = 0; si < sizeof sizes / sizeof *sizes; ++si) {
        const int n = sizes[si];
        for (int t = 0; t < trials; ++t) {
            const int nc = 1 + (int)(next_u64() % 16);
            fill_weights(GGML_TYPE_IQ3_S, x, n / QK_K);
            packed &= perseus_iq3s_pack(x, xp, n / QK_K);
            for (int c = 0; c < nc; ++c) fill_activations(GGML_TYPE_Q8_K, y + c*by, n);
            bias_columns(y, by, qb, n, nc);
            float out[16];
            perseus_iq3s_dot_p(n, out, xp, y, by, qb, n, nc);
            for (int c = 0; c < nc; ++c) {
                float ref = 0;
                ggml_vec_dot_iq3_s_q8_K(n, &ref, 0, x, 0, y + c*by, 0, 1);
                ++total;
                if (memcmp(&ref, &out[c], sizeof ref) != 0) {
                    if (bad < 3) printf("  MISMATCH iq3_s repacked n=%d nc=%d col=%d repacked=%.9g original=%.9g\n", n, nc, c, out[c], ref);
                    ++bad;
                }
            }
        }
    }
    free(x); free(xp); free(y); free(qb);
    printf("%-14s %s  %5d/%-5d columns bit-identical (repacked vs original layout)%s\n", "iq3_s_q8_K", bad || !packed ? "FAIL" : "ok  ",
           total - bad, total, packed ? "" : ", PACK REFUSED A BLOCK");
    return bad + !packed;
}
#endif

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// ns per 256-weight block (per row, all columns): best of 5, 64 rows x 5120 weights (L2-sized).
static void bench(void) {
    enum { N = 5120, R = 64 };
    const size_t by = ggml_row_size(GGML_TYPE_Q8_K, N);
    char * y = malloc(16 * by);
    for (int c = 0; c < 16; ++c) fill_activations(GGML_TYPE_Q8_K, y + c*by, N);
    void * y0 = malloc(ggml_row_size(GGML_TYPE_Q8_0, N));
    fill_activations(GGML_TYPE_Q8_0, y0, N);
    printf("bench: ns per 256-weight block, 1 thread; nc = activation columns (tokens)\n");
    for (size_t k = 0; k < sizeof KERNELS / sizeof KERNELS[0]; ++k) {
        const kernel_t * kt = &KERNELS[k];
        const int q80 = kt->type == GGML_TYPE_Q8_0;
        const size_t rs = ggml_row_size(kt->type, N);
        const int nblk = N / 256;
        char * x = malloc(rs * R);
        for (int r = 0; r < R; ++r) fill_weights(kt->type, x + r*rs, N / (int)ggml_blck_size(kt->type));
        static const int ncs[] = { 1, 2, 4, 8 };
        for (int ci = 0; ci < (kt->mc ? 4 : 1); ++ci) {
            const int nc = ncs[ci], iters = 400 / nc;
            double best_s = 1e9, best_m = 1e9;
            float out[16];
            for (int rep = 0; rep < 5; ++rep) {
                double t0 = now_s();
                for (int it = 0; it < iters; ++it)
                    for (int r = 0; r < R; ++r)
                        for (int c = 0; c < nc; ++c) kt->fast(N, &out[c], 0, x + r*rs, 0, q80 ? y0 : y + c*by, 0, 1);
                const double ts = now_s() - t0;
                if (ts < best_s) best_s = ts;
#if !defined(__AVX__) && defined(__SSSE3__)
                if (kt->mc) {
                    t0 = now_s();
                    for (int it = 0; it < iters; ++it)
                        for (int r = 0; r < R; ++r) perseus_vec_dot_mc(kt->type, N, out, x + r*rs, y, by, nc);
                    const double tm = now_s() - t0;
                    if (tm < best_m) best_m = tm;
                }
#endif
            }
            const double per = 1e9 / ((double)iters * R * nblk);
            if (best_m < 1e9) printf("bench %-14s nc=%d single %7.1f  multi-column %7.1f\n", kt->name, nc, best_s * per, best_m * per);
            else              printf("bench %-14s nc=%d single %7.1f\n", kt->name, nc, best_s * per);
        }
        free(x);
    }
#if !defined(__AVX__) && defined(__SSSE3__)
    {   // IQ3_S repacked (what repack.cpp runs), same rows and columns as above
        const size_t rs = N / QK_K * sizeof(block_iq3s_p);
        char * x = malloc(ggml_row_size(GGML_TYPE_IQ3_S, N));
        char * xp = malloc(rs * R);
        uint8_t * qb = malloc(16 * N);
        for (int r = 0; r < R; ++r) {
            fill_weights(GGML_TYPE_IQ3_S, x, N / QK_K);
            if (!perseus_iq3s_pack(x, xp + r*rs, N / QK_K)) abort();
        }
        bias_columns(y, by, qb, N, 8);
        static const int ncs[] = { 1, 2, 4, 8 };
        for (int ci = 0; ci < 4; ++ci) {
            const int nc = ncs[ci], iters = 400 / nc;
            double best = 1e9;
            float out[16];
            for (int rep = 0; rep < 5; ++rep) {
                const double t0 = now_s();
                for (int it = 0; it < iters; ++it)
                    for (int r = 0; r < R; ++r) perseus_iq3s_dot_p(N, out, xp + r*rs, y, by, qb, N, nc);
                const double tm = now_s() - t0;
                if (tm < best) best = tm;
            }
            printf("bench %-14s nc=%d repacked %7.1f\n", "iq3_s_q8_K", nc, best * 1e9 / ((double)iters * R * (N / 256)));
        }
        free(x); free(xp); free(qb);
    }
#endif
    free(y); free(y0);
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        ggml_cpu_init();
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
    ggml_cpu_init();
    printf("kernel path compiled in: %s, %d trials per size\n", path, trials);

    static const int sizes_k[] = { 256, 512, 768, 1024, 4096, 5120, 17408 };
    static const int sizes_0[] = { 32, 64, 96, 160, 5120, 17408 };
    int failures = 0;

    for (size_t k = 0; k < sizeof KERNELS / sizeof KERNELS[0]; ++k) {
        const kernel_t * kt = &KERNELS[k];
        const enum ggml_type vec_type = kt->type == GGML_TYPE_Q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K;
        const int * sizes = kt->type == GGML_TYPE_Q8_0 ? sizes_0 : sizes_k;
        const int nsizes = kt->type == GGML_TYPE_Q8_0 ? (int)(sizeof sizes_0 / sizeof *sizes_0) : (int)(sizeof sizes_k / sizeof *sizes_k);
        int exact = 0, total = 0, bad = 0, nonzero = 0, control_diff = 0;
        double worst = 0;
        for (int si = 0; si < nsizes; ++si) {
            const int n = sizes[si];
            void * x = malloc(ggml_row_size(kt->type, n));
            void * y = malloc(ggml_row_size(vec_type, n));
            for (int t = 0; t < trials; ++t) {
                fill_weights(kt->type, x, n / (int)ggml_blck_size(kt->type));
                fill_activations(vec_type, y, n);
                float a = 0, b = 0;
                kt->fast(n, &a, 0, x, 0, y, 0, 1);
                kt->generic(n, &b, 0, x, 0, y, 0, 1);
                ++total;
                if (a != 0.0f) ++nonzero;
                {   // control: nudge one activation; the fast result must move
                    int8_t * q = vec_type == GGML_TYPE_Q8_0 ? ((block_q8_0 *)y)->qs : ((block_q8_K *)y)->qs;
                    const int8_t saved = q[0];
                    q[0] = (int8_t)(saved > 0 ? saved - 1 : saved + 1);
                    if (vec_type == GGML_TYPE_Q8_K) ((block_q8_K *)y)->bsums[0] += q[0] - saved;
                    float c = 0;
                    kt->fast(n, &c, 0, x, 0, y, 0, 1);
                    if (c != a) ++control_diff;
                    if (vec_type == GGML_TYPE_Q8_K) ((block_q8_K *)y)->bsums[0] -= q[0] - saved;
                    q[0] = saved;
                }
                if (memcmp(&a, &b, sizeof a) == 0) { ++exact; continue; }
                const double scale = abs_dot(kt->type, vec_type, x, y, n);
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
        // a vacuous comparison (all zeros, or insensitive to its inputs) is a failure too
        const int vacuous = nonzero < total * 9 / 10 || control_diff < total / 2;
        printf("%-14s %s  %5d/%-5d bit-exact  worst rel err %.2g  nonzero %d  control moved %d%s\n",
               kt->name, bad || vacuous ? "FAIL" : "ok  ", exact, total, worst, nonzero, control_diff,
               ssse3 && kt->exact ? "  (must be bit-exact)" : "");
        failures += bad + vacuous;
    }
#if !defined(__AVX__) && defined(__SSSE3__)
    for (size_t k = 0; k < sizeof KERNELS / sizeof KERNELS[0]; ++k) {
        if (KERNELS[k].mc) failures += test_mc(&KERNELS[k], trials);
    }
    failures += test_repack_iq3s(trials);
#endif
    printf(failures ? "FAILED: %d mismatches\n" : "all kernels agree with the generic reference\n", failures);
    return failures != 0;
}
