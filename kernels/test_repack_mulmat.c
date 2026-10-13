// End-to-end check of the perseus IQ3_S load-time repack (ggml-cpu/repack.cpp): the same random
// IQ3_S weights multiplied once from a plain CPU buffer (llama.cpp's own kernels, with the
// tiled path off) and once from the CPU_REPACK buffer (converted at load, multiplied by the
// perseus traits). Every output must match bit for bit, across column counts around the 4- and
// 16-column groups, row counts that leave partial 16-row tiles, 3-D activations and thread counts.
// Refuses a vacuous pass: the repacked weight must actually carry the repack traits.
//
// Build: see kernels/build-test.sh. Exit code 0 = all pass.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0xD1B54A32D192ED03ull;
static uint64_t next_u64(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static ggml_backend_buffer_type_t repack_buft(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    ggml_backend_dev_get_extra_bufts_t get = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t * bufts = get ? get(dev) : NULL;
    for (; bufts && *bufts; ++bufts) {
        if (strcmp(ggml_backend_buft_name(*bufts), "CPU_REPACK") == 0) return *bufts;
    }
    return NULL;
}

// random IQ3_S blocks (all grid indices, signs and scales) with a sane fp16 scale per block
static void fill_iq3s(void * data, int64_t nblocks) {
    const size_t bs = ggml_type_size(GGML_TYPE_IQ3_S);
    uint8_t * p = data;
    for (size_t i = 0; i < bs * nblocks; ++i) p[i] = (uint8_t) next_u64();
    for (int64_t b = 0; b < nblocks; ++b) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.001f + 0.01f * (float)(next_u64() % 1000) / 1000.0f);
        memcpy(p + b*bs, &d, sizeof d);  // block_iq3_s starts with its fp16 d
    }
}

// dst for weights w (K x N) times activations (K x M x B), weights placed in buffer type buft
static void run(ggml_backend_t backend, ggml_backend_buffer_type_t buft, const void * wdata, const float * xdata,
                int64_t K, int64_t N, int64_t M, int64_t B, float * out, int * repacked) {
    struct ggml_init_params pw = { ggml_tensor_overhead() * 2, NULL, true };
    struct ggml_context * cw = ggml_init(pw);
    struct ggml_tensor * w = ggml_new_tensor_2d(cw, GGML_TYPE_IQ3_S, K, N);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, buft);
    *repacked = w->extra != NULL;
    if (!*repacked && buft != ggml_backend_cpu_buffer_type()) {
        // CPU_REPACK chose no traits for this tensor (llama.cpp would not place it there)
        ggml_backend_buffer_free(bw);
        ggml_free(cw);
        return;
    }
    ggml_backend_tensor_set(w, wdata, 0, ggml_nbytes(w));

    struct ggml_init_params px = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true };
    struct ggml_context * cx = ggml_init(px);
    struct ggml_tensor * x = ggml_new_tensor_3d(cx, GGML_TYPE_F32, K, M, B);
    struct ggml_tensor * y = ggml_mul_mat(cx, w, x);
    struct ggml_cgraph * gf = ggml_new_graph(cx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors(cx, backend);
    ggml_backend_tensor_set(x, xdata, 0, ggml_nbytes(x));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(1);
    }
    ggml_backend_tensor_get(y, out, 0, ggml_nbytes(y));
    ggml_backend_buffer_free(bx);
    ggml_backend_buffer_free(bw);
    ggml_free(cx);
    ggml_free(cw);
}

int main(void) {
    // the reference is llama.cpp's own path: vec_dot (+ perseus multi-column), not the tiled
    // matmul, whose float summation order differs; the repacked path must equal it exactly
    setenv("GGML_CPU_TILED_MM", "0", 1);
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    ggml_backend_buffer_type_t rbuft = repack_buft();
    if (!backend || !rbuft) {
        fprintf(stderr, "no CPU backend or no CPU_REPACK buffer type\n");
        return 1;
    }
    static const int64_t Ks[] = { 256, 1280 };
    static const int64_t Ns[] = { 16, 100, 257 };
    static const int64_t Ms[] = { 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 33 };
    static const int threads[] = { 1, 3, 8 };
    int fails = 0, cases = 0, not_repacked = 0;
    for (size_t ki = 0; ki < sizeof Ks / sizeof *Ks; ++ki)
    for (size_t ni = 0; ni < sizeof Ns / sizeof *Ns; ++ni)
    for (size_t mi = 0; mi < sizeof Ms / sizeof *Ms; ++mi)
    for (int B = 1; B <= 2; ++B)
    for (size_t ti = 0; ti < sizeof threads / sizeof *threads; ++ti) {
        const int64_t K = Ks[ki], N = Ns[ni], M = Ms[mi];
        ggml_backend_cpu_set_n_threads(backend, threads[ti]);
        void * w = malloc(ggml_row_size(GGML_TYPE_IQ3_S, K) * N);
        float * x = malloc(sizeof(float) * K * M * B);
        float * ref = malloc(sizeof(float) * N * M * B);
        float * got = malloc(sizeof(float) * N * M * B);
        fill_iq3s(w, K * N / 256);
        for (int64_t i = 0; i < K * M * B; ++i) x[i] = (float)((int64_t)(next_u64() % 2001) - 1000) / 250.0f;
        int r_plain, r_repack;
        run(backend, ggml_backend_cpu_buffer_type(), w, x, K, N, M, B, ref, &r_plain);
        run(backend, rbuft, w, x, K, N, M, B, got, &r_repack);
        ++cases;
        not_repacked += !r_repack || r_plain;
        int bad = 0;
        for (int64_t i = 0; i < N * M * B; ++i) bad += memcmp(&ref[i], &got[i], sizeof(float)) != 0;
        if (bad || !r_repack) {
            if (fails < 5) printf("  FAIL K=%lld N=%lld M=%lld B=%d threads=%d: %d of %lld outputs differ%s\n",
                                  (long long)K, (long long)N, (long long)M, B, threads[ti], bad, (long long)(N*M*B),
                                  r_repack ? "" : ", weight was not repacked");
            ++fails;
        }
        free(w); free(x); free(ref); free(got);
    }
    printf("repack mulmat: %d/%d cases bit-identical to the plain path%s\n", cases - fails, cases,
           not_repacked ? " (REPACK NOT ENGAGED; PERSEUS_REPACK=0 set?)" : "");
    ggml_backend_free(backend);
    return fails != 0 || not_repacked != 0;
}
