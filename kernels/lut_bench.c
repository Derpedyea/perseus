// Test of the lookup-table idea (T-MAC / bitnet.cpp style) for low-bit weights on SSSE3: instead
// of multiplying, precompute once per activation vector the sum of every subset of each group of
// 4 activations (a 16-entry table), then each weight bit-plane indexes that table, 16 rows at a
// time with one pshufb. Compared with the multiply kernel the SSE build uses (pmaddubsw).
//
// Setting: 2-bit weights (codes 0..3; ternary is the same with codes 0..2 and a -sum(x) term),
// int8 activations, one int16 scale per row per 128 weights (Bonsai's g128). Every kernel must
// produce the reference's exact int32 per row, except the int8-table variant, which is T-MAC's
// lossy shortcut (tables divided by 4 to fit a byte) and only reports its error.
//
// usage: lut_bench   (exit 0 = exact kernels agree; prints ns per 256 weights, one thread)
#include <tmmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { R = 256, K = 4096, G = 128, NG = K / G, NQ = K / 4 };

static uint64_t rng = 0x243F6A8885A308D3ull;
static uint64_t next_u64(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static uint8_t codes[R][K];
static int8_t  x[K];
static int16_t scale[R][NG];
static int32_t ref[R];

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static inline int hsum_i32(__m128i a) {
    a = _mm_add_epi32(a, _mm_unpackhi_epi64(a, a));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(a);
}

// ---- multiply kernel: Q2_K-style layout, 32 bytes per 128 weights, weight 32*s + j in byte j at bits 2s
static uint8_t qa[R][K / 4];

static void pack_a(void) {
    for (int r = 0; r < R; ++r)
        for (int g = 0; g < NG; ++g)
            for (int s = 0; s < 4; ++s)
                for (int j = 0; j < 32; ++j)
                    qa[r][32*g + j] |= (uint8_t)(codes[r][G*g + 32*s + j] << (2*s));
}

static void kernel_mul(int32_t * out) {
    const __m128i m3 = _mm_set1_epi8(3);
    for (int r = 0; r < R; ++r) {
        __m128i acc = _mm_setzero_si128();
        for (int g = 0; g < NG; ++g) {
            const __m128i b0 = _mm_loadu_si128((const __m128i *)&qa[r][32*g]);
            const __m128i b1 = _mm_loadu_si128((const __m128i *)&qa[r][32*g + 16]);
            const int8_t * xg = x + G*g;
            __m128i p = _mm_setzero_si128();  // 8 maddubs pairs of <= 762 each: int16 is safe
            for (int s = 0; s < 4; ++s) {
                const __m128i sh = _mm_cvtsi32_si128(2*s);
                p = _mm_add_epi16(p, _mm_maddubs_epi16(_mm_and_si128(_mm_srl_epi16(b0, sh), m3), _mm_loadu_si128((const __m128i *)(xg + 32*s))));
                p = _mm_add_epi16(p, _mm_maddubs_epi16(_mm_and_si128(_mm_srl_epi16(b1, sh), m3), _mm_loadu_si128((const __m128i *)(xg + 32*s + 16))));
            }
            acc = _mm_add_epi32(acc, _mm_madd_epi16(p, _mm_set1_epi16(scale[r][g])));
        }
        out[r] = hsum_i32(acc);
    }
}

// ---- table kernel: rows in SIMD lanes. For each 16-row tile, each pair of activation quads
// (q, q+1) and each bit-plane p: 16 bytes, byte i = row i's 4 bits for quad q (low nibble) and
// quad q+1 (high nibble). Scales stored per tile as [group][16 rows].
static uint8_t qb[R / 16][NQ / 2][2][16];
static int16_t sb[R / 16][NG][16];
static int8_t  tlo[NQ][16], thi[NQ][16];  // exact table: int16 split into low / high byte
static int8_t  t8[NQ][16];                // lossy table: T / 4 rounded

static void pack_b(void) {
    for (int t = 0; t < R / 16; ++t) {
        for (int qp = 0; qp < NQ / 2; ++qp)
            for (int p = 0; p < 2; ++p)
                for (int i = 0; i < 16; ++i) {
                    uint8_t v = 0;
                    for (int h = 0; h < 2; ++h)
                        for (int j = 0; j < 4; ++j)
                            v |= (uint8_t)(((codes[16*t + i][4*(2*qp + h) + j] >> p) & 1) << (4*h + j));
                    qb[t][qp][p][i] = v;
                }
        for (int g = 0; g < NG; ++g)
            for (int i = 0; i < 16; ++i) sb[t][g][i] = scale[16*t + i][g];
    }
}

// once per activation vector; amortized over every row of the matrix
static void build_tables(void) {
    for (int q = 0; q < NQ; ++q)
        for (int idx = 0; idx < 16; ++idx) {
            int v = 0;
            for (int j = 0; j < 4; ++j) if (idx >> j & 1) v += x[4*q + j];
            tlo[q][idx] = (int8_t)(v & 0xff);
            thi[q][idx] = (int8_t)(v >> 8);
            const int r4 = v >= 0 ? (v + 2) / 4 : -((-v + 2) / 4);
            t8[q][idx] = (int8_t)(r4 > 127 ? 127 : r4 < -127 ? -127 : r4);
        }
}

// int16 lanes (8 rows) times int16 scales -> two int32 vectors, added to acc[0..1]
static inline void mul_widen_add(__m128i v, __m128i s, __m128i * acc) {
    const __m128i lo = _mm_mullo_epi16(v, s), hi = _mm_mulhi_epi16(v, s);
    acc[0] = _mm_add_epi32(acc[0], _mm_unpacklo_epi16(lo, hi));
    acc[1] = _mm_add_epi32(acc[1], _mm_unpackhi_epi16(lo, hi));
}

static void kernel_lut(int32_t * out, int exact) {
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (int t = 0; t < R / 16; ++t) {
        __m128i acc[4] = { _mm_setzero_si128(), _mm_setzero_si128(), _mm_setzero_si128(), _mm_setzero_si128() };
        for (int g = 0; g < NG; ++g) {
            // per plane, rows 0-7 and 8-15 in int16: 32 lookups of |T| <= 508 sum to <= 16256
            __m128i a[2][2] = { { _mm_setzero_si128(), _mm_setzero_si128() }, { _mm_setzero_si128(), _mm_setzero_si128() } };
            for (int qp = G/8*g; qp < G/8*(g + 1); ++qp) {
                for (int p = 0; p < 2; ++p) {
                    const __m128i v = _mm_loadu_si128((const __m128i *)qb[t][qp][p]);
                    const __m128i idx[2] = { _mm_and_si128(v, m4), _mm_and_si128(_mm_srli_epi16(v, 4), m4) };
                    for (int h = 0; h < 2; ++h) {
                        const int q = 2*qp + h;
                        if (exact) {
                            const __m128i rl = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)tlo[q]), idx[h]);
                            const __m128i rh = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)thi[q]), idx[h]);
                            a[p][0] = _mm_add_epi16(a[p][0], _mm_unpacklo_epi8(rl, rh));
                            a[p][1] = _mm_add_epi16(a[p][1], _mm_unpackhi_epi8(rl, rh));
                        } else {
                            const __m128i r8 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)t8[q]), idx[h]);
                            a[p][0] = _mm_add_epi16(a[p][0], _mm_srai_epi16(_mm_unpacklo_epi8(r8, r8), 8));
                            a[p][1] = _mm_add_epi16(a[p][1], _mm_srai_epi16(_mm_unpackhi_epi8(r8, r8), 8));
                        }
                    }
                }
            }
            const __m128i s0 = _mm_loadu_si128((const __m128i *)&sb[t][g][0]), s1 = _mm_loadu_si128((const __m128i *)&sb[t][g][8]);
            mul_widen_add(a[0][0], s0, &acc[0]);
            mul_widen_add(a[0][1], s1, &acc[2]);
            mul_widen_add(a[1][0], _mm_slli_epi16(s0, 1), &acc[0]);  // plane 1 counts double
            mul_widen_add(a[1][1], _mm_slli_epi16(s1, 1), &acc[2]);
        }
        int32_t o[16];
        for (int k = 0; k < 4; ++k) _mm_storeu_si128((__m128i *)o + k, acc[k]);
        for (int i = 0; i < 16; ++i) out[16*t + i] = exact ? o[i] : 4*o[i];
    }
}

int main(void) {
    for (int r = 0; r < R; ++r) {
        for (int k = 0; k < K; ++k) codes[r][k] = (uint8_t)(next_u64() & 3);
        for (int g = 0; g < NG; ++g) scale[r][g] = (int16_t)(1 + next_u64() % 31);
    }
    for (int k = 0; k < K; ++k) x[k] = (int8_t)((int)(next_u64() % 255) - 127);
    for (int r = 0; r < R; ++r) {
        int64_t s = 0;
        for (int g = 0; g < NG; ++g) {
            int gs = 0;
            for (int k = G*g; k < G*(g + 1); ++k) gs += codes[r][k] * x[k];
            s += (int64_t)scale[r][g] * gs;
        }
        ref[r] = (int32_t)s;
    }
    pack_a();
    pack_b();
    build_tables();

    static int32_t out[R];
    int fail = 0;
    kernel_mul(out);
    for (int r = 0; r < R; ++r) fail += out[r] != ref[r];
    printf("lut: multiply kernel %s\n", fail ? "WRONG" : "exact");
    int fail_l = 0;
    kernel_lut(out, 1);
    for (int r = 0; r < R; ++r) fail_l += out[r] != ref[r];
    printf("lut: table kernel (exact int16 tables) %s\n", fail_l ? "WRONG" : "exact");
    kernel_lut(out, 0);
    double worst = 0, mag = 0;
    for (int r = 0; r < R; ++r) {
        const double e = abs(out[r] - ref[r]);
        if (e > worst) worst = e;
        mag += abs(ref[r]);
    }
    printf("lut: table kernel (lossy int8 tables) max error %.0f (%.2g of mean |result|)\n", worst, worst / (mag / R));

    // ns per 256 weights, best of 7; the tables' cost per activation vector is reported apart
    // because a real matrix reuses them over thousands of rows, not 256
    const double per = 1e9 / ((double)R * K / 256);
    double best[3] = { 1e9, 1e9, 1e9 }, best_t = 1e9;
    for (int rep = 0; rep < 7; ++rep) {
        const double t0 = now_s();
        for (int it = 0; it < 20; ++it) kernel_mul(out);
        const double t1 = now_s();
        for (int it = 0; it < 20; ++it) kernel_lut(out, 1);
        const double t2 = now_s();
        for (int it = 0; it < 20; ++it) kernel_lut(out, 0);
        const double t3 = now_s();
        for (int it = 0; it < 20; ++it) build_tables();
        const double t4 = now_s();
        if ((t1 - t0) / 20 < best[0]) best[0] = (t1 - t0) / 20;
        if ((t2 - t1) / 20 < best[1]) best[1] = (t2 - t1) / 20;
        if ((t3 - t2) / 20 < best[2]) best[2] = (t3 - t2) / 20;
        if ((t4 - t3) / 20 < best_t) best_t = (t4 - t3) / 20;
    }
    printf("lut bench: ns per 256 weights, 1 thread, 2-bit weights, %d rows x %d\n", R, K);
    printf("lut bench multiply (pmaddubsw)      %6.2f\n", best[0] * per);
    printf("lut bench table, exact              %6.2f\n", best[1] * per);
    printf("lut bench table, lossy int8         %6.2f\n", best[2] * per);
    printf("lut bench table build per vector    %6.0f ns (%d activations)\n", best_t * 1e9, K);
    return fail || fail_l;
}
