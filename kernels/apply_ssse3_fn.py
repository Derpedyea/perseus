#!/usr/bin/env python3
"""Adds SSSE3 dot-product kernels for the formats Qwen3.8-Flash-Next's GGUFs read every token:

- q6_K, q5_K: the output layer (Q6_K in nearly every build) and dense weights of the low-bit builds
- iq4_nl:     the expert down projection (row length 640 rules out every block-256 format)
- q2_0:       lookup-free 2-bit experts (ISTA GSQ-RCO builds); llama.cpp has no x86 kernel at all
- q4_1:       not a Flash-Next format, but 8 ffn_down tensors (446 MB) of the kit's Qwen3.8-27B
              Q4_0; upstream's x86 kernel is AVX-only, and generic q4_1 cost 8x q4_0 per weight

Same approach as apply_ssse3.py: the integer math of the AVX paths on 128-bit vectors, each
block summed exactly before one float multiply. iq4_nl and q2_0 keep the generic float order
(bit-exact), as does q4_1; q6_K and q5_K sum each block before scaling, as the AVX paths do.

Runs after apply_ssse3.py (uses its helpers). Tested by kernels/test_fn_kernels.c.

usage: apply_ssse3_fn.py path/to/llama.cpp   (edits in place; refuses to run twice)
"""
import os
import sys

BASE_MARK = "// perseus: SSSE3 kernels"
MARK = "// perseus: SSSE3 kernels for Flash-Next formats"

HELPERS = MARK + r"""
#if !defined(__AVX__) && defined(__SSSE3__)
// int8 -> int16 sign extension of the low 8 bytes (_mm_cvtepi8_epi16 is SSE4.1)
static inline __m128i sext8_16_ssse3(const __m128i x) {
    return _mm_unpacklo_epi8(x, _mm_cmpgt_epi8(_mm_setzero_si128(), x));
}
#endif

"""

# Inserted before the "#else" that follows each function's AVX branch.
KERNELS = {}

KERNELS["q6_K_q8_K"] = r"""#elif defined(__SSSE3__)
    const __m128i m3   = _mm_set1_epi8(3);
    const __m128i m12  = _mm_set1_epi8(12);
    const __m128i m48  = _mm_set1_epi8(48);
    const __m128i m192 = _mm_set1_epi8(-64);
    const __m128i m15  = _mm_set1_epi8(15);

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // one int8 scale per 16 weights, widened once; the -32 offset comes from the bsums
        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i sc16[2] = { sext8_16_ssse3(scales), sext8_16_ssse3(_mm_srli_si128(scales, 8)) };
        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)y[i].bsums);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)y[i].bsums + 1);
        const __m128i offset = _mm_slli_epi32(_mm_add_epi32(_mm_madd_epi16(q8sums_0, sc16[0]), _mm_madd_epi16(q8sums_1, sc16[1])), 5);

        __m128i sumi = _mm_setzero_si128();
        for (int j = 0; j < QK_K/128; ++j) {
            const __m128i qh_0 = _mm_loadu_si128((const __m128i*)qh); qh += 16;
            const __m128i qh_1 = _mm_loadu_si128((const __m128i*)qh); qh += 16;
            const __m128i ql_0 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i ql_1 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i ql_2 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i ql_3 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;

            // 6-bit weights 0..63 of elements 16k..16k+15, k = 0..7
            __m128i w[8];
            w[0] = _mm_or_si128(_mm_and_si128(ql_0, m15), _mm_slli_epi16(_mm_and_si128(qh_0, m3), 4));
            w[1] = _mm_or_si128(_mm_and_si128(ql_1, m15), _mm_slli_epi16(_mm_and_si128(qh_1, m3), 4));
            w[2] = _mm_or_si128(_mm_and_si128(ql_2, m15), _mm_slli_epi16(_mm_and_si128(qh_0, m12), 2));
            w[3] = _mm_or_si128(_mm_and_si128(ql_3, m15), _mm_slli_epi16(_mm_and_si128(qh_1, m12), 2));
            w[4] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql_0, 4), m15), _mm_and_si128(qh_0, m48));
            w[5] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql_1, 4), m15), _mm_and_si128(qh_1, m48));
            w[6] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql_2, 4), m15), _mm_srli_epi16(_mm_and_si128(qh_0, m192), 2));
            w[7] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql_3, 4), m15), _mm_srli_epi16(_mm_and_si128(qh_1, m192), 2));

            for (int k = 0; k < 8; ++k) {
                const __m128i q8k = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
                const __m128i sc = _mm_shuffle_epi8(sc16[j], bcast16_mask_ssse3(k));
                sumi = _mm_add_epi32(sumi, _mm_madd_epi16(sc, _mm_maddubs_epi16(w[k], q8k)));
            }
        }

        sumf += d * hsum_i32_4_ssse3(_mm_sub_epi32(sumi, offset));
    }

    *s = sumf;

"""

KERNELS["q5_K_q8_K"] = r"""#elif defined(__SSSE3__)
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);
    const __m128i zero = _mm_setzero_si128();

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q5 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        // bytes 0..7: 6-bit scales, bytes 8..15: 6-bit mins (all non-negative)
        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_unpacklo_epi8(utmps, zero);
        const __m128i mins   = _mm_unpackhi_epi8(utmps, zero);

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const int summs = hsum_i32_4_ssse3(_mm_madd_epi16(mins, q8s));

        const __m128i hbits_0 = _mm_loadu_si128((const __m128i*)&x[i].qh[0]);
        const __m128i hbits_1 = _mm_loadu_si128((const __m128i*)&x[i].qh[16]);

        __m128i sumi = zero;
        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {
            const __m128i scale_l = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_h = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            // high bit 2j (low nibbles) and 2j+1 (high nibbles) of each weight, moved to bit 4
            const __m128i hl = _mm_set1_epi8((char)(1 << (2*j)));
            const __m128i hh = _mm_set1_epi8((char)(1 << (2*j + 1)));
            const __m128i b4 = _mm_set1_epi8(16);

            const __m128i q5bits_0 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;
            const __m128i q5bits_1 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;
            const __m128i q5l_0 = _mm_or_si128(_mm_and_si128(q5bits_0, m4), _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(hbits_0, hl), hl), b4));
            const __m128i q5l_1 = _mm_or_si128(_mm_and_si128(q5bits_1, m4), _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(hbits_1, hl), hl), b4));
            const __m128i q5h_0 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q5bits_0, 4), m4), _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(hbits_0, hh), hh), b4));
            const __m128i q5h_1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q5bits_1, 4), m4), _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(hbits_1, hh), hh), b4));

            const __m128i q8l_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8l_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8h_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8h_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            const __m128i pl  = _mm_madd_epi16(scale_l, _mm_maddubs_epi16(q5l_0, q8l_0));
            const __m128i pl1 = _mm_madd_epi16(scale_l, _mm_maddubs_epi16(q5l_1, q8l_1));
            const __m128i ph  = _mm_madd_epi16(scale_h, _mm_maddubs_epi16(q5h_0, q8h_0));
            const __m128i ph1 = _mm_madd_epi16(scale_h, _mm_maddubs_epi16(q5h_1, q8h_1));
            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_add_epi32(pl, pl1), _mm_add_epi32(ph, ph1)));
        }

        sumf += d * hsum_i32_4_ssse3(sumi) - dmin * summs;
    }

    *s = sumf;

"""

# iq4_nl has no "#else" after its AVX branch (a scalar loop finishes the blocks after "#endif");
# this branch handles every block itself, in the generic float order.
IQ4_NL = r"""#elif defined(__SSSE3__)
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m128i mone = _mm_set1_epi16(1);

    for (; ib < nb; ++ib) {
        const __m128i q4bits = _mm_loadu_si128((const __m128i *)x[ib].qs);
        const __m128i q8_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        const __m128i q8_1 = _mm_loadu_si128((const __m128i *)y[ib].qs + 1);
        const __m128i q4_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits, m4b));
        const __m128i q4_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4b));
        // each int16 pair is at most 2*127*127, so the halves are widened before adding
        const __m128i p = _mm_add_epi32(_mm_madd_epi16(mul_add_epi8_ssse3(q4_0, q8_0), mone),
                                        _mm_madd_epi16(mul_add_epi8_ssse3(q4_1, q8_1), mone));
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_FP16_TO_FP32(x[ib].d);
        sumf += d * hsum_i32_4_ssse3(p);
    }

"""

# q4_1's only x86 branch is AVX ("#if defined(__AVX2__) || defined(__AVX__)" ... "#else" generic);
# this goes before that "#else". Unsigned 4-bit weights times int8 activations: pmaddubsw pairs
# stay within +-3840, so the block sum is exact; then the generic code's float expression.
Q4_1_ANCHOR = "#else\n    UNUSED(nb);\n    UNUSED(x);\n    UNUSED(y);\n    UNUSED(ib);\n    ggml_vec_dot_q4_1_q8_1_generic("
Q4_1 = r"""#elif defined(__SSSE3__)
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m128i mone = _mm_set1_epi16(1);
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const __m128i q4bits = _mm_loadu_si128((const __m128i *)x[ib].qs);
        const __m128i lo = _mm_and_si128(q4bits, m4b);                     // weights 0..15
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4b);  // weights 16..31
        const __m128i p = _mm_add_epi16(_mm_maddubs_epi16(lo, _mm_loadu_si128((const __m128i *)y[ib].qs)),
                                        _mm_maddubs_epi16(hi, _mm_loadu_si128((const __m128i *)y[ib].qs + 1)));
        const int sumi = hsum_i32_4_ssse3(_mm_madd_epi16(p, mone));
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
"""

# Older llama.cpp has no x86 q2_0 kernel (arch-fallback.h maps the generic one onto the public name);
# newer has an AVX2-only one, which gets this body as its SSSE3 branch.
Q2_0_FALLBACK = "#define ggml_vec_dot_q2_0_q8_0_generic ggml_vec_dot_q2_0_q8_0\n"
Q2_0 = r"""
// perseus: q2_0 (64 weights per block, codes 0..3 = -1..2 times the scale, 4 per byte with
// weight 4b+t in bits 2t of byte b). Bit-exact with the generic float order.
void ggml_vec_dot_q2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__SSSE3__)
    const int nb = n / QK2_0;

    assert(n % QK2_0 == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const __m128i m3     = _mm_set1_epi8(3);
    const __m128i ones8  = _mm_set1_epi8(1);
    const __m128i ones16 = _mm_set1_epi16(1);
    // 16 activations of 4 consecutive bytes b: lane 4t+b gets activation 4b+t
    const __m128i tr = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const __m128i qs = _mm_loadu_si128((const __m128i *)x[i].qs);
        // c_t, byte b: code of weight 4b+t
        const __m128i c0 = _mm_and_si128(qs, m3);
        const __m128i c1 = _mm_and_si128(_mm_srli_epi16(qs, 2), m3);
        const __m128i c2 = _mm_and_si128(_mm_srli_epi16(qs, 4), m3);
        const __m128i c3 = _mm_and_si128(_mm_srli_epi16(qs, 6), m3);
        // 4x4 transpose of 32-bit groups: w_k = codes of weights 16k..16k+15 in tr's order
        const __m128i c01l = _mm_unpacklo_epi32(c0, c1), c23l = _mm_unpacklo_epi32(c2, c3);
        const __m128i c01h = _mm_unpackhi_epi32(c0, c1), c23h = _mm_unpackhi_epi32(c2, c3);
        const __m128i w0 = _mm_unpacklo_epi64(c01l, c23l);
        const __m128i w1 = _mm_unpackhi_epi64(c01l, c23l);
        const __m128i w2 = _mm_unpacklo_epi64(c01h, c23h);
        const __m128i w3 = _mm_unpackhi_epi64(c01h, c23h);

        const __m128i y0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)y[2*i + 0].qs + 0), tr);
        const __m128i y1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)y[2*i + 0].qs + 1), tr);
        const __m128i y2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)y[2*i + 1].qs + 0), tr);
        const __m128i y3 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)y[2*i + 1].qs + 1), tr);

        // sum (code - 1) * q = sum code*q - sum q; every int16 lane stays within +-2032
        const __m128i p0 = _mm_add_epi16(_mm_sub_epi16(_mm_maddubs_epi16(w0, y0), _mm_maddubs_epi16(ones8, y0)),
                                         _mm_sub_epi16(_mm_maddubs_epi16(w1, y1), _mm_maddubs_epi16(ones8, y1)));
        const __m128i p1 = _mm_add_epi16(_mm_sub_epi16(_mm_maddubs_epi16(w2, y2), _mm_maddubs_epi16(ones8, y2)),
                                         _mm_sub_epi16(_mm_maddubs_epi16(w3, y3), _mm_maddubs_epi16(ones8, y3)));
        // lane 0: the first q8_0 block's sum, lane 1: the second's
        __m128i h = _mm_hadd_epi32(_mm_madd_epi16(p0, ones16), _mm_madd_epi16(p1, ones16));
        h = _mm_hadd_epi32(h, h);

        float sumi = 0.0f;
        sumi += GGML_CPU_FP16_TO_FP32(y[2*i + 0].d) * _mm_cvtsi128_si32(h);
        sumi += GGML_CPU_FP16_TO_FP32(y[2*i + 1].d) * _mm_cvtsi128_si32(_mm_srli_si128(h, 4));
        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * sumi;
    }

    *s = sumf;
#else
    ggml_vec_dot_q2_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
"""


def function_span(src, name):
    head = f"void ggml_vec_dot_{name}("
    assert src.count(head) == 1, name
    start = src.index(head)
    end = src.index("\n}\n", start) + 3
    return start, end


def main():
    root = sys.argv[1]
    cpu = os.path.join(root, "ggml", "src", "ggml-cpu")
    path = os.path.join(cpu, "arch", "x86", "quants.c")
    src = open(path).read()
    if MARK in src:
        sys.exit(f"{path} already has the perseus Flash-Next kernels")
    if BASE_MARK not in src:
        sys.exit(f"{path}: run apply_ssse3.py first (these kernels use its helpers)")

    anchor = "void ggml_vec_dot_q1_0_q8_0("
    assert src.count(anchor) == 1
    src = src.replace(anchor, HELPERS + anchor)

    for name, body in KERNELS.items():
        start, end = function_span(src, name)
        fn = src[start:end]
        avx = fn.find("#elif defined __AVX__")
        assert avx > 0, name
        els = fn.index("\n#else\n", avx) + 1
        src = src[:start] + fn[:els] + body + fn[els:] + src[end:]

    start, end = function_span(src, "iq4_nl_q8_0")
    fn = src[start:end]
    avx = fn.find("#elif defined __AVX__")
    assert avx > 0, "iq4_nl_q8_0"
    endif = fn.index("\n#endif\n", avx) + 1
    src = src[:start] + fn[:endif] + IQ4_NL + fn[endif:] + src[end:]

    start, end = function_span(src, "q4_1_q8_1")
    fn = src[start:end]
    assert fn.count(Q4_1_ANCHOR) == 1, "q4_1_q8_1 changed upstream"
    src = src[:start] + fn.replace(Q4_1_ANCHOR, Q4_1 + Q4_1_ANCHOR) + src[end:]

    if "ggml_vec_dot_q2_0_q8_0(" in src:
        # llama.cpp since 2026-10 has an x86 q2_0 kernel, AVX2 only: the SSSE3 body goes in as
        # its #elif, like the kernels above
        body = Q2_0[Q2_0.index("#if defined(__SSSE3__)\n"):Q2_0.index("#else\n")]
        start, end = function_span(src, "q2_0_q8_0")
        fn = src[start:end]
        assert fn.count("#if defined(__AVX2__)") == 1 and fn.count("\n#else\n") == 1, "q2_0_q8_0 changed upstream"
        els = fn.index("\n#else\n") + 1
        src = src[:start] + fn[:els] + body.replace("#if", "#elif", 1) + fn[els:] + src[end:]
        open(path, "w").write(src)
    else:
        open(path, "w").write(src + Q2_0)
        path = os.path.join(cpu, "arch-fallback.h")
        src = open(path).read()
        x86 = src.index("#elif defined(__x86_64__) || defined(__i386__)")
        nxt = src.index("#elif", x86 + 1)
        block = src[x86:nxt]
        assert block.count(Q2_0_FALLBACK) == 1, "x86 q2_0 fallback changed upstream"
        open(path, "w").write(src[:x86] + block.replace(Q2_0_FALLBACK, "") + src[nxt:])

    print(f"added SSSE3 kernels: {', '.join(KERNELS)}, iq4_nl_q8_0, q4_1_q8_1, q2_0_q8_0")


if __name__ == "__main__":
    main()
