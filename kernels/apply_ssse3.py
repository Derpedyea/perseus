#!/usr/bin/env python3
"""Adds SSSE3 (pre-AVX x86) dot-product kernels to llama.cpp's ggml-cpu/arch/x86/quants.c.

The kernels mirror the integer math of the existing AVX paths, which already work on 128-bit
integer vectors; only their float accumulation is 256-bit. Here each block's int32 lanes are
summed exactly and scaled once in float, in the generic code's order where that is cheap.

Also repacks IQ3_S at load time (ggml-cpu/repack.cpp) and adds multi-column versions (one weight row against 2..16 activation columns, decoding
each weight block once) and hooks them into ggml-cpu.c's mul_mat chunk loop.

usage: apply_ssse3.py path/to/llama.cpp   (edits in place; refuses to run twice)
"""
import os
import sys

MARK = "// perseus: SSSE3 kernels"

HELPERS = MARK + r"""
#if !defined(__AVX__) && defined(__SSSE3__)
// Pre-AVX x86 (Penryn/Nehalem/Westmere). See the SSSE3 branches of the vec_dot functions below.
static inline int hsum_i32_4_ssse3(const __m128i a) {
    const __m128i hi64  = _mm_unpackhi_epi64(a, a);
    const __m128i sum64 = _mm_add_epi32(hi64, a);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// signed int8 * signed int8, adjacent pairs summed into int16
static inline __m128i mul_add_epi8_ssse3(const __m128i x, const __m128i y) {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    return _mm_maddubs_epi16(ax, sy);
}

// broadcast 16-bit lane i of a vector via _mm_shuffle_epi8 (a table load, not built per use)
static inline __m128i bcast16_mask_ssse3(int i) {
    static const uint8_t k_bcast16[8][16] = {
        {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1}, {2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3},
        {4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5}, {6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7},
        {8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9}, {10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11},
        {12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13}, {14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15},
    };
    return _mm_loadu_si128((const __m128i *)k_bcast16[i]);
}

static inline __m128i get_scale_shuffle_ssse3(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}
#endif

"""

# Inserted before the "#else" that follows each function's AVX branch.
KERNELS = {}

KERNELS["q2_K_q8_K"] = r"""#elif defined(__SSSE3__)
    const __m128i m3 = _mm_set1_epi8(0x3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);
    const __m128i zero = _mm_setzero_si128();

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // 4-bit scales and mins are non-negative, so zero-extension equals sign-extension
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales16 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins16 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m128i summs_v = _mm_add_epi32(
            _mm_madd_epi16(_mm_unpacklo_epi8(mins16, zero), _mm_loadu_si128((const __m128i*)&y[i].bsums[0])),
            _mm_madd_epi16(_mm_unpackhi_epi8(mins16, zero), _mm_loadu_si128((const __m128i*)&y[i].bsums[8])));
        const __m128i scales[2] = { _mm_unpacklo_epi8(scales16, zero), _mm_unpackhi_epi8(scales16, zero) };

        __m128i sumi = zero;
        for (int j = 0; j < QK_K/128; ++j) {
            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            __m128i q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
            const __m128i q2_0 = _mm_and_si128(q2bits, m3);
            const __m128i q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_4 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_6 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);
            q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
            const __m128i q2_1 = _mm_and_si128(q2bits, m3);
            const __m128i q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_5 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_7 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);

            __m128i p0 = _mm_maddubs_epi16(q2_0, q8_0);
            __m128i p1 = _mm_maddubs_epi16(q2_1, q8_1);
            __m128i p2 = _mm_maddubs_epi16(q2_2, q8_2);
            __m128i p3 = _mm_maddubs_epi16(q2_3, q8_3);
            __m128i p4 = _mm_maddubs_epi16(q2_4, q8_4);
            __m128i p5 = _mm_maddubs_epi16(q2_5, q8_5);
            __m128i p6 = _mm_maddubs_epi16(q2_6, q8_6);
            __m128i p7 = _mm_maddubs_epi16(q2_7, q8_7);

            __m128i shuffle = _mm_set1_epi16(0x0100);
            p0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p0);
            shuffle = _mm_add_epi16(shuffle, m2);
            p1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p1);
            shuffle = _mm_add_epi16(shuffle, m2);
            p2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p2);
            shuffle = _mm_add_epi16(shuffle, m2);
            p3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p3);
            shuffle = _mm_add_epi16(shuffle, m2);
            p4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p4);
            shuffle = _mm_add_epi16(shuffle, m2);
            p5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p5);
            shuffle = _mm_add_epi16(shuffle, m2);
            p6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p6);
            shuffle = _mm_add_epi16(shuffle, m2);
            p7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p7);

            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_add_epi32(p0, p1), _mm_add_epi32(p2, p3)));
            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_add_epi32(p4, p5), _mm_add_epi32(p6, p7)));
        }

        const int isum  = hsum_i32_4_ssse3(sumi);
        const int summs = hsum_i32_4_ssse3(summs_v);
        sumf += dall * isum - dmin * summs;
    }

    *s = sumf;

"""

KERNELS["q4_K_q8_K"] = r"""#elif defined(__SSSE3__)
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);
    const __m128i zero = _mm_setzero_si128();

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
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

        __m128i sumi = zero;
        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {
            const __m128i scale_l = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_h = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            __m128i q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_0 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);
            q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_1 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);

            const __m128i q8l_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8l_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8h_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8h_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            const __m128i pl = _mm_madd_epi16(scale_l, _mm_maddubs_epi16(q4l_0, q8l_0));
            const __m128i pl1 = _mm_madd_epi16(scale_l, _mm_maddubs_epi16(q4l_1, q8l_1));
            const __m128i ph = _mm_madd_epi16(scale_h, _mm_maddubs_epi16(q4h_0, q8h_0));
            const __m128i ph1 = _mm_madd_epi16(scale_h, _mm_maddubs_epi16(q4h_1, q8h_1));
            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_add_epi32(pl, pl1), _mm_add_epi32(ph, ph1)));
        }

        sumf += d * hsum_i32_4_ssse3(sumi) - dmin * summs;
    }

    *s = sumf;

"""

KERNELS["iq2_xxs_q8_K"] = r"""#elif defined(__SSSE3__)
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            const __m128i q2_1_0 = _mm_set_epi64x(iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]]);
            const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127]);
            const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, _mm_sign_epi8(q8_1_0, s2_1_0));
            const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, _mm_sign_epi8(q8_1_1, s2_1_1));
            const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, _mm_sign_epi8(q8_2_0, s2_2_0));
            const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, _mm_sign_epi8(q8_2_1, s2_2_1));
            const __m128i ls1 = _mm_set1_epi16(2*(aux32[1] >> 28) + 1);
            const __m128i ls2 = _mm_set1_epi16(2*(aux32[3] >> 28) + 1);
            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_madd_epi16(dot1_0, ls1), _mm_madd_epi16(dot1_1, ls1)));
            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_madd_epi16(dot2_0, ls2), _mm_madd_epi16(dot2_1, ls2)));
        }
        sumf += d * hsum_i32_4_ssse3(sumi);
    }

    *s = 0.125f * sumf;

"""

KERNELS["iq2_xs_q8_K"] = r"""#elif defined(__SSSE3__)
    const __m128i mone = _mm_set1_epi8(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[16] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };
    static const uint8_t k_bit_helper[16] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };

    const __m128i bit_selector_mask = _mm_loadu_si128((const __m128i*)bit_selector_mask_bytes);
    const __m128i block_sign_shuffle_1_0 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_1);
    const __m128i block_sign_shuffle_1_1 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_1 + 1);
    const __m128i block_sign_shuffle_2_0 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_2);
    const __m128i block_sign_shuffle_2_1 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_2 + 1);
    const __m128i bit_helper = _mm_loadu_si128((const __m128i*)k_bit_helper);
    const __m128i m511 = _mm_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m128i zero = _mm_setzero_si128();

    uint64_t aux64;
    uint16_t gindex[16];

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);  // 2*s+1, all positive

        __m128i sumi = zero;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 4) {
            const __m128i q2_data_0 = _mm_loadu_si128((const __m128i*)q2);
            const __m128i q2_data_1 = _mm_loadu_si128((const __m128i*)q2 + 1);  q2 += 16;
            _mm_storeu_si128((__m128i *)gindex,     _mm_and_si128(q2_data_0, m511));
            _mm_storeu_si128((__m128i *)gindex + 1, _mm_and_si128(q2_data_1, m511));

            // the 8th sign bit of each group is implied by even parity of the 7 stored ones
            const __m128i partial_sign_bits_0 = _mm_srli_epi16(q2_data_0, 9);
            const __m128i partial_sign_bits_1 = _mm_srli_epi16(q2_data_1, 9);
            const __m128i partial_sign_bits_upper_0 = _mm_srli_epi16(q2_data_0, 13);
            const __m128i partial_sign_bits_upper_1 = _mm_srli_epi16(q2_data_1, 13);
            const __m128i odd_bits_0 = _mm_shuffle_epi8(bit_helper, _mm_xor_si128(partial_sign_bits_0, partial_sign_bits_upper_0));
            const __m128i odd_bits_1 = _mm_shuffle_epi8(bit_helper, _mm_xor_si128(partial_sign_bits_1, partial_sign_bits_upper_1));
            const __m128i full_sign_bits_0 = _mm_or_si128(partial_sign_bits_0, odd_bits_0);
            const __m128i full_sign_bits_1 = _mm_or_si128(partial_sign_bits_1, odd_bits_1);

            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_3_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_3_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_4_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_4_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;

            const __m128i q2_1_0 = _mm_set_epi64x(iq2xs_grid[gindex[1]],  iq2xs_grid[gindex[0]]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2xs_grid[gindex[3]],  iq2xs_grid[gindex[2]]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2xs_grid[gindex[5]],  iq2xs_grid[gindex[4]]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2xs_grid[gindex[7]],  iq2xs_grid[gindex[6]]);
            const __m128i q2_3_0 = _mm_set_epi64x(iq2xs_grid[gindex[9]],  iq2xs_grid[gindex[8]]);
            const __m128i q2_3_1 = _mm_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]]);
            const __m128i q2_4_0 = _mm_set_epi64x(iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);
            const __m128i q2_4_1 = _mm_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]]);

            __m128i signs_0, signs_1;
            signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask), bit_selector_mask);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask), bit_selector_mask);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask), bit_selector_mask);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask), bit_selector_mask);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask), bit_selector_mask);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask), bit_selector_mask);
            const __m128i q8s_3_0 = _mm_sign_epi8(q8_3_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_3_1 = _mm_sign_epi8(q8_3_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask), bit_selector_mask);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask), bit_selector_mask);
            const __m128i q8s_4_0 = _mm_sign_epi8(q8_4_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_4_1 = _mm_sign_epi8(q8_4_1, _mm_or_si128(signs_1, mone));

            const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const __m128i dot3_0 = _mm_maddubs_epi16(q2_3_0, q8s_3_0);
            const __m128i dot3_1 = _mm_maddubs_epi16(q2_3_1, q8s_3_1);
            const __m128i dot4_0 = _mm_maddubs_epi16(q2_4_0, q8s_4_0);
            const __m128i dot4_1 = _mm_maddubs_epi16(q2_4_1, q8s_4_1);

            __m128i sc = _mm_shuffle_epi8(scales, get_scale_shuffle_ssse3(ib32+0));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot1_0, _mm_unpacklo_epi8(sc, zero)));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot1_1, _mm_unpackhi_epi8(sc, zero)));
            sc = _mm_shuffle_epi8(scales, get_scale_shuffle_ssse3(ib32+1));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot2_0, _mm_unpacklo_epi8(sc, zero)));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot2_1, _mm_unpackhi_epi8(sc, zero)));
            sc = _mm_shuffle_epi8(scales, get_scale_shuffle_ssse3(ib32+2));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot3_0, _mm_unpacklo_epi8(sc, zero)));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot3_1, _mm_unpackhi_epi8(sc, zero)));
            sc = _mm_shuffle_epi8(scales, get_scale_shuffle_ssse3(ib32+3));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot4_0, _mm_unpacklo_epi8(sc, zero)));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot4_1, _mm_unpackhi_epi8(sc, zero)));
        }
        sumf += d * hsum_i32_4_ssse3(sumi);
    }

    *s = 0.125f * sumf;

"""

KERNELS["iq2_s_q8_K"] = r"""#elif defined(__SSSE3__)
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };
    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m128i zero = _mm_setzero_si128();
    const __m128i mask1_0 = _mm_loadu_si128((const __m128i*)k_mask1);
    const __m128i mask1_1 = _mm_loadu_si128((const __m128i*)k_mask1 + 1);
    const __m128i mask2   = _mm_loadu_si128((const __m128i*)k_mask2);

    uint64_t aux64;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // bytes 0..7: low-nibble scales, 8..15: high-nibble scales, each 2*s+1 (positive)
        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m128i scales16_0 = _mm_unpacklo_epi8(scales8, zero);
        const __m128i scales16_1 = _mm_unpackhi_epi8(scales8, zero);

        __m128i sumi = zero;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q2_1_0 = _mm_set_epi64x(iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                  iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                  iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                  iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                  iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)]);
            qs += 8;

            __m128i aux128_0 = _mm_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            __m128i aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0, mask1_0), mask2);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1, mask1_1), mask2);
            const __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2);
            const __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2);
            const __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            const __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

            aux128_0 = _mm_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0, mask1_0), mask2);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1, mask1_1), mask2);
            const __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2);
            const __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2);
            const __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            const __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

            signs += 4;

            const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);

            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot1_0, _mm_shuffle_epi8(scales16_0, bcast16_mask_ssse3(ib32+0))));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot1_1, _mm_shuffle_epi8(scales16_1, bcast16_mask_ssse3(ib32+0))));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot2_0, _mm_shuffle_epi8(scales16_0, bcast16_mask_ssse3(ib32+1))));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(dot2_1, _mm_shuffle_epi8(scales16_1, bcast16_mask_ssse3(ib32+1))));
        }
        sumf += d * hsum_i32_4_ssse3(sumi);
    }

    *s = 0.125f * sumf;

"""

KERNELS["iq3_xxs_q8_K"] = r"""#elif defined(__SSSE3__)
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q2_1_0 = _mm_set_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m128i q2_1_1 = _mm_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            const __m128i q2_2_0 = _mm_set_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m128i q2_2_1 = _mm_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;
            const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127]);
            const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, _mm_sign_epi8(q8_1_0, s2_1_0));
            const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, _mm_sign_epi8(q8_1_1, s2_1_1));
            const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, _mm_sign_epi8(q8_2_0, s2_2_0));
            const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, _mm_sign_epi8(q8_2_1, s2_2_1));
            const __m128i ls1 = _mm_set1_epi16(2*(aux32[0] >> 28) + 1);
            const __m128i ls2 = _mm_set1_epi16(2*(aux32[1] >> 28) + 1);
            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_madd_epi16(dot1_0, ls1), _mm_madd_epi16(dot1_1, ls1)));
            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_madd_epi16(dot2_0, ls2), _mm_madd_epi16(dot2_1, ls2)));
        }
        sumf += d * hsum_i32_4_ssse3(sumi);
    }

    *s = 0.25f * sumf;

"""

KERNELS["iq3_s_q8_K"] = r"""#elif defined(__SSSE3__)
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };
    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

    const __m128i mask1_0 = _mm_loadu_si128((const __m128i*)k_mask1);
    const __m128i mask1_1 = _mm_loadu_si128((const __m128i*)k_mask1 + 1);
    const __m128i mask2   = _mm_loadu_si128((const __m128i*)k_mask2);

    uint32_t idx[16];

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            // 9-bit grid index: 8 bits from qs, the 9th from one bit of qh
            for (int l = 0; l < 8; ++l) {
                idx[l]     = qs[l]     | ((qh[ib32+0] << (8 - l)) & 256);
                idx[l + 8] = qs[l + 8] | ((qh[ib32+1] << (8 - l)) & 256);
            }
            qs += 16;
            const __m128i q2_1_0 = _mm_set_epi32(iq3s_grid[idx[3]],  iq3s_grid[idx[2]],  iq3s_grid[idx[1]],  iq3s_grid[idx[0]]);
            const __m128i q2_1_1 = _mm_set_epi32(iq3s_grid[idx[7]],  iq3s_grid[idx[6]],  iq3s_grid[idx[5]],  iq3s_grid[idx[4]]);
            const __m128i q2_2_0 = _mm_set_epi32(iq3s_grid[idx[11]], iq3s_grid[idx[10]], iq3s_grid[idx[9]],  iq3s_grid[idx[8]]);
            const __m128i q2_2_1 = _mm_set_epi32(iq3s_grid[idx[15]], iq3s_grid[idx[14]], iq3s_grid[idx[13]], iq3s_grid[idx[12]]);

            __m128i aux128_0 = _mm_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            __m128i aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0, mask1_0), mask2);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1, mask1_1), mask2);
            const __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2);
            const __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2);
            const __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            const __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

            aux128_0 = _mm_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0, mask1_0), mask2);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1, mask1_1), mask2);
            const __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2);
            const __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2);
            const __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            const __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

            signs += 4;

            const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const __m128i ls1 = _mm_set1_epi16(2*(x[i].scales[ib32/2] & 0xf) + 1);
            const __m128i ls2 = _mm_set1_epi16(2*(x[i].scales[ib32/2] >>  4) + 1);
            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_madd_epi16(dot1_0, ls1), _mm_madd_epi16(dot1_1, ls1)));
            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_madd_epi16(dot2_0, ls2), _mm_madd_epi16(dot2_1, ls2)));
        }
        sumf += d * hsum_i32_4_ssse3(sumi);
    }

    *s = sumf;

"""

# IQ1 weights are grid + delta with grid in {-1,0,1} and delta = +-1/8. As integers,
# 8*grid +- 1 = u - 9 with u = 8*(grid+1) + (delta > 0 ? 2 : 0) in [0, 18], unsigned, so the dot
# is one maddubs(u, q8) per 16 weights and a -9*sum(q8) correction from the block's bsums: no sign
# ops, no separate delta pass. The integer total T is the generic code's 8*sumi + sumi1, and
# T * 0.125f equals its float(sumi) + 0.125f*float(sumi1) exactly (|T| < 2^23, so both are exact).
# perseus_iq1s_u8 holds 8*(grid+1) per byte; apply_ssse3.py generates it from iq1s_grid.
KERNELS["iq1_s_q8_K"] = r"""#elif defined(__SSSE3__)
    const __m128i m7 = _mm_set1_epi16(7);
    const __m128i one16 = _mm_set1_epi16(1);
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        // per 32-weight sub-block: scale 2*s+1 (int16) and the delta byte (0 if negative, else 2)
        const __m128i qhv = _mm_loadu_si128((const __m128i *)qh);
        const __m128i ls = _mm_or_si128(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 12), m7), 1), one16);
        const __m128i d8 = _mm_packus_epi16(_mm_sub_epi16(_mm_set1_epi16(2), _mm_slli_epi16(_mm_srli_epi16(qhv, 15), 1)), _mm_setzero_si128());
        const __m128i bsum32 = _mm_hadd_epi16(_mm_loadu_si128((const __m128i *)&y[i].bsums[0]), _mm_loadu_si128((const __m128i *)&y[i].bsums[8]));
        const int corr = hsum_i32_4_ssse3(_mm_madd_epi16(bsum32, ls));

        __m128i sumi = _mm_setzero_si128();
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const __m128i dlt = _mm_shuffle_epi8(d8, _mm_set1_epi8(ib));
            const __m128i u0 = _mm_add_epi8(_mm_set_epi64x(perseus_iq1s_u8[qs[1] | ((qh[ib] << 5) & 0x700)], perseus_iq1s_u8[qs[0] | ((qh[ib] << 8) & 0x700)]), dlt);
            const __m128i u1 = _mm_add_epi8(_mm_set_epi64x(perseus_iq1s_u8[qs[3] | ((qh[ib] >> 1) & 0x700)], perseus_iq1s_u8[qs[2] | ((qh[ib] << 2) & 0x700)]), dlt);
            qs += 4;
            // two maddubs of u <= 18 and |q8| <= 127 add up to at most 9144: int16 is safe
            const __m128i p = _mm_add_epi16(_mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)q8)),
                                            _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)q8 + 1)));
            q8 += 32;
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(p, _mm_shuffle_epi8(ls, bcast16_mask_ssse3(ib))));
        }
        const int T = hsum_i32_4_ssse3(sumi) - 9*corr;
        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d * (T * IQ1S_DELTA);
    }

    *s = sumf;

"""

# Same u trick as iq1_s; here the delta sign changes every 8 weights and the scale every 16.
KERNELS["iq1_m_q8_K"] = r"""#elif defined(__SSSE3__)
    // delta bytes for one 16-weight chunk, indexed by its two delta bits (set = negative = 0)
    static const uint8_t k_dlt[4][16] = {
        {2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2}, {0,0,0,0,0,0,0,0, 2,2,2,2,2,2,2,2},
        {2,2,2,2,2,2,2,2, 0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0},
    };
    const __m128i sc_lo = _mm_setr_epi8(0,1,0,1,0,1,0,1, 2,3,2,3,2,3,2,3);
    const __m128i sc_hi = _mm_setr_epi8(4,5,4,5,4,5,4,5, 6,7,6,7,6,7,6,7);
    const __m128i field_mul = _mm_setr_epi16(1 << 13, 1 << 10, 1 << 7, 1 << 4, 1 << 13, 1 << 10, 1 << 7, 1 << 4);
    const __m128i one16 = _mm_set1_epi16(1);
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        // 3-bit scale per 16 weights (bits 3m..3m+2 of sc[k/4], m = k%4), as 2*s+1: multiplying
        // by 2^(13-3m) moves the field to the top of the 16-bit lane, >> 13 brings it down
        const __m128i scv = _mm_loadl_epi64((const __m128i *)sc);
        const __m128i ls[2] = {
            _mm_or_si128(_mm_slli_epi16(_mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(scv, sc_lo), field_mul), 13), 1), one16),
            _mm_or_si128(_mm_slli_epi16(_mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(scv, sc_hi), field_mul), 13), 1), one16),
        };
        const int corr = hsum_i32_4_ssse3(_mm_add_epi32(_mm_madd_epi16(_mm_loadu_si128((const __m128i *)&y[i].bsums[0]), ls[0]),
                                                        _mm_madd_epi16(_mm_loadu_si128((const __m128i *)&y[i].bsums[8]), ls[1])));

        __m128i sumi = _mm_setzero_si128();
        for (int k = 0; k < 16; ++k) {
            const __m128i dlt = _mm_loadu_si128((const __m128i *)k_dlt[((qh[0] >> 3) & 1) | ((qh[0] >> 6) & 2)]);
            const __m128i u = _mm_add_epi8(_mm_set_epi64x(
                perseus_iq1s_u8[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], perseus_iq1s_u8[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]), dlt);
            qs += 2; qh += 1;
            const __m128i p = _mm_maddubs_epi16(u, _mm_loadu_si128((const __m128i *)q8));
            q8 += 16;
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(p, _mm_shuffle_epi8(ls[k/8], bcast16_mask_ssse3(k%8))));
        }
        const int T = hsum_i32_4_ssse3(sumi) - 9*corr;
        sumf += GGML_CPU_FP16_TO_FP32(scale.f16) * y[i].d * (T * IQ1M_DELTA);
    }

    *s = sumf;

"""

KERNELS["iq4_xs_q8_K"] = r"""#elif defined(__SSSE3__)
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m128i sumi = _mm_setzero_si128();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
            const __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
            const __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
            const __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));
            const __m128i p16_1_0 = mul_add_epi8_ssse3(q4b_1_0, q8b_1_0);
            const __m128i p16_1_1 = mul_add_epi8_ssse3(q4b_1_1, q8b_1_1);
            const __m128i p16_2_0 = mul_add_epi8_ssse3(q4b_2_0, q8b_2_0);
            const __m128i p16_2_1 = mul_add_epi8_ssse3(q4b_2_1, q8b_2_1);
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            const __m128i vls1 = _mm_set1_epi16(ls1);
            const __m128i vls2 = _mm_set1_epi16(ls2);
            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_madd_epi16(p16_1_0, vls1), _mm_madd_epi16(p16_1_1, vls1)));
            sumi = _mm_add_epi32(sumi, _mm_add_epi32(_mm_madd_epi16(p16_2_0, vls2), _mm_madd_epi16(p16_2_1, vls2)));
        }
        sumf += GGML_CPU_FP16_TO_FP32(x[ibl].d) * y[ibl].d * hsum_i32_4_ssse3(sumi);
    }

    *s = sumf;

"""

# Appended to quants.c, after the single-column kernels whose tables and helpers it uses.
MC = r"""
// perseus: multi-column kernels
#if !defined(__AVX__) && defined(__SSSE3__)
// Dot products of one weight row with 1..16 activation columns (MTP verification, short
// batches; 8+ go to the tiled matmul). Each 32-weight sub-block is decoded once into registers
// and dotted with up to 4 columns before the next one is decoded; more columns repeat that per
// group of 4. Per column, the integer sums and float steps are exactly those of the
// single-column kernels above, so the results are bit-identical to calling them column by column.
#include <stdlib.h>

#define PERSEUS_MC_MAX 16

// one decoded 32-weight sub-block: magnitudes g (u8; unsigned formats: the weights themselves),
// sign sources s (_mm_sign_epi8(q8, s) = q8 * sign(w)), chunk scales sc (int16 broadcast)
typedef struct { __m128i g0, s0, g1, s1, sc0, sc1; } perseus_sub_t;

// sign bytes (+1/-1) from 32 packed sign bits, as in the iq2_s / iq3_s kernels
static inline void perseus_signs32(uint32_t bits, __m128i * s0, __m128i * s1) {
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};
    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const __m128i mask2 = _mm_loadu_si128((const __m128i*)k_mask2);
    const __m128i one = _mm_set1_epi8(1);
    const __m128i v = _mm_set1_epi32(bits);
    *s0 = _mm_or_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(v, _mm_loadu_si128((const __m128i*)k_mask1)), mask2), mask2), one);
    *s1 = _mm_or_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(v, _mm_loadu_si128((const __m128i*)k_mask1 + 1)), mask2), mask2), one);
}

static inline int perseus_summs_q2_K(const __m128i mins[2], const int16_t * bsums) {
    return hsum_i32_4_ssse3(_mm_add_epi32(_mm_madd_epi16(mins[0], _mm_loadu_si128((const __m128i*)&bsums[0])),
                                          _mm_madd_epi16(mins[1], _mm_loadu_si128((const __m128i*)&bsums[8]))));
}

// iq2_xs sign expansion: parity of 7 sign bits by nibble, bit selectors, lane broadcasts
static const uint8_t perseus_iq2xs_bit_helper[16] = { 0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00 };
static const uint8_t perseus_bit_sel[16] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };
static const uint8_t perseus_iq2xs_shuf[2][16] = { {0,0,0,0,0,0,0,0, 2,2,2,2,2,2,2,2}, {4,4,4,4,4,4,4,4, 6,6,6,6,6,6,6,6} };

// delta bytes for one 16-weight iq1_m chunk, indexed by its two delta bits (set = negative = 0)
static const uint8_t perseus_iq1m_dlt[4][16] = {
    {2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2}, {0,0,0,0,0,0,0,0, 2,2,2,2,2,2,2,2},
    {2,2,2,2,2,2,2,2, 0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0},
};

// one column, one sub-block: signed weights use the sign trick; PAIR adds the two chunks in
// int16 before scaling (only when both share a scale and the sum cannot overflow)
#define PERSEUS_COL(acc, q, SIGNED, PAIR) do {                                                          \
        const __m128i y_0 = _mm_loadu_si128((const __m128i *)(q));                                     \
        const __m128i y_1 = _mm_loadu_si128((const __m128i *)(q) + 1);                                 \
        const __m128i p0 = _mm_maddubs_epi16(u.g0, (SIGNED) ? _mm_sign_epi8(y_0, u.s0) : y_0);        \
        const __m128i p1 = _mm_maddubs_epi16(u.g1, (SIGNED) ? _mm_sign_epi8(y_1, u.s1) : y_1);        \
        acc = _mm_add_epi32(acc, (PAIR) ? _mm_madd_epi16(_mm_add_epi16(p0, p1), u.sc0)                 \
                                        : _mm_add_epi32(_mm_madd_epi16(p0, u.sc0), _mm_madd_epi16(p1, u.sc1))); \
    } while (0)

// One function per format. BSETUP runs once per weight block (locals for SUB and EPI); SUB
// decodes sub-block ib into u; EPI adds column k's float term to sumf[k], written exactly as
// the single-column kernel's (same operands, same order), from isum and y.
#define PERSEUS_MC(NAME, BLOCK, SIGNED, PAIR, BSETUP, SUB, EPI, FINAL)                                  \
static void perseus_mc_##NAME(int n, float * s, const void * vx, const char * vy, size_t by, int nc) { \
    const BLOCK * x = (const BLOCK *)vx;                                                               \
    for (int c0 = 0; c0 < nc; c0 += 4) {                                                               \
        const int m = nc - c0 < 4 ? nc - c0 : 4;                                                      \
        const char * yc[4];                                                                            \
        for (int k = 0; k < 4; ++k) yc[k] = vy + (size_t)(c0 + (k < m ? k : 0))*by;                    \
        float sumf[4] = {0};                                                                           \
        for (int i = 0; i < n/QK_K; ++i) {                                                             \
            BSETUP;                                                                                    \
            const block_q8_K * yb[4] = { (const block_q8_K *)yc[0] + i, (const block_q8_K *)yc[1] + i, \
                                         (const block_q8_K *)yc[2] + i, (const block_q8_K *)yc[3] + i }; \
            __m128i a0 = _mm_setzero_si128(), a1 = a0, a2 = a0, a3 = a0;                             \
            for (int ib = 0; ib < QK_K/32; ++ib) {                                                     \
                perseus_sub_t u;                                                                       \
                SUB;                                                                                   \
                PERSEUS_COL(a0, yb[0]->qs + 32*ib, SIGNED, PAIR);                                      \
                if (m > 1) PERSEUS_COL(a1, yb[1]->qs + 32*ib, SIGNED, PAIR);                          \
                if (m > 2) PERSEUS_COL(a2, yb[2]->qs + 32*ib, SIGNED, PAIR);                          \
                if (m > 3) PERSEUS_COL(a3, yb[3]->qs + 32*ib, SIGNED, PAIR);                          \
            }                                                                                          \
            const __m128i acc[4] = { a0, a1, a2, a3 };                                                 \
            for (int k = 0; k < m; ++k) {                                                              \
                const block_q8_K * y = yb[k];                                                          \
                const int isum = hsum_i32_4_ssse3(acc[k]);                                             \
                EPI;                                                                                   \
            }                                                                                          \
        }                                                                                              \
        for (int k = 0; k < m; ++k) {                                                                  \
            s[c0 + k] = FINAL * sumf[k];                                                               \
        }                                                                                              \
    }                                                                                                  \
}

// max |grid| 43, chunks share a per-32 scale: PAIR is safe (2 * 2*43*127 < 32768)
PERSEUS_MC(iq2_xxs, block_iq2_xxs, 1, 1,
    const float dx = GGML_CPU_FP16_TO_FP32(x[i].d); const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs,
    uint32_t a[2]; memcpy(a, x[i].qs + 4*ib, 8); const uint8_t * a8 = (const uint8_t *)a;
    u.g0 = _mm_set_epi64x(iq2xxs_grid[a8[1]], iq2xxs_grid[a8[0]]);
    u.g1 = _mm_set_epi64x(iq2xxs_grid[a8[3]], iq2xxs_grid[a8[2]]);
    u.s0 = _mm_set_epi64x(signs64[(a[1] >>  7) & 127], signs64[(a[1] >>  0) & 127]);
    u.s1 = _mm_set_epi64x(signs64[(a[1] >> 21) & 127], signs64[(a[1] >> 14) & 127]);
    u.sc0 = u.sc1 = _mm_set1_epi16(2*(a[1] >> 28) + 1),
    sumf[k] += dx * y->d * isum, 0.125f)

// per-16 scales; the 8th sign bit of each group of 8 is the even parity of the 7 stored ones
PERSEUS_MC(iq2_xs, block_iq2_xs, 1, 0,
    const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);
    const __m128i bit_helper = _mm_loadu_si128((const __m128i *)perseus_iq2xs_bit_helper);
    const __m128i bsm = _mm_loadu_si128((const __m128i *)perseus_bit_sel); const __m128i one = _mm_set1_epi8(1),
    const uint16_t * q = x[i].qs + 4*ib;
    const __m128i qv = _mm_loadl_epi64((const __m128i *)q);
    const __m128i partial = _mm_srli_epi16(qv, 9);
    const __m128i full = _mm_or_si128(partial, _mm_shuffle_epi8(bit_helper, _mm_xor_si128(partial, _mm_srli_epi16(qv, 13))));
    u.s0 = _mm_or_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(full, _mm_loadu_si128((const __m128i *)perseus_iq2xs_shuf[0])), bsm), bsm), one);
    u.s1 = _mm_or_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(full, _mm_loadu_si128((const __m128i *)perseus_iq2xs_shuf[1])), bsm), bsm), one);
    u.g0 = _mm_set_epi64x(iq2xs_grid[q[1] & 511], iq2xs_grid[q[0] & 511]);
    u.g1 = _mm_set_epi64x(iq2xs_grid[q[3] & 511], iq2xs_grid[q[2] & 511]);
    u.sc0 = _mm_set1_epi16(2*(x[i].scales[ib] & 0xf) + 1);
    u.sc1 = _mm_set1_epi16(2*(x[i].scales[ib] >> 4) + 1),
    sumf[k] += dx * y->d * isum, 0.125f)

PERSEUS_MC(iq2_s, block_iq2_s, 1, 0,
    const float dx = GGML_CPU_FP16_TO_FP32(x[i].d),
    const uint8_t * qs = x[i].qs + 4*ib; const int qh = x[i].qh[ib];
    const uint16_t * sg = (const uint16_t *)(x[i].qs + QK_K/8) + 2*ib;
    u.g0 = _mm_set_epi64x(iq2s_grid[qs[1] | ((qh << 6) & 0x300)], iq2s_grid[qs[0] | ((qh << 8) & 0x300)]);
    u.g1 = _mm_set_epi64x(iq2s_grid[qs[3] | ((qh << 2) & 0x300)], iq2s_grid[qs[2] | ((qh << 4) & 0x300)]);
    perseus_signs32(sg[0] | ((uint32_t) sg[1] << 16), &u.s0, &u.s1);
    u.sc0 = _mm_set1_epi16(2*(x[i].scales[ib] & 0xf) + 1);
    u.sc1 = _mm_set1_epi16(2*(x[i].scales[ib] >> 4) + 1),
    sumf[k] += dx * y->d * isum, 0.125f)

// max |grid| 62: PAIR still fits (2 * 2*62*127 = 31496)
PERSEUS_MC(iq3_xxs, block_iq3_xxs, 1, 1,
    const float dx = GGML_CPU_FP16_TO_FP32(x[i].d); const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs,
    const uint8_t * q3 = x[i].qs + 8*ib; uint32_t a; memcpy(&a, x[i].qs + QK_K/4 + 4*ib, 4);
    u.g0 = _mm_set_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
    u.g1 = _mm_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
    u.s0 = _mm_set_epi64x(signs64[(a >>  7) & 127], signs64[(a >>  0) & 127]);
    u.s1 = _mm_set_epi64x(signs64[(a >> 21) & 127], signs64[(a >> 14) & 127]);
    u.sc0 = u.sc1 = _mm_set1_epi16(2*(a >> 28) + 1),
    sumf[k] += dx * y->d * isum, 0.25f)

// 9-bit grid index: 8 bits from qs, the 9th from bit l of qh
PERSEUS_MC(iq3_s, block_iq3_s, 1, 1,
    const float dx = GGML_CPU_FP16_TO_FP32(x[i].d),
    const uint8_t * qs = x[i].qs + 8*ib; const int qh = x[i].qh[ib];
    const uint16_t * sg = (const uint16_t *)x[i].signs + 2*ib;
    u.g0 = _mm_set_epi32(iq3s_grid[qs[3] | ((qh << 5) & 256)], iq3s_grid[qs[2] | ((qh << 6) & 256)],
                         iq3s_grid[qs[1] | ((qh << 7) & 256)], iq3s_grid[qs[0] | ((qh << 8) & 256)]);
    u.g1 = _mm_set_epi32(iq3s_grid[qs[7] | ((qh << 1) & 256)], iq3s_grid[qs[6] | ((qh << 2) & 256)],
                         iq3s_grid[qs[5] | ((qh << 3) & 256)], iq3s_grid[qs[4] | ((qh << 4) & 256)]);
    perseus_signs32(sg[0] | ((uint32_t) sg[1] << 16), &u.s0, &u.s1);
    u.sc0 = u.sc1 = _mm_set1_epi16(2*((x[i].scales[ib/2] >> (4*(ib%2))) & 0xf) + 1),
    sumf[k] += dx * y->d * isum, 1.0f)

// unsigned u = 8*(grid+1) + {0,2} as in the single-column kernel; T = isum - 9*sum(ls*q8)
PERSEUS_MC(iq1_s, block_iq1_s, 0, 1,
    const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);
    const __m128i qhv = _mm_loadu_si128((const __m128i *)x[i].qh);
    const __m128i ls = _mm_or_si128(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 12), _mm_set1_epi16(7)), 1), _mm_set1_epi16(1)),
    const uint8_t * qs = x[i].qs + 4*ib; const int qh = x[i].qh[ib];
    const __m128i dlt = _mm_set1_epi8(qh & 0x8000 ? 0 : 2);
    u.g0 = _mm_add_epi8(_mm_set_epi64x(perseus_iq1s_u8[qs[1] | ((qh << 5) & 0x700)], perseus_iq1s_u8[qs[0] | ((qh << 8) & 0x700)]), dlt);
    u.g1 = _mm_add_epi8(_mm_set_epi64x(perseus_iq1s_u8[qs[3] | ((qh >> 1) & 0x700)], perseus_iq1s_u8[qs[2] | ((qh << 2) & 0x700)]), dlt);
    u.s0 = u.s1 = _mm_setzero_si128();
    u.sc0 = u.sc1 = _mm_set1_epi16(2*((qh >> 12) & 7) + 1),
    const int corr = hsum_i32_4_ssse3(_mm_madd_epi16(_mm_hadd_epi16(_mm_loadu_si128((const __m128i *)&y->bsums[0]), _mm_loadu_si128((const __m128i *)&y->bsums[8])), ls));
    const int T = isum - 9*corr;
    sumf[k] += dx * y->d * (T * IQ1S_DELTA), 1.0f)

PERSEUS_MC(iq1_m, block_iq1_m, 0, 0,
    const uint16_t * sc = (const uint16_t *)x[i].scales; iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float dx = GGML_CPU_FP16_TO_FP32(scale.f16);
    const __m128i scv = _mm_loadl_epi64((const __m128i *)sc);
    const __m128i field_mul = _mm_setr_epi16(1 << 13, 1 << 10, 1 << 7, 1 << 4, 1 << 13, 1 << 10, 1 << 7, 1 << 4);
    __m128i ls[2];
    ls[0] = _mm_or_si128(_mm_slli_epi16(_mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(scv, _mm_setr_epi8(0,1,0,1,0,1,0,1, 2,3,2,3,2,3,2,3)), field_mul), 13), 1), _mm_set1_epi16(1));
    ls[1] = _mm_or_si128(_mm_slli_epi16(_mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(scv, _mm_setr_epi8(4,5,4,5,4,5,4,5, 6,7,6,7,6,7,6,7)), field_mul), 13), 1), _mm_set1_epi16(1)),
    const uint8_t * qs = x[i].qs + 4*ib; const uint8_t * qh = x[i].qh + 2*ib;
    u.g0 = _mm_add_epi8(_mm_set_epi64x(perseus_iq1s_u8[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], perseus_iq1s_u8[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]),
                        _mm_loadu_si128((const __m128i *)perseus_iq1m_dlt[((qh[0] >> 3) & 1) | ((qh[0] >> 6) & 2)]));
    u.g1 = _mm_add_epi8(_mm_set_epi64x(perseus_iq1s_u8[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], perseus_iq1s_u8[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)]),
                        _mm_loadu_si128((const __m128i *)perseus_iq1m_dlt[((qh[1] >> 3) & 1) | ((qh[1] >> 6) & 2)]));
    u.s0 = u.s1 = _mm_setzero_si128();
    u.sc0 = _mm_shuffle_epi8(ls[ib/4], bcast16_mask_ssse3((2*ib) % 8));
    u.sc1 = _mm_shuffle_epi8(ls[ib/4], bcast16_mask_ssse3((2*ib + 1) % 8)),
    const int corr = hsum_i32_4_ssse3(_mm_add_epi32(_mm_madd_epi16(_mm_loadu_si128((const __m128i *)&y->bsums[0]), ls[0]),
                                                    _mm_madd_epi16(_mm_loadu_si128((const __m128i *)&y->bsums[8]), ls[1])));
    const int T = isum - 9*corr;
    sumf[k] += dx * y->d * (T * IQ1M_DELTA), 1.0f)

// |w| up to 127: no PAIR (two chunks could overflow int16)
PERSEUS_MC(iq4_xs, block_iq4_xs, 1, 0,
    const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl); const __m128i m4b = _mm_set1_epi8(0x0f),
    const __m128i q4 = _mm_loadu_si128((const __m128i *)(x[i].qs + 16*ib));
    const __m128i v0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4, m4b));
    const __m128i v1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4, 4), m4b));
    u.g0 = _mm_abs_epi8(v0); u.s0 = v0; u.g1 = _mm_abs_epi8(v1); u.s1 = v1;
    u.sc0 = u.sc1 = _mm_set1_epi16((((x[i].scales_l[ib/2] >> (4*(ib%2))) & 0xf) | (((x[i].scales_h >> (2*ib)) & 3) << 4)) - 32),
    sumf[k] += dx * y->d * isum, 1.0f)

// chunk 8j + 2m + h holds the 2-bit field m of bytes 32j + 16h .. +15, so sub-block ib = 4j + m
PERSEUS_MC(q2_K, block_q2_K, 0, 0,
    const float xd = GGML_CPU_FP16_TO_FP32(x[i].d); const float xdmin = GGML_CPU_FP16_TO_FP32(x[i].dmin);
    // 4-bit mins are non-negative, so zero-extension equals sign-extension
    const __m128i mins16 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i*)x[i].scales), 4), _mm_set1_epi8(0xF));
    __m128i mins[2];
    mins[0] = _mm_unpacklo_epi8(mins16, _mm_setzero_si128());
    mins[1] = _mm_unpackhi_epi8(mins16, _mm_setzero_si128());
    const __m128i m3 = _mm_set1_epi8(3),
    const __m128i sh = _mm_cvtsi32_si128(2*(ib % 4));
    u.g0 = _mm_and_si128(_mm_srl_epi16(_mm_loadu_si128((const __m128i *)(x[i].qs + 32*(ib/4))), sh), m3);
    u.g1 = _mm_and_si128(_mm_srl_epi16(_mm_loadu_si128((const __m128i *)(x[i].qs + 32*(ib/4) + 16)), sh), m3);
    u.s0 = u.s1 = _mm_setzero_si128();
    u.sc0 = _mm_set1_epi16(x[i].scales[2*ib] & 0xf);
    u.sc1 = _mm_set1_epi16(x[i].scales[2*ib + 1] & 0xf),
    sumf[k] += y->d * xd * isum - y->d * xdmin * perseus_summs_q2_K(mins, y->bsums), 1.0f)

// IQ3_S load-time repack (used by ggml-cpu/repack.cpp). Each weight is grid (1, 3, ..., 15)
// times a sign: 16 values, so one signed 4-bit code per weight and two pshufb per 32 weights
// decode it, instead of 8 scalar grid lookups and a sign expansion. 144 bytes per 256 weights
// instead of 110. The activations are biased once per matmul (q8 + 128, unsigned), so the dot is
// maddubs(biased, signed weight) with no sign op; corr = 128 * sum(scale * weight) per block,
// computed at load, takes the bias back out. The integers equal the original kernel's exactly.
static const int8_t perseus_iq3s_levels[16] = { 1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15 };

bool perseus_iq3s_pack(const void * vx, void * vy, int64_t nblocks) {
    const block_iq3_s * x = (const block_iq3_s *)vx;
    block_iq3s_p * y = (block_iq3s_p *)vy;
    for (int64_t b = 0; b < nblocks; ++b) {
        y[b].d = ggml_fp16_to_fp32(x[b].d);
        int32_t corr = 0;
        memset(y[b].qs, 0, sizeof y[b].qs);
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = 2*((x[b].scales[ib/2] >> (4*(ib%2))) & 0xf) + 1;
            y[b].sc[ib] = (int8_t)ls;
            int32_t wsum = 0;
            for (int w = 0; w < 32; ++w) {
                const int m = w / 4;  // grid entry: 4 weights, 9-bit index (8 bits qs, 1 bit qh)
                const uint8_t * grid = (const uint8_t *)(iq3s_grid + (x[b].qs[8*ib + m] | ((x[b].qh[ib] << (8 - m)) & 256)));
                const int g = grid[w % 4];
                const int neg = (x[b].signs[4*ib + w/8] >> (w % 8)) & 1;
                if (g < 1 || g > 15 || g % 2 == 0) {
                    return false;  // not the grid this layout encodes
                }
                const int code = (g - 1)/2 + 8*neg;
                wsum += neg ? -g : g;
                // sub-block ib in bytes 16ib..16ib+15: low nibbles weights 0-15, high nibbles 16-31
                y[b].qs[16*ib + w % 16] |= (uint8_t)(code << (4*(w / 16)));
            }
            corr += 128 * ls * wsum;
        }
        y[b].corr = corr;
    }
    return true;
}

// one column, one sub-block: biased activations (unsigned) times signed weights, two chunks
// summed in int16 (each pair <= 2*255*15, two chunks <= 15300) and scaled once
#define PERSEUS_IQ3SP_COL(acc, q) do {                                                                 \
        const __m128i p = _mm_add_epi16(_mm_maddubs_epi16(_mm_loadu_si128((const __m128i *)(q)), w0), \
                                        _mm_maddubs_epi16(_mm_loadu_si128((const __m128i *)(q) + 1), w1)); \
        acc = _mm_add_epi32(acc, _mm_madd_epi16(p, scb));                                              \
    } while (0)

void perseus_iq3s_dot_p(int n, float * s, const void * vx, const void * vy, size_t by, const uint8_t * qb, size_t qbs, int nc) {
    assert(n % QK_K == 0);
    assert(nc >= 1 && nc <= PERSEUS_MC_MAX);
    const block_iq3s_p * x = (const block_iq3s_p *)vx;
    const __m128i lut = _mm_loadu_si128((const __m128i *)perseus_iq3s_levels);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (int c0 = 0; c0 < nc; c0 += 4) {
        const int m = nc - c0 < 4 ? nc - c0 : 4;
        const block_q8_K * yc[4];
        const uint8_t * qc[4];
        for (int k = 0; k < 4; ++k) {
            const int c = c0 + (k < m ? k : 0);
            yc[k] = (const block_q8_K *)((const char *)vy + (size_t)c*by);
            qc[k] = qb + (size_t)c*qbs;
        }
        float sumf[4] = {0};
        for (int i = 0; i < n/QK_K; ++i) {
                const __m128i sc16 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)x[i].sc), _mm_setzero_si128());
            __m128i a0 = _mm_setzero_si128(), a1 = a0, a2 = a0, a3 = a0;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                const __m128i q = _mm_loadu_si128((const __m128i *)(x[i].qs + 16*ib));
                const __m128i w0 = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4));
                const __m128i w1 = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
                const __m128i scb = _mm_shuffle_epi8(sc16, bcast16_mask_ssse3(ib));
                const size_t off = (size_t)QK_K*i + 32*ib;
                PERSEUS_IQ3SP_COL(a0, qc[0] + off);
                if (m > 1) PERSEUS_IQ3SP_COL(a1, qc[1] + off);
                if (m > 2) PERSEUS_IQ3SP_COL(a2, qc[2] + off);
                if (m > 3) PERSEUS_IQ3SP_COL(a3, qc[3] + off);
            }
            const __m128i acc[4] = { a0, a1, a2, a3 };
            for (int k = 0; k < m; ++k) {
                const int isum = hsum_i32_4_ssse3(acc[k]) - x[i].corr;
                sumf[k] += x[i].d * yc[k][i].d * isum;  // as the iq3_s kernel: (d * y.d) * isum
            }
        }
        for (int k = 0; k < m; ++k) {
            s[c0 + k] = sumf[k];
        }
    }
}

// PERSEUS_REPACK=0 keeps IQ3_S in its file layout (A/B runs; llama.cpp's -nr also does); unset or 1 = on
bool perseus_repack_enabled(void) {
    static int state = -1;
    int v = __atomic_load_n(&state, __ATOMIC_RELAXED);
    if (v < 0) {
        const char * e = getenv("PERSEUS_REPACK");
        if (e == NULL || strcmp(e, "1") == 0) {
            v = 1;
        } else if (strcmp(e, "0") == 0) {
            v = 0;
        } else {
            GGML_ABORT("PERSEUS_REPACK must be 0 or 1, got '%s'", e);
        }
        __atomic_store_n(&state, v, __ATOMIC_RELAXED);
    }
    return v;
}

typedef void (*perseus_mc_fn)(int n, float * s, const void * vx, const char * vy, size_t by, int nc);

static perseus_mc_fn perseus_mc_for(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS: return perseus_mc_iq2_xxs;
        case GGML_TYPE_IQ2_XS:  return perseus_mc_iq2_xs;
        case GGML_TYPE_IQ2_S:   return perseus_mc_iq2_s;
        case GGML_TYPE_IQ3_XXS: return perseus_mc_iq3_xxs;
        case GGML_TYPE_IQ3_S:   return perseus_mc_iq3_s;
        case GGML_TYPE_IQ1_S:   return perseus_mc_iq1_s;
        case GGML_TYPE_IQ1_M:   return perseus_mc_iq1_m;
        case GGML_TYPE_IQ4_XS:  return perseus_mc_iq4_xs;
        case GGML_TYPE_Q2_K:    return perseus_mc_q2_K;
        default:                return NULL;
    }
}

// PERSEUS_MC=0 turns the multi-column path off (A/B runs); unset or 1 = on
static bool perseus_mc_enabled(void) {
    static int state = -1;
    int v = __atomic_load_n(&state, __ATOMIC_RELAXED);
    if (v < 0) {
        const char * e = getenv("PERSEUS_MC");
        if (e == NULL || strcmp(e, "1") == 0) {
            v = 1;
        } else if (strcmp(e, "0") == 0) {
            v = 0;
        } else {
            GGML_ABORT("PERSEUS_MC must be 0 or 1, got '%s'", e);
        }
        __atomic_store_n(&state, v, __ATOMIC_RELAXED);
    }
    return v;
}

bool perseus_mc_supported(enum ggml_type type) {
    return perseus_mc_for(type) != NULL && perseus_mc_enabled();
}

void perseus_vec_dot_mc(enum ggml_type type, int n, float * s, const void * vx, const void * vy, size_t by, int nc) {
    assert(n % QK_K == 0);
    assert(nc >= 1 && nc <= PERSEUS_MC_MAX);
    perseus_mc_for(type)(n, s, vx, (const char *)vy, by, nc);
}
#endif
"""

# Declared in ggml-cpu/quants.h, which quants.c, ggml-cpu.c and the self-test all include.
MC_DECL_ANCHOR = "\n#ifdef __cplusplus\n}\n#endif\n"
MC_DECL = r"""
#if defined(__SSSE3__) && !defined(__AVX__)
// perseus multi-column kernels (arch/x86/quants.c): false if the type has none or PERSEUS_MC=0
bool perseus_mc_supported(enum ggml_type type);
// s[c] = row vx . column c, for nc (1..16) q8_K columns at vy + c*by
void perseus_vec_dot_mc(enum ggml_type type, int n, float * s, const void * vx, const void * vy, size_t by, int nc);

// IQ3_S load-time repack (repack.cpp): signed 4-bit codes, see arch/x86/quants.c
typedef struct {
    float    d;        // the block's fp16 d, widened
    int32_t  corr;     // 128 * sum over sub-blocks of scale * sum of weights (activation-bias correction)
    int8_t   sc[8];    // per-32 scale 2s+1
    uint8_t  qs[128];  // sub-block ib: bytes 16ib..16ib+15, low nibble weight j, high nibble weight j+16
} block_iq3s_p;
// false if the grid holds a value the 4-bit codes cannot represent
bool perseus_iq3s_pack(const void * x, void * y, int64_t nblocks);
// s[c] = repacked row vx . column c for nc (1..16): q8_K blocks at vy + c*by (for d), biased
// activations (q8 ^ 0x80) at qb + c*qbs
void perseus_iq3s_dot_p(int n, float * s, const void * vx, const void * vy, size_t by, const uint8_t * qb, size_t qbs, int nc);
bool perseus_repack_enabled(void);
#endif
"""

MC_HOOK_ANCHOR = "    float tmp[32];\n\n    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {\n"
MC_HOOK = r"""    float tmp[32];

#if defined(__SSSE3__) && !defined(__AVX__)
    // perseus: 2+ columns of one 2D weight matrix decode each weight block once for up to 16 columns
    if (num_rows_per_vec_dot == 1 && ir1_end - ir1_start > 1 && ne02 == 1 && ne03 == 1 &&
        src1->type != vec_dot_type && perseus_mc_supported(type)) {
        float tmp_mc[16*16];
        for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
            const int nc = (int) (MIN(iir1 + blck_1, ir1_end) - iir1);
            // src1 was converted into wdata, one row_size row per column, in ir1 order
            const char * src1_cols = (const char *) wdata + iir1 * row_size;
            for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
                const int64_t nr = MIN(iir0 + blck_0, ir0_end) - iir0;
                for (int64_t r = 0; r < nr; ++r) {
                    float out[16];
                    perseus_vec_dot_mc(type, ne00, out, (const char *) src0->data + (iir0 + r) * nb01, src1_cols, row_size, nc);
                    for (int c = 0; c < nc; ++c) {
                        tmp_mc[c*16 + r] = out[c];
                    }
                }
                for (int c = 0; c < nc; ++c) {
                    const int64_t ir1 = iir1 + c;
                    const int64_t i13 = (ir1 / (ne12 * ne1));
                    const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                    const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);
                    float * dst_col = (float *) ((char *) dst->data + (i11 * nb1 + i12 * nb2 + i13 * nb3));
                    memcpy(&dst_col[iir0], &tmp_mc[c*16], nr * sizeof(float));
                }
            }
        }
        return;
    }
#endif

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
"""

# ggml-cpu/repack.cpp: IQ3_S weights go to the CPU_REPACK buffer (llama.cpp's load-time repack
# mechanism; -nr turns it off) as perseus block_iq3s_p, and these traits run their matmuls.
REPACK_TRAITS = r"""#if defined(__SSSE3__) && !defined(__AVX__)
// perseus: IQ3_S repacked at load into signed 4-bit codes (arch/x86/quants.c) and multiplied
// here for every batch size; the buffer holds the larger layout (see get_alloc_size below).
#include "quants.h"

namespace ggml::cpu::repack {
class perseus_iq3s_traits : public tensor_traits_base {
    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_ASSERT(t->type == GGML_TYPE_IQ3_S);
        const int64_t nblocks = ggml_nelements(t) / QK_K;
        GGML_ASSERT(data_size == (size_t) nblocks * sizeof(block_iq3_s));
        return perseus_iq3s_pack(data, t->data, nblocks) ? 0 : -1;
    }

    // wdata: src1 as q8_K rows, then the same activations biased to unsigned bytes (q8 ^ 0x80)
    static size_t q8_bytes(const struct ggml_tensor * src1) {
        return GGML_PAD(ggml_nrows(src1) * ggml_row_size(GGML_TYPE_Q8_K, src1->ne[0]), 64);
    }

    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        size = q8_bytes(op->src[1]) + (size_t) ggml_nrows(op->src[1]) * op->src[1]->ne[0];
        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        const struct ggml_tensor * src0 = op->src[0];
        const struct ggml_tensor * src1 = op->src[1];
        struct ggml_tensor * dst = op;
        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        GGML_ASSERT(ne00 % QK_K == 0 && ne10 == ne00 && ne02 == 1 && ne03 == 1);
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && nb10 == sizeof(float) && nb0 == sizeof(float));

        const int64_t nblk  = ne00 / QK_K;
        const int64_t nr1   = ne11 * ne12 * ne13;
        const size_t  q8row = ggml_row_size(GGML_TYPE_Q8_K, ne10);
        char *    wq8 = (char *) params->wdata;
        uint8_t * wqb = (uint8_t *) wq8 + q8_bytes(src1);
        GGML_ASSERT(params->wsize >= q8_bytes(src1) + (size_t) nr1 * ne10);

        // 1) each thread converts its share of every column's blocks: q8_K, then the biased copy
        const int64_t b0 = ith * nblk / nth, b1 = (ith + 1) * nblk / nth;
        const __m128i bias = _mm_set1_epi8((char) 0x80);
        for (int64_t c = 0; c < nr1 && b1 > b0; ++c) {
            const int64_t i13 = c / (ne11 * ne12), i12 = (c / ne11) % ne12, i11 = c % ne11;
            const float * src = (const float *) ((const char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);
            block_q8_K * q = (block_q8_K *) (wq8 + c*q8row);
            quantize_row_q8_K(src + b0*QK_K, q + b0, (b1 - b0)*QK_K);
            for (int64_t b = b0; b < b1; ++b) {
                for (int j = 0; j < QK_K/16; ++j) {
                    _mm_storeu_si128((__m128i *) (wqb + c*ne10 + b*QK_K + 16*j),
                                     _mm_xor_si128(_mm_loadu_si128((const __m128i *) (q[b].qs + 16*j)), bias));
                }
            }
        }
        ggml_barrier(params->threadpool);

        // 2) a fixed row range per thread (no chunk stealing, so rows stay on the same thread and
        //    NUMA node every token), in 16-row tiles against each tile of up to 16 columns
        const int64_t r0 = ith * ne01 / nth, r1 = (ith + 1) * ne01 / nth;
        const size_t  xrow = nblk * sizeof(block_iq3s_p);
        float out[16];
        for (int64_t ir = r0; ir < r1; ir += 16) {
            const int64_t ir_end = MIN(ir + 16, r1);
            for (int64_t c0 = 0; c0 < nr1; c0 += 16) {
                const int nc = (int) MIN(16, nr1 - c0);
                for (int64_t r = ir; r < ir_end; ++r) {
                    perseus_iq3s_dot_p(ne00, out, (const char *) src0->data + r*xrow, wq8 + c0*q8row, q8row, wqb + c0*ne10, ne10, nc);
                    for (int c = 0; c < nc; ++c) {
                        const int64_t cc = c0 + c;
                        const int64_t i13 = cc / (ne11 * ne12), i12 = (cc / ne11) % ne12, i11 = cc % ne11;
                        *(float *) ((char *) dst->data + i11*nb1 + i12*nb2 + i13*nb3 + r*nb0) = out[c];
                    }
                }
            }
        }
        return true;
    }
};
}  // namespace ggml::cpu::repack
#endif

"""

REPACK_SELECT = """#if defined(__SSSE3__) && !defined(__AVX__)
    // perseus: IQ3_S as signed 4-bit codes (perseus_iq3s_traits above)
    static const ggml::cpu::repack::perseus_iq3s_traits perseus_iq3s;
    if (cur->type == GGML_TYPE_IQ3_S && ggml_n_dims(cur) == 2 && cur->ne[0] % QK_K == 0 && perseus_repack_enabled()) {
        return &perseus_iq3s;
    }
#endif
"""

REPACK_ALLOC_SIZE = """static size_t ggml_backend_cpu_repack_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);
#if defined(__SSSE3__) && !defined(__AVX__)
    // perseus: the repacked IQ3_S layout is larger than the file's
    if (tensor->type == GGML_TYPE_IQ3_S && ggml_repack_get_optimal_repack_type(tensor) != nullptr) {
        return (size_t) (ggml_nelements(tensor) / QK_K) * sizeof(block_iq3s_p);
    }
#endif
    return ggml_nbytes(tensor);
}

"""

Q8_0_ANCHOR = "    sumf = hsum_float_8(accum);\n#endif\n"
Q8_0_KERNEL = r"""    sumf = hsum_float_8(accum);
#elif defined(__SSSE3__)
    for (; ib < nb; ++ib) {
        const __m128i qx_0 = _mm_loadu_si128((const __m128i *)x[ib].qs);
        const __m128i qx_1 = _mm_loadu_si128((const __m128i *)x[ib].qs + 1);
        const __m128i qy_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        const __m128i qy_1 = _mm_loadu_si128((const __m128i *)y[ib].qs + 1);
        const int sumi = hsum_i32_4_ssse3(_mm_add_epi32(mul_sum_i8_pairs(qx_0, qy_0), mul_sum_i8_pairs(qx_1, qy_1)));
        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }
#endif
"""


def iq1s_u8_table(root):
    """8*(grid+1) per byte of llama.cpp's iq1s_grid (grid bytes are -1, 0, 1)."""
    import re
    common = open(os.path.join(root, "ggml", "src", "ggml-common.h")).read()
    a = common.index("GGML_TABLE_BEGIN(uint64_t, iq1s_grid, NGRID_IQ1S)")
    body = common[a:common.index("GGML_TABLE_END()", a)].split("\n", 1)[1]
    grid = [int(v, 16) for v in re.findall(r"0x[0-9a-f]{16}", body)]
    assert len(grid) == 2048, len(grid)
    out = []
    for g in grid:
        v = 0
        for j in range(8):
            b = (g >> (8*j)) & 0xff
            b = b - 256 if b >= 128 else b
            assert b in (-1, 0, 1)
            v |= (8*(b + 1)) << (8*j)
        out.append(f"0x{v:016x}")
    rows = [", ".join(out[i:i+4]) for i in range(0, len(out), 4)]
    return ("#if !defined(__AVX__) && defined(__SSSE3__)\n"
            "// 8*(grid+1) for each byte of iq1s_grid (generated by kernels/apply_ssse3.py)\n"
            "static const uint64_t perseus_iq1s_u8[2048] = {\n    " + ",\n    ".join(rows) + "\n};\n#endif\n\n")


def function_span(src, name):
    head = f"void ggml_vec_dot_{name}("
    start = src.index(head)
    end = src.index("\n}\n", start) + 3
    return start, end


def patch_mul_mat(cpu):
    path = os.path.join(cpu, "quants.h")
    src = open(path).read()
    assert src.endswith(MC_DECL_ANCHOR), "quants.h ending changed upstream"
    open(path, "w").write(src[:-len(MC_DECL_ANCHOR)] + MC_DECL + MC_DECL_ANCHOR)

    path = os.path.join(cpu, "ggml-cpu.c")
    src = open(path).read()
    assert src.count(MC_HOOK_ANCHOR) == 1, "mul_mat chunk loop changed upstream"
    open(path, "w").write(src.replace(MC_HOOK_ANCHOR, MC_HOOK))


def patch_repack(cpu):
    path = os.path.join(cpu, "repack.cpp")
    src = open(path).read()
    fn = "static const ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(const struct ggml_tensor * cur) {\n"
    getter = "ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void) {\n"
    slot = "                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes\n"
    assert src.count(fn) == 1 and src.count(getter) == 1 and src.count(slot) == 1, "repack.cpp changed upstream"
    src = src.replace(fn, REPACK_TRAITS + fn + REPACK_SELECT)
    src = src.replace(getter, REPACK_ALLOC_SIZE + getter)
    src = src.replace(slot, "                           /* .get_alloc_size       = */ ggml_backend_cpu_repack_buffer_type_get_alloc_size,\n")
    open(path, "w").write(src)


def main():
    cpu = os.path.join(sys.argv[1], "ggml", "src", "ggml-cpu")
    path = os.path.join(cpu, "arch", "x86", "quants.c")
    src = open(path).read()
    if MARK in src:
        sys.exit(f"{path} already has the perseus SSSE3 kernels")

    # the sign lookup table the IQ2/IQ3 kernels use is only compiled for AVX builds
    guard = "#if defined (__AVX__) || defined (__AVX2__)\nstatic const int8_t keven_signs_q2xs[1024]"
    assert src.count(guard) == 1, "keven_signs_q2xs guard changed upstream"
    src = src.replace(guard, "#if defined (__AVX__) || defined (__AVX2__) || defined(__SSSE3__)\nstatic const int8_t keven_signs_q2xs[1024]")

    anchor = "void ggml_vec_dot_q1_0_q8_0("
    assert src.count(anchor) == 1
    src = src.replace(anchor, HELPERS + iq1s_u8_table(sys.argv[1]) + anchor)

    for name, body in KERNELS.items():
        start, end = function_span(src, name)
        fn = src[start:end]
        avx = max(fn.find("#elif defined(__AVX__)"), fn.find("#elif defined __AVX__"))
        assert avx > 0, name
        els = fn.index("\n#else\n", avx) + 1
        fn = fn[:els] + body + fn[els:]
        src = src[:start] + fn + src[end:]

    start, end = function_span(src, "q8_0_q8_0")
    fn = src[start:end]
    assert fn.count(Q8_0_ANCHOR) == 1
    src = src[:start] + fn.replace(Q8_0_ANCHOR, Q8_0_KERNEL) + src[end:]

    open(path, "w").write(src + MC)
    patch_mul_mat(cpu)
    patch_repack(cpu)
    print(f"added SSSE3 kernels: q8_0 + {', '.join(KERNELS)}; multi-column kernels + mul_mat hook")


if __name__ == "__main__":
    main()
