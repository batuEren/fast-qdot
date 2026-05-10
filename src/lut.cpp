#include "lut.h"
#include <immintrin.h>
#include <cassert>

// LUT-based ternary dot product (SSSE3 + SSE4.1).
//
// Weight packing (different from mad.cpp's 2-bit format):
//   Each byte holds two 4-bit pair-indices, high nibble first.
//   pair_index = 3*(c_even + 1) + (c_odd + 1), c ∈ {-1, 0, +1} → index ∈ {0..8}.
//   Both mad.cpp and lut.cpp consume n/4 weight bytes for n elements.
//
// Each outer iteration processes 16 activations = 8 pairs.
// For each pair k, lut[c][k] = c_even*a_e[k] + c_odd*a_o[k] is precomputed
// for all 9 index values c. Selection via 9 SIMD compare-and-mask iterations
// gives the 8 contributions simultaneously without any scalar branching.

static inline int32_t hsum_epi32(__m128i v) {
    __m128i hi64 = _mm_unpackhi_epi64(v, v);
    __m128i sum64 = _mm_add_epi32(v, hi64);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// n must be a multiple of 16.
int32_t lut_ternary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    assert(n % 16 == 0);

    // Shuffle masks: gather even/odd-indexed bytes into positions 0-7 of a 128-bit register.
    // Positions 8-15 use index 0x80 (-1) which _mm_shuffle_epi8 maps to zero.
    const __m128i EVEN = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m128i ODD = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1);

    const __m128i one16 = _mm_set1_epi16(1);
    __m128i acc32 = _mm_setzero_si128();
    __m128i acc16 = _mm_setzero_si128();

    for (int i = 0; i < n / 16; i++) {
        // Load 16 activations and deinterleave into 8 even and 8 odd int16 lanes.
        __m128i acts = _mm_loadu_si128((const __m128i*)(activations + i * 16));
        __m128i a_e = _mm_cvtepi8_epi16(_mm_shuffle_epi8(acts, EVEN)); // a[0],a[2],...,a[14]
        __m128i a_o = _mm_cvtepi8_epi16(_mm_shuffle_epi8(acts, ODD));  // a[1],a[3],...,a[15]

        // Build 9 LUT columns. lut[c] is an 8-lane int16 vector; lut[c][k] is the
        // contribution of pair k when its pair_index equals c:
        //   idx 0: (-1,-1) → -a_e - a_o
        //   idx 1: (-1, 0) → -a_e
        //   idx 2: (-1,+1) → -a_e + a_o
        //   idx 3: ( 0,-1) → -a_o
        //   idx 4: ( 0, 0) →  0
        //   idx 5: ( 0,+1) →  a_o
        //   idx 6: (+1,-1) →  a_e - a_o
        //   idx 7: (+1, 0) →  a_e
        //   idx 8: (+1,+1) →  a_e + a_o
        __m128i zero = _mm_setzero_si128();
        __m128i neg_e = _mm_sub_epi16(zero, a_e);
        __m128i neg_o = _mm_sub_epi16(zero, a_o);
        const __m128i lut[9] = {
            _mm_add_epi16(neg_e, neg_o),
            neg_e,
            _mm_add_epi16(neg_e, a_o),
            neg_o,
            zero,
            a_o,
            _mm_sub_epi16(a_e, a_o),
            a_e,
            _mm_add_epi16(a_e, a_o),
        };

        // Unpack 4 nibble-packed weight bytes into 8 pair-indices (one per pair, as int16).
        // High nibble of byte j → pair 2j, low nibble → pair 2j+1.
        const uint8_t* wb = weights + i * 4;
        __m128i idx = _mm_set_epi16(
            wb[3] & 0xf, (wb[3] >> 4) & 0xf,  // pairs 7, 6
            wb[2] & 0xf, (wb[2] >> 4) & 0xf,  // pairs 5, 4
            wb[1] & 0xf, (wb[1] >> 4) & 0xf,  // pairs 3, 2
            wb[0] & 0xf, (wb[0] >> 4) & 0xf   // pairs 1, 0
        );

        // Select: result[k] = lut[idx[k]][k].
        // For each possible index value c, mask lanes where idx[k]==c and OR in lut[c][k].
        __m128i result = zero;
        for (int c = 0; c < 9; c++) {
            __m128i mask = _mm_cmpeq_epi16(idx, _mm_set1_epi16(c));
            result = _mm_or_si128(result, _mm_and_si128(mask, lut[c]));
        }

        // result holds 8 int16 pair contributions; accumulate horizontally.
        acc16 = _mm_add_epi16(acc16, result);

        // Flush to int32 every 64 iterations: max acc16 per lane = 64 × 254 = 16256 < 32767.
        if ((i & 63) == 63) {
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(acc16, one16));
            acc16 = _mm_setzero_si128();
        }
    }

    acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(acc16, one16));
    return hsum_epi32(acc32);
}

int32_t lut_binary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    return 0;
}
