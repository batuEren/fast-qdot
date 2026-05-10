#include "lut.h"
#include <immintrin.h>
#include <cassert>


static inline int32_t hsum_epi32(__m128i v) {
    __m128i hi64  = _mm_unpackhi_epi64(v, v);
    __m128i sum64 = _mm_add_epi32(v, hi64);
    __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

int32_t lut_ternary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    assert(n % 16 == 0);

    const __m128i EVEN = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1, 14,12,10,8,6,4,2,0);
    const __m128i ODD  = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1, 15,13,11,9,7,5,3,1);

    const __m128i one16 = _mm_set1_epi16(1);
    __m128i acc32 = _mm_setzero_si128();
    __m128i acc16 = _mm_setzero_si128();

    for (int i = 0; i < n / 16; i++) {
        __m128i acts = _mm_loadu_si128((const __m128i*)(activations + i * 16));
        __m128i a_e  = _mm_cvtepi8_epi16(_mm_shuffle_epi8(acts, EVEN)); 
        __m128i a_o  = _mm_cvtepi8_epi16(_mm_shuffle_epi8(acts, ODD));  

        //   idx 0: (-1,-1) → -a_e - a_o
        //   idx 1: (-1, 0) → -a_e
        //   idx 2: (-1,+1) → -a_e + a_o
        //   idx 3: ( 0,-1) → -a_o
        //   idx 4: ( 0, 0) →  0
        //   idx 5: ( 0,+1) →  a_o
        //   idx 6: (+1,-1) →  a_e - a_o
        __m128i zero  = _mm_setzero_si128();
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

        const uint8_t* wb = weights + i * 4;
        __m128i idx = _mm_set_epi16(
             wb[3] & 0xf, (wb[3] >> 4) & 0xf,  // pairs 7, 6
             wb[2] & 0xf, (wb[2] >> 4) & 0xf,  // pairs 5, 4
             wb[1] & 0xf, (wb[1] >> 4) & 0xf,  // pairs 3, 2
             wb[0] & 0xf, (wb[0] >> 4) & 0xf   // pairs 1, 0
        );

        __m128i result = zero;
        for (int c = 0; c < 9; c++) {
            __m128i mask = _mm_cmpeq_epi16(idx, _mm_set1_epi16(c));
            result = _mm_or_si128(result, _mm_and_si128(mask, lut[c]));
        }

        acc16 = _mm_add_epi16(acc16, result);

        if ((i & 63) == 63) {
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(acc16, one16));
            acc16 = _mm_setzero_si128();
        }
    }

    acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(acc16, one16));
    return hsum_epi32(acc32);
}


