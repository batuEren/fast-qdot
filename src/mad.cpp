 #include "mad.h"
#include <immintrin.h>
#include <cassert>

// Uses SSSE3 128-bit ops with AVX-style structure (mirrors mad_avx2.cpp at half width).
// All processors supporting AVX also support SSE4.1 and SSSE3.

static inline int32_t hsum_epi32(__m128i v) {
    __m128i hi64  = _mm_unpackhi_epi64(v, v);
    __m128i sum64 = _mm_add_epi32(v, hi64);
    __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// Ternary MAD (SSE4/SSSE3)
// Same 2-bit packing as naive: 4 weights per byte, bits[7:6]=w0 .. bits[1:0]=w3.
// Each iteration: 4 packed bytes -> 16 codes -> 16 activations -> accumulate.
// SPREAD broadcasts each input byte to 4 consecutive output lanes; shifted/masked
// extractions give the 4 code groups; position masks select the right group per lane.
// n must be a multiple of 16.

int32_t mad_ternary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    assert(n % 16 == 0);

    // Each of the 4 packed bytes spreads to 4 consecutive output positions.
    // _mm_set_epi8: arg 0 = element 15 (high), arg 15 = element 0 (low).
    const __m128i SPREAD = _mm_set_epi8(3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0);

    // 0xFF at positions where element_index % 4 == k (one per 32-bit group, little-endian).
    const __m128i PM0 = _mm_set1_epi32(0x000000FF);
    const __m128i PM1 = _mm_set1_epi32(0x0000FF00);
    const __m128i PM2 = _mm_set1_epi32(0x00FF0000);
    const __m128i PM3 = _mm_set1_epi32((int)0xFF000000);

    const __m128i mask03 = _mm_set1_epi8(0x03);
    const __m128i minus1 = _mm_set1_epi8(-1);
    const __m128i ones8  = _mm_set1_epi8(1);
    const __m128i one16  = _mm_set1_epi16(1);

    __m128i acc32 = _mm_setzero_si128();
    __m128i acc16 = _mm_setzero_si128();

    for (int i = 0; i < n / 16; i++) {
        // Load 4 packed bytes (16 ternary codes) into low 32 bits.
        __m128i packed = _mm_cvtsi32_si128(*(const int*)(weights + i * 4));

        // Spread: each byte → 4 consecutive positions.
        __m128i S = _mm_shuffle_epi8(packed, SPREAD);

        // Extract the 4 code sub-groups via shift + mask.
        // Cross-byte contamination from srli_epi16 falls above bit 1 and is masked out.
        __m128i s6 = _mm_and_si128(_mm_srli_epi16(S, 6), mask03); // bits[7:6] = w0 group
        __m128i s4 = _mm_and_si128(_mm_srli_epi16(S, 4), mask03); // bits[5:4] = w1 group
        __m128i s2 = _mm_and_si128(_mm_srli_epi16(S, 2), mask03); // bits[3:2] = w2 group
        __m128i s0 = _mm_and_si128(S,                    mask03); // bits[1:0] = w3 group

        // Select per output position: position j gets shift for bit-pair (j%4).
        __m128i codes = _mm_or_si128(
            _mm_or_si128(_mm_and_si128(s6, PM0), _mm_and_si128(s4, PM1)),
            _mm_or_si128(_mm_and_si128(s2, PM2), _mm_and_si128(s0, PM3))
        );

        // Map codes {0,1,2} → signs {-1,0,+1}.
        __m128i signs = _mm_add_epi8(codes, minus1);

        __m128i acts = _mm_loadu_si128((const __m128i*)(activations + i * 16));
        __m128i r    = _mm_sign_epi8(acts, signs);

        // ones8 (uint8=1) × r (int8) summed pairwise → int16; max ±254 per lane.
        acc16 = _mm_add_epi16(acc16, _mm_maddubs_epi16(ones8, r));

        // Widen to int32 every 64 iterations: max int16 = 64×254 = 16256 < 32767.
        if ((i & 63) == 63) {
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(acc16, one16));
            acc16 = _mm_setzero_si128();
        }
    }

    acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(acc16, one16));
    return hsum_epi32(acc32);
}

// Binary MAD (SSE4/SSSE3)
// Same 1-bit packing as naive: 8 weights per byte, MSB = weight 0.
// Each iteration: 2 packed bytes -> 16 sign bytes -> 16 activations -> accumulate.
// BSPREAD broadcasts each byte to 8 consecutive lanes; BIT_MASK isolates one bit
// per lane; min_epu8 normalises to {0,1}; sign = 2*is_set - 1 gives {+1,-1}.
// n must be a multiple of 16.

int32_t mad_binary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    assert(n % 16 == 0);

    // Each of the 2 packed bytes spreads to 8 consecutive output positions.
    // Low byte → positions 0-7, high byte → positions 8-15.
    const __m128i BSPREAD = _mm_set_epi8(1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);

    // Bit masks: MSB (bit 7) at position 0, repeated for both 8-element groups.
    const __m128i BIT_MASK = _mm_set_epi8(1,2,4,8,16,32,64,-128, 1,2,4,8,16,32,64,-128);

    const __m128i ones8 = _mm_set1_epi8(1);
    const __m128i one16 = _mm_set1_epi16(1);

    __m128i acc32 = _mm_setzero_si128();
    __m128i acc16 = _mm_setzero_si128();

    for (int i = 0; i < n / 16; i++) {
        // Load 2 packed bytes (16 binary weight bits).
        __m128i w2 = _mm_cvtsi32_si128(*(const uint16_t*)(weights + i * 2));

        // Spread: byte k → 8 consecutive positions.
        __m128i w_spread = _mm_shuffle_epi8(w2, BSPREAD);

        // Isolate one bit per lane, normalise to {0,1}.
        __m128i bits   = _mm_and_si128(w_spread, BIT_MASK);
        __m128i is_set = _mm_min_epu8(bits, ones8);

        // sign: 0x01 (+1) where bit=1, 0xFF (-1) where bit=0.
        __m128i sign = _mm_sub_epi8(_mm_add_epi8(is_set, is_set), ones8);

        __m128i acts = _mm_loadu_si128((const __m128i*)(activations + i * 16));
        __m128i r    = _mm_sign_epi8(acts, sign);

        acc16 = _mm_add_epi16(acc16, _mm_maddubs_epi16(ones8, r));

        if ((i & 63) == 63) {
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(acc16, one16));
            acc16 = _mm_setzero_si128();
        }
    }

    acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(acc16, one16));
    return hsum_epi32(acc32);
}
