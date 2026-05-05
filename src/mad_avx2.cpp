#include "mad_avx2.h"
#include <immintrin.h>
#include <cassert>

// Reduce 8 int32 lanes of a 256-bit register to a scalar.
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    const __m128i hi64   = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64  = _mm_add_epi32(hi64, sum128);
    const __m128i hi32   = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// Ternary MAD (AVX2)
// Same 2-bit packing as mad.cpp: 4 weights per byte, bits[7:6]=w0 .. bits[1:0]=w3.
// Each iteration: 8 packed bytes -> 32 codes -> 32 activations -> accumulate.
// SPREAD broadcasts each input byte to 4 consecutive lanes (shuffle is per 128-bit
// half; both halves see the same 8 bytes via broadcastsi128).
// n must be a multiple of 32.

int32_t mad_ternary_dot_avx2(const uint8_t* weights, const int8_t* activations, int n) {
    assert(n % 32 == 0);

    // _mm256_set_epi8: arg 0 = element 31 (high), arg 31 = element 0 (low)
    // Each of the 8 packed bytes spreads to 4 consecutive positions.
    // Shuffle indices reference within each 128-bit half independently.
    const __m256i SPREAD = _mm256_set_epi8(
        7,7,7,7, 6,6,6,6, 5,5,5,5, 4,4,4,4,   // high half: bytes 4-7 → pos 16-31
        3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0    // low half:  bytes 0-3 → pos  0-15
    );

    // 0xFF at positions where element_index % 4 == k
    // _mm256_set1_epi32(0x000000FF) in little-endian bytes: [FF,00,00,00] per word
    const __m256i PM0 = _mm256_set1_epi32(0x000000FF);
    const __m256i PM1 = _mm256_set1_epi32(0x0000FF00);
    const __m256i PM2 = _mm256_set1_epi32(0x00FF0000);
    const __m256i PM3 = _mm256_set1_epi32((int)0xFF000000);

    const __m256i mask03 = _mm256_set1_epi8(0x03);
    const __m256i minus1 = _mm256_set1_epi8(-1);
    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i one16  = _mm256_set1_epi16(1);

    __m256i acc32 = _mm256_setzero_si256();
    __m256i acc16 = _mm256_setzero_si256();

    for (int i = 0; i < n / 32; i++) {
        // Load 8 packed bytes (32 ternary codes) into low 64 bits, broadcast to both halves.
        __m128i lo8    = _mm_loadl_epi64((const __m128i*)(weights + i * 8));
        __m256i packed = _mm256_broadcastsi128_si256(lo8);

        // Spread: each byte → 4 consecutive positions in the output.
        __m256i S = _mm256_shuffle_epi8(packed, SPREAD);

        // Extract the 4 code sub-groups via shift + mask.
        // Since each group-of-4 holds the same byte, cross-byte contamination
        // from srli_epi16 falls above bit 1 and is removed by mask03.
        __m256i s6 = _mm256_and_si256(_mm256_srli_epi16(S, 6), mask03); // bits[7:6] = w0 group
        __m256i s4 = _mm256_and_si256(_mm256_srli_epi16(S, 4), mask03); // bits[5:4] = w1 group
        __m256i s2 = _mm256_and_si256(_mm256_srli_epi16(S, 2), mask03); // bits[3:2] = w2 group
        __m256i s0 = _mm256_and_si256(S,                        mask03); // bits[1:0] = w3 group

        // Select per output position: position j gets shift for bit-pair (j%4).
        __m256i codes = _mm256_or_si256(
            _mm256_or_si256(_mm256_and_si256(s6, PM0), _mm256_and_si256(s4, PM1)),
            _mm256_or_si256(_mm256_and_si256(s2, PM2), _mm256_and_si256(s0, PM3))
        );

        // Map codes {0,1,2} → signs {-1,0,+1}
        __m256i signs = _mm256_add_epi8(codes, minus1);

        __m256i acts = _mm256_loadu_si256((const __m256i*)(activations + i * 32));
        __m256i r    = _mm256_sign_epi8(acts, signs);

        // ones8 (uint8=1) × r (int8) summed pairwise → int16; max ±254 per lane.
        acc16 = _mm256_add_epi16(acc16, _mm256_maddubs_epi16(ones8, r));

        // Widen to int32 every 64 iterations: max int16 = 64×254 = 16256 < 32767
        if ((i & 63) == 63) {
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(acc16, one16));
            acc16 = _mm256_setzero_si256();
        }
    }

    acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(acc16, one16));
    return hsum_i32_8(acc32);
}

// Binary MAD (AVX2)
// Same 1-bit packing as mad.cpp: 8 weights per byte, MSB = weight 0.
// Each iteration: 4 packed bytes -> 32 sign bytes -> 32 activations -> accumulate.
// BSPREAD broadcasts each byte to 8 consecutive lanes; BIT_MASK isolates one bit
// per lane; min_epu8 normalises to {0,1}; sign = 2*is_set - 1 gives {+1,-1}.
// n must be a multiple of 32.

int32_t mad_binary_dot_avx2(const uint8_t* weights, const int8_t* activations, int n) {
    assert(n % 32 == 0);

    // Each of the 4 packed bytes spreads to 8 consecutive positions.
    // Low half: bytes 0-1 → positions 0-15 (8 each)
    // High half: bytes 2-3 → positions 16-31 (8 each, indices within high half)
    const __m256i BSPREAD = _mm256_set_epi8(
        3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2,   // high half: bytes 2-3
        1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0    // low half:  bytes 0-1
    );

    // Bit masks repeated for every 8-element group: MSB (bit 7) at position 0.
    const __m128i bm128    = _mm_set_epi8(1,2,4,8,16,32,64,-128, 1,2,4,8,16,32,64,-128);
    const __m256i BIT_MASK = _mm256_broadcastsi128_si256(bm128);

    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i one16 = _mm256_set1_epi16(1);

    __m256i acc32 = _mm256_setzero_si256();
    __m256i acc16 = _mm256_setzero_si256();

    for (int i = 0; i < n / 32; i++) {
        // Load 4 packed bytes (32 binary weight bits), broadcast to both 128-bit halves.
        __m128i w4   = _mm_cvtsi32_si128(*(const int*)(weights + i * 4));
        __m256i w256 = _mm256_broadcastsi128_si256(w4);

        // Spread: byte k → 8 consecutive positions
        __m256i w_spread = _mm256_shuffle_epi8(w256, BSPREAD);

        // Isolate one bit per lane, normalise to {0,1}
        __m256i bits   = _mm256_and_si256(w_spread, BIT_MASK);
        __m256i is_set = _mm256_min_epu8(bits, ones8); // 0 or 1

        // sign: 0x01 (+1) where bit=1, 0xFF (-1) where bit=0
        __m256i sign = _mm256_sub_epi8(_mm256_add_epi8(is_set, is_set), ones8);

        __m256i acts = _mm256_loadu_si256((const __m256i*)(activations + i * 32));
        __m256i r    = _mm256_sign_epi8(acts, sign);

        acc16 = _mm256_add_epi16(acc16, _mm256_maddubs_epi16(ones8, r));

        if ((i & 63) == 63) {
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(acc16, one16));
            acc16 = _mm256_setzero_si256();
        }
    }

    acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(acc16, one16));
    return hsum_i32_8(acc32);
}
