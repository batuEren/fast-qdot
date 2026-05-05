#include <gtest/gtest.h>
#include "src/naive.h"
#include "src/mad_avx2.h"

static uint8_t pack4(int w0, int w1, int w2, int w3) {
    return (uint8_t)(((w0 + 1) << 6) | ((w1 + 1) << 4) | ((w2 + 1) << 2) | (w3 + 1));
}

// mad_ternary_dot_avx2 (n must be a multiple of 32)

TEST(MadTernaryDotAVX2, AllZeroWeights) {
    uint8_t w[8];
    for (auto& b : w) b = pack4(0,0,0,0);
    int8_t a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)(i + 1);
    EXPECT_EQ(mad_ternary_dot_avx2(w, a, 32), 0);
}

TEST(MadTernaryDotAVX2, AllPositiveWeights) {
    // all +1 weights, activations 1..32 => sum = 32*33/2 = 528
    uint8_t w[8];
    for (auto& b : w) b = pack4(1,1,1,1);
    int8_t a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)(i + 1);
    EXPECT_EQ(mad_ternary_dot_avx2(w, a, 32), 528);
}

TEST(MadTernaryDotAVX2, AllNegativeWeights) {
    // all -1 weights, activations 1..32 => sum = -528
    uint8_t w[8];
    for (auto& b : w) b = pack4(-1,-1,-1,-1);
    int8_t a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)(i + 1);
    EXPECT_EQ(mad_ternary_dot_avx2(w, a, 32), -528);
}

TEST(MadTernaryDotAVX2, AllZeroActivations) {
    uint8_t w[8];
    for (auto& b : w) b = pack4(1,-1,0,1);
    int8_t a[32] = {};
    EXPECT_EQ(mad_ternary_dot_avx2(w, a, 32), 0);
}

TEST(MadTernaryDotAVX2, MixedWeightsKnownResult) {
    // weights: +1,0,-1,0 repeating, activations 1..32
    // +1 at positions 0,4,8,...,28 (values 1,5,9,...,29) sum=120
    // -1 at positions 2,6,10,...,30 (values 3,7,11,...,31) sum=136
    // expected: 120 - 136 = -16
    uint8_t w[8];
    for (auto& b : w) b = pack4(1,0,-1,0);
    int8_t a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)(i + 1);
    EXPECT_EQ(mad_ternary_dot_avx2(w, a, 32), -16);
}

TEST(MadTernaryDotAVX2, MatchesNaive_Basic) {
    uint8_t w[8] = {
        pack4(-1,0,1,-1), pack4(0,1,-1,0), pack4(1,1,-1,0), pack4(-1,-1,1,1),
        pack4(0,-1,0,1),  pack4(1,0,1,-1), pack4(-1,1,0,0), pack4(0,-1,1,1)
    };
    int8_t a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)((i % 13) - 6);
    EXPECT_EQ(mad_ternary_dot_avx2(w, a, 32), naive_ternary_dot(w, a, 32));
}

TEST(MadTernaryDotAVX2, MatchesNaive_NegativeActivations) {
    uint8_t w[8];
    for (int i = 0; i < 8; i++) w[i] = pack4((i%3)-1, ((i+1)%3)-1, ((i+2)%3)-1, ((i+3)%3)-1);
    int8_t a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)(-(i % 100 + 1));
    EXPECT_EQ(mad_ternary_dot_avx2(w, a, 32), naive_ternary_dot(w, a, 32));
}

TEST(MadTernaryDotAVX2, MatchesNaive_Large) {
    // n=1024 exercises the acc16→acc32 flush path (at i=63, 127, ...)
    const int N = 1024;
    uint8_t w[N / 4];
    int8_t  a[N];
    for (int i = 0; i < N / 4; i++)
        w[i] = pack4((i%3)-1, ((i+1)%3)-1, ((i+2)%3)-1, ((i+3)%3)-1);
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    EXPECT_EQ(mad_ternary_dot_avx2(w, a, N), naive_ternary_dot(w, a, N));
}

// mad_binary_dot_avx2 (n must be a multiple of 32)

TEST(MadBinaryDotAVX2, AllPositiveWeights) {
    // 0xFF => all bits set => all +1; activations 1..32 => sum = 528
    uint8_t w[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    int8_t  a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)(i + 1);
    EXPECT_EQ(mad_binary_dot_avx2(w, a, 32), 528);
}

TEST(MadBinaryDotAVX2, AllNegativeWeights) {
    // 0x00 => all bits clear => all -1; activations 1..32 => sum = -528
    uint8_t w[4] = { 0x00, 0x00, 0x00, 0x00 };
    int8_t  a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)(i + 1);
    EXPECT_EQ(mad_binary_dot_avx2(w, a, 32), -528);
}

TEST(MadBinaryDotAVX2, AlternatingWeights) {
    // 0xAA = 10101010 => [+1,-1,+1,-1,...] per byte
    // 4 bytes, activations 1..32: each byte contributes -4 => total -16
    uint8_t w[4] = { 0xAA, 0xAA, 0xAA, 0xAA };
    int8_t  a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)(i + 1);
    EXPECT_EQ(mad_binary_dot_avx2(w, a, 32), -16);
}

TEST(MadBinaryDotAVX2, AllZeroActivations) {
    uint8_t w[4] = { 0xFF, 0xAA, 0x55, 0x00 };
    int8_t  a[32] = {};
    EXPECT_EQ(mad_binary_dot_avx2(w, a, 32), 0);
}

TEST(MadBinaryDotAVX2, MixedWeightsKnownResult) {
    // 0xF0 = 11110000 => +1,+1,+1,+1,-1,-1,-1,-1 per byte
    // 4 bytes, activations 1..32: each byte contributes -16 => total -64
    uint8_t w[4] = { 0xF0, 0xF0, 0xF0, 0xF0 };
    int8_t  a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)(i + 1);
    EXPECT_EQ(mad_binary_dot_avx2(w, a, 32), -64);
}

TEST(MadBinaryDotAVX2, MatchesNaive_Basic) {
    uint8_t w[4] = { 0xAA, 0x55, 0xF0, 0x0F };
    int8_t  a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)((i % 13) - 6);
    EXPECT_EQ(mad_binary_dot_avx2(w, a, 32), naive_binary_dot(w, a, 32));
}

TEST(MadBinaryDotAVX2, MatchesNaive_NegativeActivations) {
    uint8_t w[4] = { 0xFF, 0x00, 0xAA, 0x55 };
    int8_t  a[32];
    for (int i = 0; i < 32; i++) a[i] = (int8_t)(-(i % 100 + 1));
    EXPECT_EQ(mad_binary_dot_avx2(w, a, 32), naive_binary_dot(w, a, 32));
}

TEST(MadBinaryDotAVX2, MatchesNaive_Large) {
    // n=1024 exercises the acc16→acc32 flush path
    const int N = 1024;
    uint8_t w[N / 8];
    int8_t  a[N];
    for (int i = 0; i < N / 8; i++) w[i] = (uint8_t)(i * 37 + 13);
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    EXPECT_EQ(mad_binary_dot_avx2(w, a, N), naive_binary_dot(w, a, N));
}
