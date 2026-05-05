#include <gtest/gtest.h>
#include "src/naive.h"
#include "src/mad.h"

static uint8_t pack4(int w0, int w1, int w2, int w3) {
    return (uint8_t)(((w0 + 1) << 6) | ((w1 + 1) << 4) | ((w2 + 1) << 2) | (w3 + 1));
}

// mad_ternary_dot (n must be a multiple of 16)

TEST(MadTernaryDot, AllZeroWeights) {
    uint8_t w[4] = { pack4(0,0,0,0), pack4(0,0,0,0), pack4(0,0,0,0), pack4(0,0,0,0) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(mad_ternary_dot(w, a, 16), 0);
}

TEST(MadTernaryDot, AllPositiveWeights) {
    // all +1 weights, activations 1..16 => sum = 136
    uint8_t w[4] = { pack4(1,1,1,1), pack4(1,1,1,1), pack4(1,1,1,1), pack4(1,1,1,1) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(mad_ternary_dot(w, a, 16), 136);
}

TEST(MadTernaryDot, AllNegativeWeights) {
    // all -1 weights, activations 1..16 => sum = -136
    uint8_t w[4] = { pack4(-1,-1,-1,-1), pack4(-1,-1,-1,-1), pack4(-1,-1,-1,-1), pack4(-1,-1,-1,-1) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(mad_ternary_dot(w, a, 16), -136);
}

TEST(MadTernaryDot, AllZeroActivations) {
    uint8_t w[4] = { pack4(1,-1,0,1), pack4(-1,1,0,-1), pack4(1,1,-1,0), pack4(-1,-1,1,1) };
    int8_t  a[16] = {};
    EXPECT_EQ(mad_ternary_dot(w, a, 16), 0);
}

TEST(MadTernaryDot, MixedWeightsKnownResult) {
    // weights: [-1,0,+1,-1, 0,+1,-1,0, +1,+1,-1,0, -1,-1,+1,+1]
    // activations: [1..16]
    // sum = (-1+0+3-4) + (0+6-7+0) + (9+10-11+0) + (-13-14+15+16) = -2-1+8+4 = 9
    uint8_t w[4] = { pack4(-1,0,1,-1), pack4(0,1,-1,0), pack4(1,1,-1,0), pack4(-1,-1,1,1) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(mad_ternary_dot(w, a, 16), 9);
}

TEST(MadTernaryDot, MatchesNaive_Basic) {
    uint8_t w[4] = { pack4(-1,0,1,-1), pack4(0,1,-1,0), pack4(1,1,-1,0), pack4(-1,-1,1,1) };
    int8_t  a[16] = { 10,-5,3,7,-2,8,-4,6,1,-3,5,-7,2,9,-6,4 };
    EXPECT_EQ(mad_ternary_dot(w, a, 16), naive_ternary_dot(w, a, 16));
}

TEST(MadTernaryDot, MatchesNaive_NegativeActivations) {
    uint8_t w[4] = { pack4(1,1,1,1), pack4(-1,-1,-1,-1), pack4(1,-1,0,0), pack4(0,0,1,-1) };
    int8_t  a[16] = { -1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16 };
    EXPECT_EQ(mad_ternary_dot(w, a, 16), naive_ternary_dot(w, a, 16));
}

TEST(MadTernaryDot, MatchesNaive_Large) {
    // n=1024 exercises the acc16→acc32 flush path (at i=63, 127, ...)
    const int N = 1024;
    uint8_t w[N / 4];
    int8_t  a[N];
    for (int i = 0; i < N / 4; i++) {
        w[i] = pack4((i % 3) - 1, ((i + 1) % 3) - 1,
                     ((i + 2) % 3) - 1, ((i + 3) % 3) - 1);
    }
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    EXPECT_EQ(mad_ternary_dot(w, a, N), naive_ternary_dot(w, a, N));
}

// mad_binary_dot (n must be a multiple of 16)

TEST(MadBinaryDot, AllPositiveWeights) {
    // 0xFF = 11111111 => all +1; second byte/activations are zero and don't contribute.
    uint8_t w[] = { 0xFF, 0x00 };
    int8_t  a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0 };
    EXPECT_EQ(mad_binary_dot(w, a, 16), 36);
}

TEST(MadBinaryDot, AllNegativeWeights) {
    // 0x00 = 00000000 => all -1
    uint8_t w[] = { 0x00, 0x00 };
    int8_t  a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0 };
    EXPECT_EQ(mad_binary_dot(w, a, 16), -36);
}

TEST(MadBinaryDot, AlternatingWeights) {
    // 0xAA = 10101010 => [+1,-1,+1,-1,+1,-1,+1,-1]
    // expected: 1-2+3-4+5-6+7-8 = -4
    uint8_t w[] = { 0xAA, 0x00 };
    int8_t  a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0 };
    EXPECT_EQ(mad_binary_dot(w, a, 16), -4);
}

TEST(MadBinaryDot, AllZeroActivations) {
    uint8_t w[] = { 0xFF, 0x00 };
    int8_t  a[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    EXPECT_EQ(mad_binary_dot(w, a, 16), 0);
}

TEST(MadBinaryDot, NegativeActivations) {
    uint8_t w[] = { 0xFF, 0x00 };
    int8_t  a[] = { -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };
    EXPECT_EQ(mad_binary_dot(w, a, 16), -8);
}

TEST(MadBinaryDot, TwoBytes) {
    // byte0=0xFF (+1), byte1=0x00 (-1), all activations=1 => 8-8=0
    uint8_t w[] = { 0xFF, 0x00 };
    int8_t  a[] = { 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1 };
    EXPECT_EQ(mad_binary_dot(w, a, 16), 0);
}

TEST(MadBinaryDot, MatchesNaive_Basic) {
    uint8_t w[] = { 0xAA, 0x55 };
    int8_t  a[] = { 10,-5,3,7,-2,8,-4,6, 1,-3,5,-7,2,9,-6,4 };
    EXPECT_EQ(mad_binary_dot(w, a, 16), naive_binary_dot(w, a, 16));
}

TEST(MadBinaryDot, MatchesNaive_Large) {
    // n=1024 exercises the acc16→acc32 flush path
    const int N = 1024;
    uint8_t w[N / 8];
    int8_t  a[N];
    for (int i = 0; i < N / 8; i++) w[i] = (uint8_t)(i * 37 + 13);
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    EXPECT_EQ(mad_binary_dot(w, a, N), naive_binary_dot(w, a, N));
}
