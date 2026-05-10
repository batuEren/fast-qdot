#include <gtest/gtest.h>
#include "src/lut.h"

// Packs 4 ternary weights into one byte using the nibble-packed pair-index format:
//   high nibble = 3*(w0+1) + (w1+1)  for pair (w0, w1)
//   low  nibble = 3*(w2+1) + (w3+1)  for pair (w2, w3)
static uint8_t pack2(int w0, int w1, int w2, int w3) {
    uint8_t hi = (uint8_t)(3*(w0+1) + (w1+1));
    uint8_t lo = (uint8_t)(3*(w2+1) + (w3+1));
    return (uint8_t)((hi << 4) | lo);
}

// Naive reference that decodes the same nibble-packed format.
static int32_t naive_lut_ternary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n / 4; i++) {
        uint8_t b = weights[i];
        int hi = (b >> 4) & 0xf;
        int lo = b & 0xf;
        sum += (hi / 3 - 1) * (int32_t)activations[4*i + 0];
        sum += (hi % 3 - 1) * (int32_t)activations[4*i + 1];
        sum += (lo / 3 - 1) * (int32_t)activations[4*i + 2];
        sum += (lo % 3 - 1) * (int32_t)activations[4*i + 3];
    }
    return sum;
}

// Packs 8 binary weights into one byte using the 2-bit pair-index format:
//   bits[7:6] = pair 0 index, bits[5:4] = pair 1, bits[3:2] = pair 2, bits[1:0] = pair 3.
//   pair_index = 2*(w_even==+1) + (w_odd==+1), w ∈ {-1,+1}.
static uint8_t pack_binary(int w0, int w1, int w2, int w3, int w4, int w5, int w6, int w7) {
    auto p = [](int a, int b) -> uint8_t { return (uint8_t)(2*((a+1)/2) + (b+1)/2); };
    return (uint8_t)((p(w0,w1) << 6) | (p(w2,w3) << 4) | (p(w4,w5) << 2) | p(w6,w7));
}

static int32_t naive_lut_binary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n / 8; i++) {
        uint8_t b = weights[i];
        for (int j = 0; j < 4; j++) {
            int idx = (b >> (6 - 2*j)) & 0x3;
            int we = (idx >> 1) ? +1 : -1;
            int wo = (idx  & 1) ? +1 : -1;
            sum += we * (int32_t)activations[8*i + 2*j    ];
            sum += wo * (int32_t)activations[8*i + 2*j + 1];
        }
    }
    return sum;
}

// lut_ternary_dot (n must be a multiple of 16)

TEST(LutTernaryDot, AllZeroWeights) {
    // pair index 4 = (0,0) → zero contribution regardless of activations
    uint8_t w[4] = { pack2(0,0,0,0), pack2(0,0,0,0), pack2(0,0,0,0), pack2(0,0,0,0) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(lut_ternary_dot(w, a, 16), 0);
}

TEST(LutTernaryDot, AllPositiveWeights) {
    // all +1 weights, activations 1..16 → sum = 136
    uint8_t w[4] = { pack2(1,1,1,1), pack2(1,1,1,1), pack2(1,1,1,1), pack2(1,1,1,1) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(lut_ternary_dot(w, a, 16), 136);
}

TEST(LutTernaryDot, AllNegativeWeights) {
    // all -1 weights, activations 1..16 → sum = -136
    uint8_t w[4] = { pack2(-1,-1,-1,-1), pack2(-1,-1,-1,-1), pack2(-1,-1,-1,-1), pack2(-1,-1,-1,-1) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(lut_ternary_dot(w, a, 16), -136);
}

TEST(LutTernaryDot, AllZeroActivations) {
    uint8_t w[4] = { pack2(1,-1,0,1), pack2(-1,1,0,-1), pack2(1,1,-1,0), pack2(-1,-1,1,1) };
    int8_t  a[16] = {};
    EXPECT_EQ(lut_ternary_dot(w, a, 16), 0);
}

TEST(LutTernaryDot, MixedWeightsKnownResult) {
    // pairs: (-1,-1) (0,+1) (+1,-1) (0,0) (+1,+1) (-1,0) (0,+1) (+1,-1)
    // activations: 1..16
    // pair 0: -1*1  + -1*2  = -3
    // pair 1:  0*3  +  1*4  =  4
    // pair 2:  1*5  + -1*6  = -1
    // pair 3:  0*7  +  0*8  =  0
    // pair 4:  1*9  +  1*10 = 19
    // pair 5: -1*11 +  0*12 = -11
    // pair 6:  0*13 +  1*14 = 14
    // pair 7:  1*15 + -1*16 = -1
    // total = 21
    uint8_t w[4] = { pack2(-1,-1,0,1), pack2(1,-1,0,0), pack2(1,1,-1,0), pack2(0,1,1,-1) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(lut_ternary_dot(w, a, 16), 21);
}

TEST(LutTernaryDot, MatchesNaive_Basic) {
    uint8_t w[4] = { pack2(-1,0,1,-1), pack2(0,1,-1,0), pack2(1,1,-1,0), pack2(-1,-1,1,1) };
    int8_t  a[16] = { 10,-5,3,7,-2,8,-4,6,1,-3,5,-7,2,9,-6,4 };
    EXPECT_EQ(lut_ternary_dot(w, a, 16), naive_lut_ternary_dot(w, a, 16));
}

TEST(LutTernaryDot, MatchesNaive_NegativeActivations) {
    uint8_t w[4] = { pack2(1,1,1,1), pack2(-1,-1,-1,-1), pack2(1,-1,0,0), pack2(0,0,1,-1) };
    int8_t  a[16] = { -1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16 };
    EXPECT_EQ(lut_ternary_dot(w, a, 16), naive_lut_ternary_dot(w, a, 16));
}

TEST(LutTernaryDot, MatchesNaive_MaxActivations) {
    // int8 extremes to stress the int16 overflow guard in LUT construction
    uint8_t w[4] = { pack2(1,-1,1,-1), pack2(-1,1,-1,1), pack2(1,1,-1,-1), pack2(-1,-1,1,1) };
    int8_t  a[16] = { 127,-128,127,-128,127,-128,127,-128,127,-128,127,-128,127,-128,127,-128 };
    EXPECT_EQ(lut_ternary_dot(w, a, 16), naive_lut_ternary_dot(w, a, 16));
}

TEST(LutTernaryDot, MatchesNaive_Large) {
    // n=1024 exercises the acc16→acc32 flush path (fires at i=63, 127, ...)
    const int N = 1024;
    uint8_t w[N / 4];
    int8_t  a[N];
    for (int i = 0; i < N / 4; i++) {
        int ws[4];
        for (int j = 0; j < 4; j++) ws[j] = (i + j) % 3 - 1;
        w[i] = pack2(ws[0], ws[1], ws[2], ws[3]);
    }
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    EXPECT_EQ(lut_ternary_dot(w, a, N), naive_lut_ternary_dot(w, a, N));
}

// lut_binary_dot (n must be a multiple of 16)

TEST(LutBinaryDot, AllPositiveWeights) {
    // all (+1,+1) pairs → pair_index 3, activations 1..16 → sum = 136
    uint8_t w[2] = { pack_binary(1,1,1,1,1,1,1,1), pack_binary(1,1,1,1,1,1,1,1) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(lut_binary_dot(w, a, 16), 136);
}

TEST(LutBinaryDot, AllNegativeWeights) {
    // all (-1,-1) pairs → pair_index 0, activations 1..16 → sum = -136
    uint8_t w[2] = { pack_binary(-1,-1,-1,-1,-1,-1,-1,-1), pack_binary(-1,-1,-1,-1,-1,-1,-1,-1) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(lut_binary_dot(w, a, 16), -136);
}

TEST(LutBinaryDot, AlternatingWeights) {
    // (+1,-1) repeating → pair_index 2 (a_e - a_o) for all pairs
    // pairs: (1-2), (3-4), (5-6), ..., (15-16) = -1 × 8 = -8
    uint8_t w[2] = { pack_binary(1,-1,1,-1,1,-1,1,-1), pack_binary(1,-1,1,-1,1,-1,1,-1) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(lut_binary_dot(w, a, 16), -8);
}

TEST(LutBinaryDot, AllZeroActivations) {
    uint8_t w[2] = { pack_binary(1,-1,1,-1,1,-1,1,-1), pack_binary(-1,1,-1,1,-1,1,-1,1) };
    int8_t  a[16] = {};
    EXPECT_EQ(lut_binary_dot(w, a, 16), 0);
}

TEST(LutBinaryDot, MixedWeightsKnownResult) {
    // weights: (+1,+1)(-1,+1)(+1,-1)(-1,-1) | (+1,-1)(+1,+1)(-1,+1)(-1,-1)
    // activations: 1..16
    // pair 0 (+1,+1):  1*1  + 1*2  =  3
    // pair 1 (-1,+1): -1*3  + 1*4  =  1
    // pair 2 (+1,-1):  1*5  +-1*6  = -1
    // pair 3 (-1,-1): -1*7  +-1*8  = -15
    // pair 4 (+1,-1):  1*9  +-1*10 = -1
    // pair 5 (+1,+1):  1*11 + 1*12 =  23
    // pair 6 (-1,+1): -1*13 + 1*14 =  1
    // pair 7 (-1,-1): -1*15 +-1*16 = -31
    // total = 3+1-1-15-1+23+1-31 = -20
    uint8_t w[2] = { pack_binary(1,1,-1,1,1,-1,-1,-1), pack_binary(1,-1,1,1,-1,1,-1,-1) };
    int8_t  a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    EXPECT_EQ(lut_binary_dot(w, a, 16), -20);
}

TEST(LutBinaryDot, MatchesNaive_Basic) {
    uint8_t w[2] = { pack_binary(1,-1,1,1,-1,1,-1,-1), pack_binary(-1,1,-1,1,1,-1,1,1) };
    int8_t  a[16] = { 10,-5,3,7,-2,8,-4,6,1,-3,5,-7,2,9,-6,4 };
    EXPECT_EQ(lut_binary_dot(w, a, 16), naive_lut_binary_dot(w, a, 16));
}

TEST(LutBinaryDot, MatchesNaive_MaxActivations) {
    uint8_t w[2] = { pack_binary(1,-1,1,-1,1,-1,1,-1), pack_binary(-1,1,-1,1,-1,1,-1,1) };
    int8_t  a[16] = { 127,-128,127,-128,127,-128,127,-128,127,-128,127,-128,127,-128,127,-128 };
    EXPECT_EQ(lut_binary_dot(w, a, 16), naive_lut_binary_dot(w, a, 16));
}

TEST(LutBinaryDot, MatchesNaive_Large) {
    const int N = 1024;
    uint8_t w[N / 8];
    int8_t  a[N];
    // alternating sign pattern across pairs
    for (int i = 0; i < N / 8; i++) {
        int s[8];
        for (int j = 0; j < 8; j++) s[j] = ((i + j) % 2) ? +1 : -1;
        w[i] = pack_binary(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7]);
    }
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    EXPECT_EQ(lut_binary_dot(w, a, N), naive_lut_binary_dot(w, a, N));
}
