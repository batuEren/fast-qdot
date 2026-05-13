#include <gtest/gtest.h>
#include "src/procedural_lut.h"

// weight encoding: 0 → -1, 1 → 0, 2 → +1
static int32_t naive_p_ternary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++)
        sum += ((int)weights[i] - 1) * (int32_t)activations[i];
    return sum;
}

// weight encoding: 0 → -1, 1 → +1
static int32_t naive_p_binary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++)
        sum += (weights[i] ? +1 : -1) * (int32_t)activations[i];
    return sum;
}

// p_lut_ternary_dot — n must be a multiple of 3

TEST(PLutTernaryDot, AllZeroWeights) {
    // weight 1 encodes zero; all activations are irrelevant
    uint8_t w[12] = { 1,1,1,1,1,1,1,1,1,1,1,1 };
    int8_t  a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    EXPECT_EQ(p_lut_ternary_dot(w, a, 12), 0);
}

TEST(PLutTernaryDot, AllPositiveWeights) {
    // weight 2 = +1; sum = 1+2+...+12 = 78
    uint8_t w[12] = { 2,2,2,2,2,2,2,2,2,2,2,2 };
    int8_t  a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    EXPECT_EQ(p_lut_ternary_dot(w, a, 12), 78);
}

TEST(PLutTernaryDot, AllNegativeWeights) {
    // weight 0 = -1; sum = -78
    uint8_t w[12] = { 0,0,0,0,0,0,0,0,0,0,0,0 };
    int8_t  a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    EXPECT_EQ(p_lut_ternary_dot(w, a, 12), -78);
}

TEST(PLutTernaryDot, AllZeroActivations) {
    uint8_t w[12] = { 0,1,2,0,1,2,0,1,2,0,1,2 };
    int8_t  a[12] = {};
    EXPECT_EQ(p_lut_ternary_dot(w, a, 12), 0);
}

TEST(PLutTernaryDot, MixedWeightsKnownResult) {
    // pattern 0,2,1 repeats: -1*1 +1*2 +0*3 -1*4 +1*5 +0*6 -1*7 +1*8 +0*9 -1*10 +1*11 +0*12 = 4
    uint8_t w[12] = { 0,2,1,0,2,1,0,2,1,0,2,1 };
    int8_t  a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    EXPECT_EQ(p_lut_ternary_dot(w, a, 12), 4);
}

TEST(PLutTernaryDot, MatchesNaive_Basic) {
    uint8_t w[12] = { 0,1,2,2,0,1,1,2,0,0,2,1 };
    int8_t  a[12] = { 10,-5,3,7,-2,8,-4,6,1,-3,5,-7 };
    EXPECT_EQ(p_lut_ternary_dot(w, a, 12), naive_p_ternary_dot(w, a, 12));
}

TEST(PLutTernaryDot, MatchesNaive_NegativeActivations) {
    uint8_t w[12] = { 2,2,2,2,0,0,0,0,1,1,1,1 };
    int8_t  a[12] = { -1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 };
    EXPECT_EQ(p_lut_ternary_dot(w, a, 12), naive_p_ternary_dot(w, a, 12));
}

TEST(PLutTernaryDot, MatchesNaive_MaxActivations) {
    // int8 extremes to stress LUT entry arithmetic
    uint8_t w[12] = { 2,0,2,0,2,0,2,0,2,0,2,0 };
    int8_t  a[12] = { 127,-128,127,-128,127,-128,127,-128,127,-128,127,-128 };
    EXPECT_EQ(p_lut_ternary_dot(w, a, 12), naive_p_ternary_dot(w, a, 12));
}

TEST(PLutTernaryDot, MatchesNaive_Large) {
    // 1023 = 3 * 341, exercises many LUT groups
    const int N = 1023;
    uint8_t w[N];
    int8_t  a[N];
    for (int i = 0; i < N; i++) w[i] = (uint8_t)(i % 3);
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    EXPECT_EQ(p_lut_ternary_dot(w, a, N), naive_p_ternary_dot(w, a, N));
}

// p_lut_binary_dot

TEST(PLutBinaryDot, AllPositiveWeights) {
    // weight 1 = +1; sum = 1+2+...+12 = 78
    uint8_t w[12] = { 1,1,1,1,1,1,1,1,1,1,1,1 };
    int8_t  a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    EXPECT_EQ(p_lut_binary_dot(w, a, 12), 78);
}

TEST(PLutBinaryDot, AllNegativeWeights) {
    // weight 0 = -1; sum = -78
    uint8_t w[12] = { 0,0,0,0,0,0,0,0,0,0,0,0 };
    int8_t  a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    EXPECT_EQ(p_lut_binary_dot(w, a, 12), -78);
}

TEST(PLutBinaryDot, AlternatingWeights) {
    // pairs (+1,-1) repeating: (1-2)+(3-4)+...+(11-12) = -1*6 = -6
    uint8_t w[12] = { 1,0,1,0,1,0,1,0,1,0,1,0 };
    int8_t  a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    EXPECT_EQ(p_lut_binary_dot(w, a, 12), -6);
}

TEST(PLutBinaryDot, AllZeroActivations) {
    uint8_t w[12] = { 1,0,1,0,1,0,1,0,1,0,1,0 };
    int8_t  a[12] = {};
    EXPECT_EQ(p_lut_binary_dot(w, a, 12), 0);
}

TEST(PLutBinaryDot, MixedWeightsKnownResult) {
    // -1*1 +1*2 -1*3 +1*4 +1*5 -1*6 +1*7 -1*8 +1*9 +1*10 -1*11 -1*12
    // = -1+2-3+4+5-6+7-8+9+10-11-12 = -4
    uint8_t w[12] = { 0,1,0,1,1,0,1,0,1,1,0,0 };
    int8_t  a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    EXPECT_EQ(p_lut_binary_dot(w, a, 12), -4);
}

TEST(PLutBinaryDot, MatchesNaive_Basic) {
    uint8_t w[12] = { 1,0,1,1,0,1,0,1,1,0,0,1 };
    int8_t  a[12] = { 10,-5,3,7,-2,8,-4,6,1,-3,5,-7 };
    EXPECT_EQ(p_lut_binary_dot(w, a, 12), naive_p_binary_dot(w, a, 12));
}

TEST(PLutBinaryDot, MatchesNaive_MaxActivations) {
    uint8_t w[12] = { 1,0,1,0,1,0,1,0,1,0,1,0 };
    int8_t  a[12] = { 127,-128,127,-128,127,-128,127,-128,127,-128,127,-128 };
    EXPECT_EQ(p_lut_binary_dot(w, a, 12), naive_p_binary_dot(w, a, 12));
}

TEST(PLutBinaryDot, MatchesNaive_Large) {
    const int N = 1024;
    uint8_t w[N];
    int8_t  a[N];
    for (int i = 0; i < N; i++) w[i] = (uint8_t)(i % 2);
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    EXPECT_EQ(p_lut_binary_dot(w, a, N), naive_p_binary_dot(w, a, N));
}
