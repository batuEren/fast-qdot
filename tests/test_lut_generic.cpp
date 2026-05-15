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

// LCM(1..5) = 60 — all group sizes N=1..5 divide this exactly
static constexpr int kN60  = 60;
static constexpr int kN300 = 300;

// Runs fn<N>() for every N in 1..5
#define FOR_ALL_N(fn) fn<1>(); fn<2>(); fn<3>(); fn<4>(); fn<5>()

// ============================================================
// p_lut_ternary_dot — helpers templated on group size N
// ============================================================

template<int N>
static void t_all_zero_weights() {
    uint8_t w[kN60];
    int8_t  a[kN60];
    for (int i = 0; i < kN60; i++) { w[i] = 1; a[i] = (int8_t)(i % 100 + 1); }
    EXPECT_EQ(p_lut_ternary_dot<N>(w, a, kN60), 0) << "N=" << N;
}

template<int N>
static void t_all_positive_weights() {
    uint8_t w[kN60];
    int8_t  a[kN60];
    int32_t expected = 0;
    for (int i = 0; i < kN60; i++) { w[i] = 2; a[i] = (int8_t)(i + 1); expected += a[i]; }
    EXPECT_EQ(p_lut_ternary_dot<N>(w, a, kN60), expected) << "N=" << N;
}

template<int N>
static void t_all_negative_weights() {
    uint8_t w[kN60];
    int8_t  a[kN60];
    int32_t expected = 0;
    for (int i = 0; i < kN60; i++) { w[i] = 0; a[i] = (int8_t)(i + 1); expected -= a[i]; }
    EXPECT_EQ(p_lut_ternary_dot<N>(w, a, kN60), expected) << "N=" << N;
}

template<int N>
static void t_all_zero_activations() {
    uint8_t w[kN60];
    int8_t  a[kN60] = {};
    for (int i = 0; i < kN60; i++) w[i] = (uint8_t)(i % 3);
    EXPECT_EQ(p_lut_ternary_dot<N>(w, a, kN60), 0) << "N=" << N;
}

template<int N>
static void t_matches_naive_basic() {
    uint8_t w[kN60];
    int8_t  a[kN60];
    for (int i = 0; i < kN60; i++) { w[i] = (uint8_t)(i % 3); a[i] = (int8_t)((i % 20) - 10); }
    EXPECT_EQ(p_lut_ternary_dot<N>(w, a, kN60), naive_p_ternary_dot(w, a, kN60)) << "N=" << N;
}

template<int N>
static void t_matches_naive_negative_activations() {
    uint8_t w[kN60];
    int8_t  a[kN60];
    for (int i = 0; i < kN60; i++) { w[i] = (uint8_t)(i % 3); a[i] = (int8_t)(-(i % 100 + 1)); }
    EXPECT_EQ(p_lut_ternary_dot<N>(w, a, kN60), naive_p_ternary_dot(w, a, kN60)) << "N=" << N;
}

template<int N>
static void t_matches_naive_max_activations() {
    uint8_t w[kN60];
    int8_t  a[kN60];
    for (int i = 0; i < kN60; i++) {
        w[i] = (uint8_t)(i % 3);
        a[i] = (i % 2 == 0) ? (int8_t)127 : (int8_t)-128;
    }
    EXPECT_EQ(p_lut_ternary_dot<N>(w, a, kN60), naive_p_ternary_dot(w, a, kN60)) << "N=" << N;
}

template<int N>
static void t_matches_naive_large() {
    uint8_t w[kN300];
    int8_t  a[kN300];
    for (int i = 0; i < kN300; i++) { w[i] = (uint8_t)(i % 3); a[i] = (int8_t)(i % 127 + 1); }
    EXPECT_EQ(p_lut_ternary_dot<N>(w, a, kN300), naive_p_ternary_dot(w, a, kN300)) << "N=" << N;
}

// ============================================================
// p_lut_ternary_dot — tests (N = 1..5)
// ============================================================

TEST(PLutTernaryDot, AllZeroWeights)              { FOR_ALL_N(t_all_zero_weights); }
TEST(PLutTernaryDot, AllPositiveWeights)          { FOR_ALL_N(t_all_positive_weights); }
TEST(PLutTernaryDot, AllNegativeWeights)          { FOR_ALL_N(t_all_negative_weights); }
TEST(PLutTernaryDot, AllZeroActivations)          { FOR_ALL_N(t_all_zero_activations); }
TEST(PLutTernaryDot, MatchesNaive_Basic)          { FOR_ALL_N(t_matches_naive_basic); }
TEST(PLutTernaryDot, MatchesNaive_NegativeActs)   { FOR_ALL_N(t_matches_naive_negative_activations); }
TEST(PLutTernaryDot, MatchesNaive_MaxActivations) { FOR_ALL_N(t_matches_naive_max_activations); }
TEST(PLutTernaryDot, MatchesNaive_Large)          { FOR_ALL_N(t_matches_naive_large); }

// N=3 specific: pattern 0,2,1 repeating — -1*1+1*2+0*3 ... = 4
TEST(PLutTernaryDot, KnownResult_N3) {
    uint8_t w[12] = { 0,2,1,0,2,1,0,2,1,0,2,1 };
    int8_t  a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    EXPECT_EQ(p_lut_ternary_dot<3>(w, a, 12), 4);
}

// ============================================================
// p_lut_binary_dot — helpers templated on group size N
// ============================================================

template<int N>
static void b_all_positive_weights() {
    uint8_t w[kN60];
    int8_t  a[kN60];
    int32_t expected = 0;
    for (int i = 0; i < kN60; i++) { w[i] = 1; a[i] = (int8_t)(i + 1); expected += a[i]; }
    EXPECT_EQ(p_lut_binary_dot<N>(w, a, kN60), expected) << "N=" << N;
}

template<int N>
static void b_all_negative_weights() {
    uint8_t w[kN60];
    int8_t  a[kN60];
    int32_t expected = 0;
    for (int i = 0; i < kN60; i++) { w[i] = 0; a[i] = (int8_t)(i + 1); expected -= a[i]; }
    EXPECT_EQ(p_lut_binary_dot<N>(w, a, kN60), expected) << "N=" << N;
}

template<int N>
static void b_alternating_weights() {
    // w[i] = i%2==0 ? 1 : 0 → contribution (+1 if even index, -1 if odd), independent of N
    uint8_t w[kN60];
    int8_t  a[kN60];
    int32_t expected = 0;
    for (int i = 0; i < kN60; i++) {
        w[i] = (uint8_t)(i % 2 == 0 ? 1 : 0);
        a[i] = (int8_t)(i + 1);
        expected += (i % 2 == 0 ? +1 : -1) * (int32_t)a[i];
    }
    EXPECT_EQ(p_lut_binary_dot<N>(w, a, kN60), expected) << "N=" << N;
}

template<int N>
static void b_all_zero_activations() {
    uint8_t w[kN60];
    int8_t  a[kN60] = {};
    for (int i = 0; i < kN60; i++) w[i] = (uint8_t)(i % 2);
    EXPECT_EQ(p_lut_binary_dot<N>(w, a, kN60), 0) << "N=" << N;
}

template<int N>
static void b_matches_naive_basic() {
    uint8_t w[kN60];
    int8_t  a[kN60];
    for (int i = 0; i < kN60; i++) { w[i] = (uint8_t)(i % 2); a[i] = (int8_t)((i % 20) - 10); }
    EXPECT_EQ(p_lut_binary_dot<N>(w, a, kN60), naive_p_binary_dot(w, a, kN60)) << "N=" << N;
}

template<int N>
static void b_matches_naive_max_activations() {
    uint8_t w[kN60];
    int8_t  a[kN60];
    for (int i = 0; i < kN60; i++) {
        w[i] = (uint8_t)(i % 2);
        a[i] = (i % 2 == 0) ? (int8_t)127 : (int8_t)-128;
    }
    EXPECT_EQ(p_lut_binary_dot<N>(w, a, kN60), naive_p_binary_dot(w, a, kN60)) << "N=" << N;
}

template<int N>
static void b_matches_naive_large() {
    uint8_t w[kN300];
    int8_t  a[kN300];
    for (int i = 0; i < kN300; i++) { w[i] = (uint8_t)(i % 2); a[i] = (int8_t)(i % 127 + 1); }
    EXPECT_EQ(p_lut_binary_dot<N>(w, a, kN300), naive_p_binary_dot(w, a, kN300)) << "N=" << N;
}

// ============================================================
// p_lut_binary_dot — tests (N = 1..5)
// ============================================================

TEST(PLutBinaryDot, AllPositiveWeights)          { FOR_ALL_N(b_all_positive_weights); }
TEST(PLutBinaryDot, AllNegativeWeights)          { FOR_ALL_N(b_all_negative_weights); }
TEST(PLutBinaryDot, AlternatingWeights)          { FOR_ALL_N(b_alternating_weights); }
TEST(PLutBinaryDot, AllZeroActivations)          { FOR_ALL_N(b_all_zero_activations); }
TEST(PLutBinaryDot, MatchesNaive_Basic)          { FOR_ALL_N(b_matches_naive_basic); }
TEST(PLutBinaryDot, MatchesNaive_MaxActivations) { FOR_ALL_N(b_matches_naive_max_activations); }
TEST(PLutBinaryDot, MatchesNaive_Large)          { FOR_ALL_N(b_matches_naive_large); }

// N=4 specific: -1+2-3+4+5-6+7-8+9+10-11-12 = -4
TEST(PLutBinaryDot, KnownResult_N4) {
    uint8_t w[12] = { 0,1,0,1,1,0,1,0,1,1,0,0 };
    int8_t  a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    EXPECT_EQ(p_lut_binary_dot<4>(w, a, 12), -4);
}
