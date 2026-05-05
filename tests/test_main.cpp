#include <gtest/gtest.h>
#include "src/naive.h"

// Encodes four ternary weights {-1,0,+1} into one packed byte (codes: 0=-1, 1=0, 2=+1).
static uint8_t pack4(int w0, int w1, int w2, int w3) {
    return (uint8_t)(((w0 + 1) << 6) | ((w1 + 1) << 4) | ((w2 + 1) << 2) | (w3 + 1));
}

// naive_ternary_dot

TEST(NaiveTernaryDot, AllZeroWeights) {
    uint8_t w[] = { pack4(0, 0, 0, 0) };
    int8_t  a[] = { 1, 2, 3, 4 };
    EXPECT_EQ(naive_ternary_dot(w, a, 4), 0);
}

TEST(NaiveTernaryDot, AllPositiveWeights) {
    uint8_t w[] = { pack4(1, 1, 1, 1) };
    int8_t  a[] = { 1, 2, 3, 4 };
    EXPECT_EQ(naive_ternary_dot(w, a, 4), 10);
}

TEST(NaiveTernaryDot, AllNegativeWeights) {
    uint8_t w[] = { pack4(-1, -1, -1, -1) };
    int8_t  a[] = { 1, 2, 3, 4 };
    EXPECT_EQ(naive_ternary_dot(w, a, 4), -10);
}

TEST(NaiveTernaryDot, MixedWeights) {
    // weights [-1, 0, +1, -1], activations [10, 20, 30, 40]
    // expected: -10 + 0 + 30 - 40 = -20
    uint8_t w[] = { pack4(-1, 0, 1, -1) };
    int8_t  a[] = { 10, 20, 30, 40 };
    EXPECT_EQ(naive_ternary_dot(w, a, 4), -20);
}

TEST(NaiveTernaryDot, NegativeActivations) {
    // weights [+1, +1, +1, +1], activations [-1, -2, -3, -4]
    // expected: -1 - 2 - 3 - 4 = -10
    uint8_t w[] = { pack4(1, 1, 1, 1) };
    int8_t  a[] = { -1, -2, -3, -4 };
    EXPECT_EQ(naive_ternary_dot(w, a, 4), -10);
}

TEST(NaiveTernaryDot, TwoBytes) {
    // weights [+1,-1, 0,+1, -1,+1, 0,-1], activations [1,2,3,4,5,6,7,8]
    // expected: 1 - 2 + 0 + 4 - 5 + 6 + 0 - 8 = -4
    uint8_t w[] = { pack4(1, -1, 0, 1), pack4(-1, 1, 0, -1) };
    int8_t  a[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    EXPECT_EQ(naive_ternary_dot(w, a, 8), -4);
}

TEST(NaiveTernaryDot, AllZeroActivations) {
    uint8_t w[] = { pack4(1, -1, 1, -1) };
    int8_t  a[] = { 0, 0, 0, 0 };
    EXPECT_EQ(naive_ternary_dot(w, a, 4), 0);
}

// naive_binary_dot

TEST(NaiveBinaryDot, AllPositiveWeights) {
    // 0xFF = 11111111 => all +1
    uint8_t w[] = { 0xFF };
    int8_t  a[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    EXPECT_EQ(naive_binary_dot(w, a, 8), 36);
}

TEST(NaiveBinaryDot, AllNegativeWeights) {
    // 0x00 = 00000000 => all -1
    uint8_t w[] = { 0x00 };
    int8_t  a[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    EXPECT_EQ(naive_binary_dot(w, a, 8), -36);
}

TEST(NaiveBinaryDot, AlternatingWeights) {
    // 0xAA = 10101010 => [+1,-1,+1,-1,+1,-1,+1,-1]
    // expected: 1-2+3-4+5-6+7-8 = -4
    uint8_t w[] = { 0xAA };
    int8_t  a[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    EXPECT_EQ(naive_binary_dot(w, a, 8), -4);
}

TEST(NaiveBinaryDot, AllZeroActivations) {
    uint8_t w[] = { 0xFF };
    int8_t  a[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    EXPECT_EQ(naive_binary_dot(w, a, 8), 0);
}

TEST(NaiveBinaryDot, TwoBytes) {
    // byte0=0xFF (+1 for positions 0-7), byte1=0x00 (-1 for positions 8-15)
    // activations all 1 => sum = 8*1 + 8*(-1) = 0
    uint8_t w[] = { 0xFF, 0x00 };
    int8_t  a[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    EXPECT_EQ(naive_binary_dot(w, a, 16), 0);
}

TEST(NaiveBinaryDot, NegativeActivations) {
    // 0xFF => all +1, activations all -1 => sum = -8
    uint8_t w[] = { 0xFF };
    int8_t  a[] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    EXPECT_EQ(naive_binary_dot(w, a, 8), -8);
}
