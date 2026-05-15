#include <gtest/gtest.h>
#include "src/procedural_lut.h"
#include "src/naive.h"

// Scalar references using the procedural_lut weight encoding (one uint8_t per weight).
static int32_t ref_p_binary(const uint8_t* w, const int8_t* a, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += (w[i] ? +1 : -1) * (int32_t)a[i];
    return sum;
}

static int32_t ref_p_ternary(const uint8_t* w, const int8_t* a, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += ((int)w[i] - 1) * (int32_t)a[i];
    return sum;
}

// Naive packing helpers matching the naive.cpp bit layout.
// Ternary: 4 weights per byte, MSB-first, code = w+1 (0→-1, 1→0, 2→+1)
static uint8_t pack_nt(int w0, int w1, int w2, int w3) {
    return (uint8_t)(((w0+1) << 6) | ((w1+1) << 4) | ((w2+1) << 2) | (w3+1));
}

// Binary: 8 weights per byte, MSB-first, bit=1→+1, bit=0→-1
static uint8_t pack_nb(int w0, int w1, int w2, int w3, int w4, int w5, int w6, int w7) {
    int ws[8] = {w0,w1,w2,w3,w4,w5,w6,w7};
    uint8_t b = 0;
    for (int j = 0; j < 8; j++)
        if (ws[j] == +1) b |= (uint8_t)(1 << (7-j));
    return b;
}

// ── PLutBinaryMatVec ──────────────────────────────────────────────────────────

TEST(PLutBinaryMatVec, AllPositiveWeights) {
    // weight 1 = +1; all rows sum to 1+2+...+12 = 78
    int8_t a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    uint8_t w[3 * 12];
    for (int i = 0; i < 3 * 12; i++) w[i] = 1;
    auto result = p_lut_binary_matrix_vector_prod<4>(w, a, 3, 12);
    ASSERT_EQ((int)result.size(), 3);
    for (int row = 0; row < 3; row++) EXPECT_EQ(result[row], 78) << "row " << row;
}

TEST(PLutBinaryMatVec, AllNegativeWeights) {
    // weight 0 = -1; all rows sum to -78
    int8_t a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    uint8_t w[3 * 12];
    for (int i = 0; i < 3 * 12; i++) w[i] = 0;
    auto result = p_lut_binary_matrix_vector_prod<4>(w, a, 3, 12);
    ASSERT_EQ((int)result.size(), 3);
    for (int row = 0; row < 3; row++) EXPECT_EQ(result[row], -78) << "row " << row;
}

TEST(PLutBinaryMatVec, MatchesPerRowDot) {
    // Each result[row] must equal p_lut_binary_dot<4> applied to that row.
    const int M = 4, N = 12;
    int8_t a[N] = { 10,-5,3,7,-2,8,-4,6,1,-3,5,-7 };
    uint8_t w[M * N] = {
        1,0,1,1,0,1,0,1,1,0,0,1,
        0,1,0,0,1,0,1,0,0,1,1,0,
        1,1,0,0,1,1,0,0,1,1,0,0,
        0,0,1,1,0,0,1,1,0,0,1,1,
    };
    auto result = p_lut_binary_matrix_vector_prod<4>(w, a, M, N);
    ASSERT_EQ((int)result.size(), M);
    for (int row = 0; row < M; row++)
        EXPECT_EQ(result[row], p_lut_binary_dot<4>(w + row * N, a, N)) << "row " << row;
}

TEST(PLutBinaryMatVec, MatchesNaiveRef) {
    // Cross-validate against the scalar reference.
    const int M = 4, N = 12;
    int8_t a[N] = { 10,-5,3,7,-2,8,-4,6,1,-3,5,-7 };
    uint8_t w[M * N] = {
        1,0,1,1,0,1,0,1,1,0,0,1,
        0,1,0,0,1,0,1,0,0,1,1,0,
        1,1,0,0,1,1,0,0,1,1,0,0,
        0,0,1,1,0,0,1,1,0,0,1,1,
    };
    auto result = p_lut_binary_matrix_vector_prod<4>(w, a, M, N);
    ASSERT_EQ((int)result.size(), M);
    for (int row = 0; row < M; row++)
        EXPECT_EQ(result[row], ref_p_binary(w + row * N, a, N)) << "row " << row;
}

TEST(PLutBinaryMatVec, Large) {
    const int M = 8, N = 1024;
    uint8_t w[M * N];
    int8_t a[N];
    for (int i = 0; i < M * N; i++) w[i] = (uint8_t)(i % 2);
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    auto result = p_lut_binary_matrix_vector_prod<4>(w, a, M, N);
    ASSERT_EQ((int)result.size(), M);
    for (int row = 0; row < M; row++)
        EXPECT_EQ(result[row], p_lut_binary_dot<4>(w + row * N, a, N)) << "row " << row;
}

// ── PLutTernaryMatVec ─────────────────────────────────────────────────────────

TEST(PLutTernaryMatVec, AllPositiveWeights) {
    // weight 2 = +1; sum = 78
    int8_t a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    uint8_t w[3 * 12];
    for (int i = 0; i < 3 * 12; i++) w[i] = 2;
    auto result = p_lut_ternary_matrix_vector_prod<3>(w, a, 3, 12);
    ASSERT_EQ((int)result.size(), 3);
    for (int row = 0; row < 3; row++) EXPECT_EQ(result[row], 78) << "row " << row;
}

TEST(PLutTernaryMatVec, AllNegativeWeights) {
    // weight 0 = -1; sum = -78
    int8_t a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    uint8_t w[3 * 12];
    for (int i = 0; i < 3 * 12; i++) w[i] = 0;
    auto result = p_lut_ternary_matrix_vector_prod<3>(w, a, 3, 12);
    ASSERT_EQ((int)result.size(), 3);
    for (int row = 0; row < 3; row++) EXPECT_EQ(result[row], -78) << "row " << row;
}

TEST(PLutTernaryMatVec, AllZeroWeights) {
    // weight 1 = 0; sum = 0
    int8_t a[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    uint8_t w[3 * 12];
    for (int i = 0; i < 3 * 12; i++) w[i] = 1;
    auto result = p_lut_ternary_matrix_vector_prod<3>(w, a, 3, 12);
    ASSERT_EQ((int)result.size(), 3);
    for (int row = 0; row < 3; row++) EXPECT_EQ(result[row], 0) << "row " << row;
}

TEST(PLutTernaryMatVec, MatchesPerRowDot) {
    const int M = 4, N = 12;
    int8_t a[N] = { 10,-5,3,7,-2,8,-4,6,1,-3,5,-7 };
    uint8_t w[M * N] = {
        0,1,2,2,0,1,1,2,0,0,2,1,
        2,0,1,1,2,0,0,1,2,2,0,1,
        1,2,0,0,1,2,2,0,1,1,2,0,
        0,0,2,1,2,0,1,1,0,2,1,2,
    };
    auto result = p_lut_ternary_matrix_vector_prod<3>(w, a, M, N);
    ASSERT_EQ((int)result.size(), M);
    for (int row = 0; row < M; row++)
        EXPECT_EQ(result[row], p_lut_ternary_dot<3>(w + row * N, a, N)) << "row " << row;
}

TEST(PLutTernaryMatVec, MatchesNaiveRef) {
    const int M = 4, N = 12;
    int8_t a[N] = { 10,-5,3,7,-2,8,-4,6,1,-3,5,-7 };
    uint8_t w[M * N] = {
        0,1,2,2,0,1,1,2,0,0,2,1,
        2,0,1,1,2,0,0,1,2,2,0,1,
        1,2,0,0,1,2,2,0,1,1,2,0,
        0,0,2,1,2,0,1,1,0,2,1,2,
    };
    auto result = p_lut_ternary_matrix_vector_prod<3>(w, a, M, N);
    ASSERT_EQ((int)result.size(), M);
    for (int row = 0; row < M; row++)
        EXPECT_EQ(result[row], ref_p_ternary(w + row * N, a, N)) << "row " << row;
}

TEST(PLutTernaryMatVec, Large) {
    const int M = 8, N = 1023; // 1023 = 3*341
    uint8_t w[M * N];
    int8_t a[N];
    for (int i = 0; i < M * N; i++) w[i] = (uint8_t)(i % 3);
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    auto result = p_lut_ternary_matrix_vector_prod<3>(w, a, M, N);
    ASSERT_EQ((int)result.size(), M);
    for (int row = 0; row < M; row++)
        EXPECT_EQ(result[row], p_lut_ternary_dot<3>(w + row * N, a, N)) << "row " << row;
}

// ── NaiveTernaryMatVec ────────────────────────────────────────────────────────

TEST(NaiveTernaryMatVec, AllPositiveWeights) {
    // n=16, 4 bytes per row; all +1 → sum = 136
    int8_t a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    uint8_t w[3 * 4] = {
        pack_nt(1,1,1,1), pack_nt(1,1,1,1), pack_nt(1,1,1,1), pack_nt(1,1,1,1),
        pack_nt(1,1,1,1), pack_nt(1,1,1,1), pack_nt(1,1,1,1), pack_nt(1,1,1,1),
        pack_nt(1,1,1,1), pack_nt(1,1,1,1), pack_nt(1,1,1,1), pack_nt(1,1,1,1),
    };
    auto result = naive_ternary_matrix_vector_prod(w, a, 3, 16);
    ASSERT_EQ((int)result.size(), 3);
    for (int row = 0; row < 3; row++) EXPECT_EQ(result[row], 136) << "row " << row;
}

TEST(NaiveTernaryMatVec, AllNegativeWeights) {
    int8_t a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    uint8_t w[3 * 4] = {
        pack_nt(-1,-1,-1,-1), pack_nt(-1,-1,-1,-1), pack_nt(-1,-1,-1,-1), pack_nt(-1,-1,-1,-1),
        pack_nt(-1,-1,-1,-1), pack_nt(-1,-1,-1,-1), pack_nt(-1,-1,-1,-1), pack_nt(-1,-1,-1,-1),
        pack_nt(-1,-1,-1,-1), pack_nt(-1,-1,-1,-1), pack_nt(-1,-1,-1,-1), pack_nt(-1,-1,-1,-1),
    };
    auto result = naive_ternary_matrix_vector_prod(w, a, 3, 16);
    ASSERT_EQ((int)result.size(), 3);
    for (int row = 0; row < 3; row++) EXPECT_EQ(result[row], -136) << "row " << row;
}

TEST(NaiveTernaryMatVec, MatchesPerRowDot) {
    // Stride is n/4 bytes per row in the packed array.
    const int M = 3, N = 16;
    int8_t a[N] = { 10,-5,3,7,-2,8,-4,6,1,-3,5,-7,2,9,-6,4 };
    uint8_t w[M * (N/4)] = {
        pack_nt(-1,0,1,-1), pack_nt(0,1,-1,0),  pack_nt(1,1,-1,0),  pack_nt(-1,-1,1,1),
        pack_nt(1,-1,0,1),  pack_nt(-1,0,1,-1),  pack_nt(0,-1,1,0),  pack_nt(1,0,-1,1),
        pack_nt(0,0,1,-1),  pack_nt(1,-1,0,0),   pack_nt(-1,1,0,1),  pack_nt(0,1,-1,-1),
    };
    auto result = naive_ternary_matrix_vector_prod(w, a, M, N);
    ASSERT_EQ((int)result.size(), M);
    for (int row = 0; row < M; row++)
        EXPECT_EQ(result[row], naive_ternary_dot(w + row * (N/4), a, N)) << "row " << row;
}

TEST(NaiveTernaryMatVec, Large) {
    const int M = 4, N = 64;
    uint8_t w[M * (N/4)];
    int8_t a[N];
    for (int i = 0; i < M * (N/4); i++) {
        int ws[4];
        for (int j = 0; j < 4; j++) ws[j] = (i + j) % 3 - 1;
        w[i] = pack_nt(ws[0], ws[1], ws[2], ws[3]);
    }
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    auto result = naive_ternary_matrix_vector_prod(w, a, M, N);
    ASSERT_EQ((int)result.size(), M);
    for (int row = 0; row < M; row++)
        EXPECT_EQ(result[row], naive_ternary_dot(w + row * (N/4), a, N)) << "row " << row;
}

// ── NaiveBinaryMatVec ─────────────────────────────────────────────────────────

TEST(NaiveBinaryMatVec, AllPositiveWeights) {
    // n=16, 2 bytes per row; all +1 → sum = 136
    int8_t a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    uint8_t w[3 * 2] = {
        pack_nb(1,1,1,1,1,1,1,1), pack_nb(1,1,1,1,1,1,1,1),
        pack_nb(1,1,1,1,1,1,1,1), pack_nb(1,1,1,1,1,1,1,1),
        pack_nb(1,1,1,1,1,1,1,1), pack_nb(1,1,1,1,1,1,1,1),
    };
    auto result = naive_binary_matrix_vector_prod(w, a, 3, 16);
    ASSERT_EQ((int)result.size(), 3);
    for (int row = 0; row < 3; row++) EXPECT_EQ(result[row], 136) << "row " << row;
}

TEST(NaiveBinaryMatVec, AllNegativeWeights) {
    int8_t a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    uint8_t w[3 * 2] = {
        pack_nb(-1,-1,-1,-1,-1,-1,-1,-1), pack_nb(-1,-1,-1,-1,-1,-1,-1,-1),
        pack_nb(-1,-1,-1,-1,-1,-1,-1,-1), pack_nb(-1,-1,-1,-1,-1,-1,-1,-1),
        pack_nb(-1,-1,-1,-1,-1,-1,-1,-1), pack_nb(-1,-1,-1,-1,-1,-1,-1,-1),
    };
    auto result = naive_binary_matrix_vector_prod(w, a, 3, 16);
    ASSERT_EQ((int)result.size(), 3);
    for (int row = 0; row < 3; row++) EXPECT_EQ(result[row], -136) << "row " << row;
}

TEST(NaiveBinaryMatVec, MatchesPerRowDot) {
    // Stride is n/8 bytes per row in the packed array.
    const int M = 3, N = 16;
    int8_t a[N] = { 10,-5,3,7,-2,8,-4,6,1,-3,5,-7,2,9,-6,4 };
    uint8_t w[M * (N/8)] = {
        pack_nb(1,-1,1,1,-1,1,-1,-1), pack_nb(-1,1,-1,1,1,-1,1,1),
        pack_nb(-1,-1,1,1,1,-1,-1,1), pack_nb(1,1,-1,-1,-1,1,1,-1),
        pack_nb(1,1,1,-1,-1,-1,1,-1), pack_nb(-1,1,1,-1,1,-1,-1,1),
    };
    auto result = naive_binary_matrix_vector_prod(w, a, M, N);
    ASSERT_EQ((int)result.size(), M);
    for (int row = 0; row < M; row++)
        EXPECT_EQ(result[row], naive_binary_dot(w + row * (N/8), a, N)) << "row " << row;
}

TEST(NaiveBinaryMatVec, Large) {
    const int M = 4, N = 64;
    uint8_t w[M * (N/8)];
    int8_t a[N];
    for (int i = 0; i < M * (N/8); i++) {
        int s[8];
        for (int j = 0; j < 8; j++) s[j] = ((i + j) % 2) ? +1 : -1;
        w[i] = pack_nb(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7]);
    }
    for (int i = 0; i < N; i++) a[i] = (int8_t)(i % 127 + 1);
    auto result = naive_binary_matrix_vector_prod(w, a, M, N);
    ASSERT_EQ((int)result.size(), M);
    for (int row = 0; row < M; row++)
        EXPECT_EQ(result[row], naive_binary_dot(w + row * (N/8), a, N)) << "row " << row;
}
