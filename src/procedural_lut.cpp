#include "procedural_lut.h"
#include <array>
#include <vector>

// Precomputes partial dot products for every ternary weight combination in a group of N.
// LUT index is base-3 encoded: w[0] + w[1]*3 + ... + w[N-1]*3^(N-1), w[k] in {0,1,2}.
// LUT value is the partial sum using weight mapping 0->-1, 1->0, 2->+1.
template<int N>
static auto create_ternary_lut(const int8_t* activations, int n) {
    constexpr int LUT_SIZE = []() {
        int r = 1;
        for (int i = 0; i < N; i++) r *= 3;
        return r;
    }();

    std::vector<std::array<int16_t, LUT_SIZE>> lut(n / N);

    for (int i = 0; i < n / N; i++) {
        for (int j = 0; j < LUT_SIZE; j++) {
            int rem = j;
            int16_t total = 0;
            for (int k = 0; k < N; k++) {
                int w = (rem % 3) - 1;  // (0, 1, 2) -> (-1, 0, 1)
                total = (int16_t)(total + w * activations[i * N + k]);
                rem /= 3;
            }
            lut[i][j] = total;
        }
    }

    return lut;
}

// Precomputes partial dot products for every binary weight combination in a group of N.
// LUT index is base-2 encoded: bit k of index selects w[k] in {0->-1, 1->+1}.
template<int N>
static auto create_binary_lut(const int8_t* activations, int n) {
    constexpr int LUT_SIZE = 1 << N;

    std::vector<std::array<int16_t, LUT_SIZE>> lut(n / N);

    for (int i = 0; i < n / N; i++) {
        for (int j = 0; j < LUT_SIZE; j++) {
            int16_t total = 0;
            for (int k = 0; k < N; k++) {
                int w = ((j >> k) & 1) ? +1 : -1;
                total = (int16_t)(total + w * activations[i * N + k]);
            }
            lut[i][j] = total;
        }
    }

    return lut;
}


int32_t p_lut_ternary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    constexpr int N = 3;
    auto lut = create_ternary_lut<N>(activations, n);
    int32_t result = 0;

    for (int i = 0; i < n / N; i++) {
        int lutIdx = 0;
        int powAcc = 1;
        for (int j = 0; j < N; j++) {
            lutIdx += weights[i * N + j] * powAcc;
            powAcc *= 3;
        }
        result += lut[i][lutIdx];
    }

    return result;
}

int32_t p_lut_binary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    constexpr int N = 2;
    auto lut = create_binary_lut<N>(activations, n);
    int32_t result = 0;

    for (int i = 0; i < n / N; i++) {
        int lutIdx = 0;
        for (int j = 0; j < N; j++) {
            lutIdx |= (weights[i * N + j] & 1) << j;
        }
        result += lut[i][lutIdx];
    }

    return result;
}

int32_t* p_lut_binary_matrix_vector_prod(const uint8_t* weights, const int8_t* activations, int n) {
    
    return 0;
}
