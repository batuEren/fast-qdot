#include "naive.h"
#include <cassert>

// Weight packing: 2-bit per weight, 4 per byte
// Bit layout: bits[7:6]=w0, bits[5:4]=w1, bits[3:2]=w2, bits[1:0]=w3
// Encoding: 0b00=-1, 0b01=0, 0b10=+1  =>  decoded = code - 1

int32_t naive_ternary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    assert(n % 4 == 0);

    int32_t sum = 0;

    for (int i = 0; i < n / 4; i++) {
        uint8_t b = weights[i];
        for (int j = 0; j < 4; j++) {
            sum += ((int32_t)((b >> (3-j)*2) & 0x03) - 1) * (int32_t)activations[i * 4 + j];
        }
    }

    return sum;
}

// Weight packing: 1-bit per weight, 8 per byte
// Bit layout: bits[7]=w0, bits[6]=w1, ..., bits[0]=w7
// Encoding: 0=-1, 1=+1  =>  decoded = 2*bit - 1


int32_t naive_binary_dot(const uint8_t* weights, const int8_t* activations, int n) {
    assert(n % 8 == 0);

    int32_t sum = 0;

    for (int i = 0; i < n / 8; i++) {
        uint8_t b = weights[i];
        for (int j = 0; j < 8; j++) {
            int32_t bit = (b >> (7 - j)) & 0x01;
            sum += (2 * bit - 1) * (int32_t)activations[i * 8 + j];
        }
    }

    return sum;
}

std::vector<int32_t> naive_ternary_matrix_vector_prod(const uint8_t* weights, const int8_t* activations, int m, int n) {
    std::vector<int32_t> result(m);
    for (int row = 0; row < m; row++)
        result[row] = naive_ternary_dot(weights + row * (n / 4), activations, n);
    return result;
}

std::vector<int32_t> naive_binary_matrix_vector_prod(const uint8_t* weights, const int8_t* activations, int m, int n) {
    std::vector<int32_t> result(m);
    for (int row = 0; row < m; row++)
        result[row] = naive_binary_dot(weights + row * (n / 8), activations, n);
    return result;
}
