#pragma once
#include <cstdint>

// n must be a multiple of 32
int32_t mad_ternary_dot_avx2(const uint8_t* weights, const int8_t* activations, int n);
int32_t mad_binary_dot_avx2 (const uint8_t* weights, const int8_t* activations, int n);
