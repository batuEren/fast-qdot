#pragma once
#include <cstdint>

int32_t naive_ternary_dot(const uint8_t* weights, const int8_t* activations, int n);
int32_t naive_binary_dot(const uint8_t* weights, const int8_t* activations, int n);
