#pragma once
#include <cstdint>

int32_t mad_ternary_dot(const uint8_t* weights, const int8_t* activations, int n);
int32_t mad_binary_dot (const uint8_t* weights, const int8_t* activations, int n);
