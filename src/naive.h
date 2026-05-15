#pragma once
#include <cstdint>
#include <vector>

int32_t naive_ternary_dot(const uint8_t* weights, const int8_t* activations, int n);
int32_t naive_binary_dot(const uint8_t* weights, const int8_t* activations, int n);

std::vector<int32_t> naive_ternary_matrix_vector_prod(const uint8_t* weights, const int8_t* activations, int m, int n);
std::vector<int32_t> naive_binary_matrix_vector_prod(const uint8_t* weights, const int8_t* activations, int m, int n);
