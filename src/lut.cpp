#include "lut.h"
#include <immintrin.h>
#include <cassert>
#include <array>

constexpr int LUT_SIZE = 3;

template<int N>
constexpr auto lut_initializer() {
	std::array<uint8_t, N> LUT{};
}


int32_t lut_ternary_dot(const uint8_t* weights, const int8_t* activations, int n) {
	return 0;
}

int32_t lut_binary_dot(const uint8_t* weights, const int8_t* activations, int n) {
	return 0;
}