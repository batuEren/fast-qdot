#include "lut.h"
#include <immintrin.h>
#include <cassert>
#include <array>
#include <vector>

constexpr int LUT_SIZE = 3;

template<typename T, std::size_t Size, std::size_t Capacity> // capacity is 3 for ternary and 2 for binary
struct LUT {
	using type = std::array<typename LUT<T, Size - 1, Capacity>::type, Capacity>;
};

template<typename T, std::size_t Capacity>
struct LUT<T, 0, Capacity> {
	using type = T;
};



template<int N>
constexpr auto lut_initializer() {
	typename LUT<uint8_t, N, 3>::type lut{};

	

	return 1;
}

constexpr auto helper(int depth, std::vector<uint8_t>& arr1, std::vector<uint8_t>& arr2)

constexpr int i = lut_initializer<3>();


int32_t lut_ternary_dot(const uint8_t* weights, const int8_t* activations, int n) {
	return 0;
}

int32_t lut_binary_dot(const uint8_t* weights, const int8_t* activations, int n) {
	return 0;
}