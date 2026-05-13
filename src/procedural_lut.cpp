#include "procedural_lut.h"
#include <array>

template<int N>
uint8_t* create_lut(const uint8_t* weights, const int8_t* activations, int n) {
	std::array<std::array<uint8_t, 3^N>, n / N> lut; // lut[c][k], c = index, k = corresponding weight combination

	for(size_t i = 0; i < n / N; i++) {
		for(size_t j = 0; j < 3^N; j++) {
			lut[i][j] = 0;
		}
	}

	int weightIdx = 0;


}
