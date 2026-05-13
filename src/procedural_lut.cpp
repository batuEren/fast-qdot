#include "procedural_lut.h"
#include <array>
#include <vector>

template<int SIZE, int N>
uint8_t* create_lut(const int8_t* activations, int n) {
	std::array<std::array<uint8_t, 3^N>, n / N> lut; // lut[c][k], c = index, k = corresponding weight combination

	for(size_t i = 0; i < n / N; i++) {
		lut_helper_ternary<N>()
		for(int j = 0; j < 3^N; j++) {
			//get correct weights in an array
			int rem = j;
			std::array<int, N> weightsTaken;
			for (int k = 0; k < N; k++) {
				weightsTaken[k] = (rem % N) - 1; // (0, 1, 2) -> (-1, 0, 1)
				rem /= N;
			}

			for (int k = 0; k < N; k++) {

			}

			lut[i][j] =
		}
	}

	int weightIdx = 0;


}

template<int N>
void lut_helper_ternary(auto& lut, auto& activations, int n, int idx, std::vector<int>& weightsTaken) {
	if (weightsTaken.size() == N - 1) {
		for (int i = -1; i <= 1; i++) {
			for (auto& i : weightsTaken) {
			
			}
			weightsTaken.push_back(i);
		}
	}
}