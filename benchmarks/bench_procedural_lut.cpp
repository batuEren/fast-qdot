#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>
#include <array>
#include <random>

// Templated procedural LUT ternary dot product.
// Weights: one byte per element in {0,1,2} mapping to {-1, 0, +1}.
// LUT index is base-3 encoded over a group of N consecutive weights.
template<int N>
static int32_t p_lut_ternary_dot_n(const uint8_t* weights, const int8_t* activations, int n) {
    constexpr int LUT_SIZE = []() {
        int r = 1; for (int i = 0; i < N; i++) r *= 3; return r;
    }();

    std::vector<std::array<int16_t, LUT_SIZE>> lut(n / N);
    for (int i = 0; i < n / N; i++) {
        for (int j = 0; j < LUT_SIZE; j++) {
            int rem = j;
            int16_t total = 0;
            for (int k = 0; k < N; k++) {
                total = (int16_t)(total + (rem % 3 - 1) * activations[i * N + k]);
                rem /= 3;
            }
            lut[i][j] = total;
        }
    }

    int32_t result = 0;
    for (int i = 0; i < n / N; i++) {
        int idx = 0, pow3 = 1;
        for (int j = 0; j < N; j++) { idx += weights[i * N + j] * pow3; pow3 *= 3; }
        result += lut[i][idx];
    }
    return result;
}

// Templated procedural LUT binary dot product.
// Weights: one byte per element in {0,1} mapping to {-1, +1}.
// LUT index is base-2 encoded over a group of N consecutive weights.
template<int N>
static int32_t p_lut_binary_dot_n(const uint8_t* weights, const int8_t* activations, int n) {
    constexpr int LUT_SIZE = 1 << N;

    std::vector<std::array<int16_t, LUT_SIZE>> lut(n / N);
    for (int i = 0; i < n / N; i++) {
        for (int j = 0; j < LUT_SIZE; j++) {
            int16_t total = 0;
            for (int k = 0; k < N; k++)
                total = (int16_t)(total + (((j >> k) & 1) ? +1 : -1) * activations[i * N + k]);
            lut[i][j] = total;
        }
    }

    int32_t result = 0;
    for (int i = 0; i < n / N; i++) {
        int idx = 0;
        for (int j = 0; j < N; j++) idx |= (weights[i * N + j] & 1) << j;
        result += lut[i][idx];
    }
    return result;
}

// ── Data generators ────────────────────────────────────────────────────────────

static std::vector<int8_t> make_activations(int n) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-127, 127);
    std::vector<int8_t> v(n);
    for (auto& x : v) x = (int8_t)dist(rng);
    return v;
}

static std::vector<uint8_t> make_ternary_weights(int n) {
    std::mt19937 rng(13);
    std::uniform_int_distribution<int> dist(0, 2);
    std::vector<uint8_t> v(n);
    for (auto& x : v) x = (uint8_t)dist(rng);
    return v;
}

static std::vector<uint8_t> make_binary_weights(int n) {
    std::mt19937 rng(17);
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<uint8_t> v(n);
    for (auto& x : v) x = (uint8_t)dist(rng);
    return v;
}

// ── Benchmark fixtures ─────────────────────────────────────────────────────────

template<int N>
static void BM_PLutTernary(benchmark::State& state) {
    int raw_n = (int)state.range(0);
    int n = raw_n - (raw_n % N);
    auto w = make_ternary_weights(raw_n);
    auto a = make_activations(raw_n);
    for (auto _ : state)
        benchmark::DoNotOptimize(p_lut_ternary_dot_n<N>(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}

template<int N>
static void BM_PLutBinary(benchmark::State& state) {
    int raw_n = (int)state.range(0);
    int n = raw_n - (raw_n % N);
    auto w = make_binary_weights(raw_n);
    auto a = make_activations(raw_n);
    for (auto _ : state)
        benchmark::DoNotOptimize(p_lut_binary_dot_n<N>(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}

// ── Ternary N=1..6 ─────────────────────────────────────────────────────────────

BENCHMARK_TEMPLATE(BM_PLutTernary, 1)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutTernary, 2)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutTernary, 3)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutTernary, 4)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutTernary, 5)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutTernary, 6)->Arg(4096)->Arg(8192);

// ── Binary N=1..6 ──────────────────────────────────────────────────────────────

BENCHMARK_TEMPLATE(BM_PLutBinary, 1)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutBinary, 2)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutBinary, 3)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutBinary, 4)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutBinary, 5)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutBinary, 6)->Arg(4096)->Arg(8192);
