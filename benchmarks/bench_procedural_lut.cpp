#include <benchmark/benchmark.h>
#include <src/procedural_lut.h>
#include <cstdint>
#include <vector>
#include <array>
#include <random>

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

// ── LUT creation only ─────────────────────────────────────────────────────────

template<int N>
static void BM_PLutTernaryLutOnly(benchmark::State& state) {
    int raw_n = (int)state.range(0);
    int n = raw_n - (raw_n % N);
    auto a = make_activations(raw_n);
    for (auto _ : state)
        benchmark::DoNotOptimize(create_ternary_lut<N>(a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}

template<int N>
static void BM_PLutBinaryLutOnly(benchmark::State& state) {
    int raw_n = (int)state.range(0);
    int n = raw_n - (raw_n % N);
    auto a = make_activations(raw_n);
    for (auto _ : state)
        benchmark::DoNotOptimize(create_binary_lut<N>(a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}

// ── Dot product only (LUT pre-built outside the timed region) ─────────────────

template<int N>
static void BM_PLutTernaryDotOnly(benchmark::State& state) {
    int raw_n = (int)state.range(0);
    int n = raw_n - (raw_n % N);
    auto w = make_ternary_weights(raw_n);
    auto a = make_activations(raw_n);
    auto lut = create_ternary_lut<N>(a.data(), n);
    for (auto _ : state)
        benchmark::DoNotOptimize(p_lut_ternary_dot<N>(w.data(), lut, n));
    state.SetItemsProcessed(state.iterations() * n);
}

template<int N>
static void BM_PLutBinaryDotOnly(benchmark::State& state) {
    int raw_n = (int)state.range(0);
    int n = raw_n - (raw_n % N);
    auto w = make_binary_weights(raw_n);
    auto a = make_activations(raw_n);
    auto lut = create_binary_lut<N>(a.data(), n);
    for (auto _ : state)
        benchmark::DoNotOptimize(p_lut_binary_dot<N>(w.data(), lut, n));
    state.SetItemsProcessed(state.iterations() * n);
}

// ── Ternary: LUT build only ────────────────────────────────────────────────────

BENCHMARK_TEMPLATE(BM_PLutTernaryLutOnly, 1)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutTernaryLutOnly, 2)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutTernaryLutOnly, 3)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutTernaryLutOnly, 4)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutTernaryLutOnly, 5)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutTernaryLutOnly, 6)->Arg(4096)->Arg(8192);

// ── Ternary: dot-product only ─────────────────────────────────────────────────

BENCHMARK_TEMPLATE(BM_PLutTernaryDotOnly, 1)->Arg(4096)->Arg(8192)->MinTime(2.0);
BENCHMARK_TEMPLATE(BM_PLutTernaryDotOnly, 2)->Arg(4096)->Arg(8192)->MinTime(2.0);
BENCHMARK_TEMPLATE(BM_PLutTernaryDotOnly, 3)->Arg(4096)->Arg(8192)->MinTime(2.0);
BENCHMARK_TEMPLATE(BM_PLutTernaryDotOnly, 4)->Arg(4096)->Arg(8192)->MinTime(2.0);
BENCHMARK_TEMPLATE(BM_PLutTernaryDotOnly, 5)->Arg(4096)->Arg(8192)->MinTime(2.0);
BENCHMARK_TEMPLATE(BM_PLutTernaryDotOnly, 6)->Arg(4096)->Arg(8192)->MinTime(2.0);

// ── Binary: LUT build only ────────────────────────────────────────────────────

BENCHMARK_TEMPLATE(BM_PLutBinaryLutOnly, 1)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutBinaryLutOnly, 2)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutBinaryLutOnly, 3)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutBinaryLutOnly, 4)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutBinaryLutOnly, 5)->Arg(4096)->Arg(8192);
BENCHMARK_TEMPLATE(BM_PLutBinaryLutOnly, 6)->Arg(4096)->Arg(8192);

// ── Binary: dot-product only ──────────────────────────────────────────────────

BENCHMARK_TEMPLATE(BM_PLutBinaryDotOnly, 1)->Arg(4096)->Arg(8192)->MinTime(2.0);
BENCHMARK_TEMPLATE(BM_PLutBinaryDotOnly, 2)->Arg(4096)->Arg(8192)->MinTime(2.0);
BENCHMARK_TEMPLATE(BM_PLutBinaryDotOnly, 3)->Arg(4096)->Arg(8192)->MinTime(2.0);
BENCHMARK_TEMPLATE(BM_PLutBinaryDotOnly, 4)->Arg(4096)->Arg(8192)->MinTime(2.0);
BENCHMARK_TEMPLATE(BM_PLutBinaryDotOnly, 5)->Arg(4096)->Arg(8192)->MinTime(2.0);
BENCHMARK_TEMPLATE(BM_PLutBinaryDotOnly, 6)->Arg(4096)->Arg(8192)->MinTime(2.0);
