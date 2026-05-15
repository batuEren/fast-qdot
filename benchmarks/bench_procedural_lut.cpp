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

// ── Benchmark fixtures ─────────────────────────────────────────────────────────

template<int N>
static void BM_PLutTernary(benchmark::State& state) {
    int raw_n = (int)state.range(0);
    int n = raw_n - (raw_n % N);
    auto w = make_ternary_weights(raw_n);
    auto a = make_activations(raw_n);
    for (auto _ : state)
        benchmark::DoNotOptimize(p_lut_ternary_dot<N>(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}

template<int N>
static void BM_PLutBinary(benchmark::State& state) {
    int raw_n = (int)state.range(0);
    int n = raw_n - (raw_n % N);
    auto w = make_binary_weights(raw_n);
    auto a = make_activations(raw_n);
    for (auto _ : state)
        benchmark::DoNotOptimize(p_lut_binary_dot<N>(w.data(), a.data(), n));
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
