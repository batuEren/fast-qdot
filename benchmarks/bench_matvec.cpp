#include <benchmark/benchmark.h>
#include <src/procedural_lut.h>
#include <cstdint>
#include <vector>
#include <random>

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

// ── Ternary matrix-vector benchmark ────────────────────────────────────────────

template<int N>
static void BM_PLutTernaryMatVec(benchmark::State& state) {
    int m      = (int)state.range(0);
    int raw_n  = (int)state.range(1);
    int n      = raw_n - (raw_n % N);
    auto a = make_activations(n);
    auto w = make_ternary_weights(m * n);
    for (auto _ : state)
        benchmark::DoNotOptimize(p_lut_ternary_matrix_vector_prod<N>(w.data(), a.data(), m, n));
    state.SetItemsProcessed(state.iterations() * (long long)m * n);
}

// ── Binary matrix-vector benchmark ─────────────────────────────────────────────

template<int N>
static void BM_PLutBinaryMatVec(benchmark::State& state) {
    int m      = (int)state.range(0);
    int raw_n  = (int)state.range(1);
    int n      = raw_n - (raw_n % N);
    auto a = make_activations(n);
    auto w = make_binary_weights(m * n);
    for (auto _ : state)
        benchmark::DoNotOptimize(p_lut_binary_matrix_vector_prod<N>(w.data(), a.data(), m, n));
    state.SetItemsProcessed(state.iterations() * (long long)m * n);
}

// ── Matrix sizes: (m=rows, n=cols) ─────────────────────────────────────────────
//    Small:  64 x  512
//    Medium: 128 x 1024
//    Large:  256 x 2048

#define MATVEC_ARGS \
    ->Args({64,  512}) \
    ->Args({128, 1024}) \
    ->Args({256, 2048})

// ── Ternary N=1..5 ─────────────────────────────────────────────────────────────

BENCHMARK_TEMPLATE(BM_PLutTernaryMatVec, 1) MATVEC_ARGS;
BENCHMARK_TEMPLATE(BM_PLutTernaryMatVec, 2) MATVEC_ARGS;
BENCHMARK_TEMPLATE(BM_PLutTernaryMatVec, 3) MATVEC_ARGS;
BENCHMARK_TEMPLATE(BM_PLutTernaryMatVec, 4) MATVEC_ARGS;
BENCHMARK_TEMPLATE(BM_PLutTernaryMatVec, 5) MATVEC_ARGS;

// ── Binary N=1..5 ──────────────────────────────────────────────────────────────

BENCHMARK_TEMPLATE(BM_PLutBinaryMatVec, 1) MATVEC_ARGS;
BENCHMARK_TEMPLATE(BM_PLutBinaryMatVec, 2) MATVEC_ARGS;
BENCHMARK_TEMPLATE(BM_PLutBinaryMatVec, 3) MATVEC_ARGS;
BENCHMARK_TEMPLATE(BM_PLutBinaryMatVec, 4) MATVEC_ARGS;
BENCHMARK_TEMPLATE(BM_PLutBinaryMatVec, 5) MATVEC_ARGS;
