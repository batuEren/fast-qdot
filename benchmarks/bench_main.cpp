#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>
#include <random>
#include "src/naive.h"
#include "src/mad.h"
#include "src/mad_avx2.h"
#include "src/lut.h"
#include "src/procedural_lut.h"

// Helpers

static std::vector<int8_t> make_activations(int n) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-127, 127);
    std::vector<int8_t> v(n);
    for (auto& x : v) x = (int8_t)dist(rng);
    return v;
}

// Ternary weights: 2 bits per weight, 4 per byte → n/4 bytes
static std::vector<uint8_t> make_ternary_weights(int n) {
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> v(n / 4);
    for (auto& x : v) x = (uint8_t)dist(rng);
    return v;
}

// Binary weights: 1 bit per weight, 8 per byte → n/8 bytes
static std::vector<uint8_t> make_binary_weights(int n) {
    std::mt19937 rng(77);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> v(n / 8);
    for (auto& x : v) x = (uint8_t)dist(rng);
    return v;
}

// LUT ternary weights: nibble-packed pair-indices, 2 pairs per byte → n/4 bytes.
// Each nibble is a valid pair-index in {0..8}: 3*(c_even+1) + (c_odd+1).
static std::vector<uint8_t> make_lut_ternary_weights(int n) {
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> dist(0, 8);
    std::vector<uint8_t> v(n / 4);
    for (auto& x : v) x = (uint8_t)((dist(rng) << 4) | dist(rng));
    return v;
}

// Procedural ternary weights: one byte per weight, value in {0,1,2} → n bytes.
static std::vector<uint8_t> make_p_ternary_weights(int n) {
    std::mt19937 rng(13);
    std::uniform_int_distribution<int> dist(0, 2);
    std::vector<uint8_t> v(n);
    for (auto& x : v) x = (uint8_t)dist(rng);
    return v;
}

// Procedural binary weights: one byte per weight, value in {0,1} → n bytes.
static std::vector<uint8_t> make_p_binary_weights(int n) {
    std::mt19937 rng(17);
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<uint8_t> v(n);
    for (auto& x : v) x = (uint8_t)dist(rng);
    return v;
}

// LUT binary weights: 2-bit pair-indices, 4 pairs per byte → n/8 bytes.
// Each 2-bit field encodes (w_even==+1)*2 + (w_odd==+1) ∈ {0..3}.
// Any random byte already contains valid indices, so reuse make_binary_weights layout.
static std::vector<uint8_t> make_lut_binary_weights(int n) {
    std::mt19937 rng(55);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> v(n / 8);
    for (auto& x : v) x = (uint8_t)dist(rng);
    return v;
}

// naive_ternary_dot

static void BM_NaiveTernary(benchmark::State& state) {
    int n = (int)state.range(0);
    auto w = make_ternary_weights(n);
    auto a = make_activations(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(naive_ternary_dot(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_NaiveTernary)->RangeMultiplier(2)->Range(64, 8192);

// mad_ternary_dot

static void BM_MadTernary(benchmark::State& state) {
    int n = (int)state.range(0);
    auto w = make_ternary_weights(n);
    auto a = make_activations(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(mad_ternary_dot(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}
// mad_ternary requires n % 16 == 0; smallest power-of-2 that qualifies is 64
BENCHMARK(BM_MadTernary)->RangeMultiplier(2)->Range(64, 8192);

// naive_binary_dot

static void BM_NaiveBinary(benchmark::State& state) {
    int n = (int)state.range(0);
    auto w = make_binary_weights(n);
    auto a = make_activations(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(naive_binary_dot(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_NaiveBinary)->RangeMultiplier(2)->Range(64, 8192);

// mad_binary_dot

static void BM_MadBinary(benchmark::State& state) {
    int n = (int)state.range(0);
    auto w = make_binary_weights(n);
    auto a = make_activations(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(mad_binary_dot(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_MadBinary)->RangeMultiplier(2)->Range(64, 8192);

// mad_ternary_dot_avx2

static void BM_MadTernaryAVX2(benchmark::State& state) {
    int n = (int)state.range(0);
    auto w = make_ternary_weights(n);
    auto a = make_activations(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(mad_ternary_dot_avx2(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}
// mad_ternary_dot_avx2 requires n % 32 == 0; smallest power-of-2 that qualifies is 64
BENCHMARK(BM_MadTernaryAVX2)->RangeMultiplier(2)->Range(64, 8192);

// mad_binary_dot_avx2

static void BM_MadBinaryAVX2(benchmark::State& state) {
    int n = (int)state.range(0);
    auto w = make_binary_weights(n);
    auto a = make_activations(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(mad_binary_dot_avx2(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}
// mad_binary_dot_avx2 requires n % 32 == 0; smallest power-of-2 that qualifies is 64
BENCHMARK(BM_MadBinaryAVX2)->RangeMultiplier(2)->Range(64, 8192);

// lut_ternary_dot

static void BM_LutTernary(benchmark::State& state) {
    int n = (int)state.range(0);
    auto w = make_lut_ternary_weights(n);
    auto a = make_activations(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(lut_ternary_dot(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_LutTernary)->RangeMultiplier(2)->Range(64, 8192);

// p_lut_ternary_dot — n must be a multiple of 3; use 96*2^k series

static void BM_PLutTernary(benchmark::State& state) {
    int n = (int)state.range(0);
    auto w = make_p_ternary_weights(n);
    auto a = make_activations(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(p_lut_ternary_dot<3>(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_PLutTernary)->RangeMultiplier(2)->Range(96, 6144);

// p_lut_binary_dot

static void BM_PLutBinary(benchmark::State& state) {
    int n = (int)state.range(0);
    auto w = make_p_binary_weights(n);
    auto a = make_activations(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(p_lut_binary_dot<4>(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_PLutBinary)->RangeMultiplier(2)->Range(64, 8192);

// lut_binary_dot

static void BM_LutBinary(benchmark::State& state) {
    int n = (int)state.range(0);
    auto w = make_lut_binary_weights(n);
    auto a = make_activations(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(lut_binary_dot(w.data(), a.data(), n));
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_LutBinary)->RangeMultiplier(2)->Range(64, 8192);
