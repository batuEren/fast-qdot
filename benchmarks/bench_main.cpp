#include <benchmark/benchmark.h>

static void BM_Placeholder(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(1 + 1);
    }
}
BENCHMARK(BM_Placeholder);
