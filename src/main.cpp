#include "math_ops.hpp"
#include "benchmark.hpp"
#include <vector>
#include <random>
#include <numeric>
#include <iostream>
#include <cstring>
#include <cmath>

struct Args {
    std::size_t n = 50'000'000;  // 50M elements
    int iters = 1;               // repetitions for stability
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) a.n = std::stoull(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) a.iters = std::stoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--help")) {
            std::cout << "Usage: modfuse [--n N] [--iters K]\n";
            std::exit(0);
        }
    }
    return a;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    const std::size_t n = args.n;
    const int iters = args.iters;

    std::vector<float> a(n), b(n), c(n), out(n), tmp(n);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
        c[i] = dist(rng);
    }

    // Traffic and flop estimates per element
    const std::size_t bytes_per_float = sizeof(float);
    const std::size_t bytes_separate = n * 6 * bytes_per_float; // two passes (3 + 3 floats)
    const std::size_t bytes_fused    = n * 4 * bytes_per_float; // one pass  (a,b,c,out)
    const std::size_t flops_muladd   = n * 2; // 1 mul + 1 add

    std::vector<BenchResult> results;

    // Warm-up
    ops::vec_fused_mul_add(a.data(), b.data(), c.data(), out.data(), n);

    // Separate
    {
        auto run = [&](){
            for (int k = 0; k < iters; ++k) ops::vec_separate_mul_add(a.data(), b.data(), c.data(), out.data(), n);
            volatile float sink = std::accumulate(out.begin(), out.end(), 0.0f);
            (void)sink;
        };
        results.push_back(time_kernel("separate mul+add", run, bytes_separate * iters, flops_muladd * iters));
    }

    // Fused (FMA)
    {
        auto run = [&](){
            for (int k = 0; k < iters; ++k) ops::vec_fused_mul_add(a.data(), b.data(), c.data(), out.data(), n);
            volatile float sink = std::accumulate(out.begin(), out.end(), 0.0f);
            (void)sink;
        };
        results.push_back(time_kernel("fused mul-add (FMA)", run, bytes_fused * iters, flops_muladd * iters));
    }

    // Individual ops (optional completeness)
    {
        auto run = [&](){
            for (int k = 0; k < iters; ++k) ops::vec_mul(a.data(), b.data(), tmp.data(), n);
            volatile float sink = std::accumulate(tmp.begin(), tmp.end(), 0.0f);
            (void)sink;
        };
        results.push_back(time_kernel("mul only", run, n * 3 * bytes_per_float * iters, n * 1 * iters));
    }
    {
        auto run = [&](){
            for (int k = 0; k < iters; ++k) ops::vec_add(a.data(), b.data(), tmp.data(), n);
            volatile float sink = std::accumulate(tmp.begin(), tmp.end(), 0.0f);
            (void)sink;
        };
        results.push_back(time_kernel("add only", run, n * 3 * bytes_per_float * iters, n * 1 * iters));
    }

    print_table(results);

    // Correctness check: separate vs fused (expect tiny diff)
    ops::vec_separate_mul_add(a.data(), b.data(), c.data(), tmp.data(), n);
    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = std::abs(static_cast<double>(tmp[i]) - static_cast<double>(out[i]));
        if (d > max_abs_err) max_abs_err = d;
    }
    std::cout << "\nmax |separate - fused| = " << max_abs_err << "\n";
    return 0;
}
