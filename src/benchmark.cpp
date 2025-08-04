#include "benchmark.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

BenchResult time_kernel(const std::string& name, Kernel fn, std::size_t bytes_moved, std::size_t flops) {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    fn();
    auto t1 = clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double seconds = ms / 1000.0;
    double gbytes_per_s = seconds > 0 ? (bytes_moved / 1e9) / seconds : 0.0;
    double gflops       = seconds > 0 ? (flops / 1e9) / seconds : 0.0;

    return BenchResult{name, ms, gbytes_per_s, gflops};
}

void print_table(const std::vector<BenchResult>& results) {
    std::cout << std::left << std::setw(26) << "Kernel"
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(14) << "GB/s"
              << std::setw(14) << "GFLOP/s"
              << "\n";
    std::cout << std::string(66, '-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(26) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.ms
                  << std::setw(14) << std::setprecision(2) << r.gbytes_per_s
                  << std::setw(14) << std::setprecision(2) << r.gflops
                  << "\n";
    }
}
