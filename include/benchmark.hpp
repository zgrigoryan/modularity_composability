#pragma once
#include <string>
#include <vector>
#include <functional>
#include <cstddef>

struct BenchResult {
    std::string name;
    double ms;
    double gbytes_per_s;
    double gflops;
};

using Kernel = std::function<void()>;

BenchResult time_kernel(const std::string& name, Kernel fn, std::size_t bytes_moved, std::size_t flops);
void print_table(const std::vector<BenchResult>& results);
