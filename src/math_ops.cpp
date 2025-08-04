#include "math_ops.hpp"

namespace ops {

void vec_mul(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

void vec_add(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void vec_separate_mul_add(const float* a, const float* b, const float* c, float* out, std::size_t n) {
    // pass 1: multiply into out
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
    // pass 2: add c
    for (std::size_t i = 0; i < n; ++i) out[i] = out[i] + c[i];
}

void vec_fused_mul_add(const float* a, const float* b, const float* c, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = std::fma(a[i], b[i], c[i]);
}

} // namespace ops
