#pragma once
#include <cstddef>
#include <cmath>

namespace ops {

// Scalar ops (demonstrates logical modularity)
inline float multiply(float a, float b) { return a * b; }
inline float add(float a, float b)      { return a + b; }

// Fused multiply-add; typically maps to an FMA instruction
inline float fused_mul_add(float a, float b, float c) { return std::fma(a, b, c); }

// Vector kernels
void vec_mul(const float* a, const float* b, float* out, std::size_t n);
void vec_add(const float* a, const float* b, float* out, std::size_t n);

// Two-pass: out = a*b + c (mul pass, then add pass)
void vec_separate_mul_add(const float* a, const float* b, const float* c, float* out, std::size_t n);

// One-pass fused: out = fma(a, b, c)
void vec_fused_mul_add(const float* a, const float* b, const float* c, float* out, std::size_t n);

} // namespace ops
