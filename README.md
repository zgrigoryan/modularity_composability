# Modularity & Fused Multiply-Add (FMA) 

This repo completes the following:

1. Code is decomposed into logical modules (`ops` for math, `benchmark` for timing) and physical modules (headers in `include/`, sources in `src/`), with a small CLI in `main.cpp`.
2. Compare a *two‑pass* `mul`+`add` pipeline vs a *one‑pass* fused `fma` for `out = a*b + c` on CPU. Measure runtime, effective GB/s, and GFLOP/s.

## Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

## Run

```bash
./modfuse --n 50000000 --iters 1
# Flags:
#   --n N        number of elements (default 50,000,000)
#   --iters K    repetitions per kernel (default 1)
```

### Output (example)
```
Kernel                       Time (ms)          GB/s       GFLOP/s
------------------------------------------------------------------
separate mul+add                61.026         19.66          1.64
fused mul-add (FMA)             57.455         13.92          1.74
mul only                        64.414          9.31          0.78
add only                        54.142         11.08          0.92

max |separate - fused| = 0.00
```
#### Reading the output

- Time: fused is faster (57.5 ms vs 61.0 ms) → ~6% speedup.
- GFLOP/s: fused is higher (1.74 > 1.64) — expected.
- GB/s: fused is lower (13.9 < 19.7) — also expected here because our GB/s metric is “effective bytes moved per algorithm.” The fused kernel moves 16 B/elem vs 24 B/elem for the two-pass version, so even if it’s faster, it reports fewer GB/s because it did less memory traffic. The win is lower traffic and lower time.

**Interpretation.** The fused kernel performs the same arithmetic in **one pass** (fewer bytes moved: 16 B/elem vs 24 B/elem), so it’s typically faster and achieves higher effective bandwidth and FLOP rate. Minor numeric differences are expected because FMA does the multiply and add with a single rounding step (often slightly more accurate).

## Notes
- `std::fma` encourages use of the hardware FMA instruction when available.
- Build with `-march=native -O3` (set in CMake) to enable vectorization/FMA locally.
- For deeper analysis, profile with `perf`, VTune, or `-fopt-info-vec` (GCC/Clang).
