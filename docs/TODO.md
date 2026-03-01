# SushiBLAS Development Roadmap (TODO)

## 1. Benchmarking and Performance Analysis - *If you can't measure it, you can't improve it*
- [ ] **Google Benchmark Integration:** Create micro-benchmarks for all operations (GEMM, Sum, FFT, etc.).
- [ ] **Theoretical Limit Comparison:** Calculate kernel efficiency based on hardware GFLOPS capacity.
- [ ] **Baseline Comparison:** Report performance differences compared to "gold standard" libraries like Intel MKL, cuBLAS, and oneDNN.

## 2. Automation and Continuous Integration (CI/CD)
- [ ] **GitHub Actions Integration:** Automate builds for Windows and Linux after every `push` and `PR`.
- [ ] **Automated Test Execution:** Ensure the `gtest` suite passes automatically on all platforms.
- [ ] **Linter & Format Check:** Use CI to check if code follows `STYLE_GUIDE.md`.

## 3. Type Dispatch Automation (Meta-Programming)
- [ ] **Universal Dispatcher:** Remove `switch (t.dtype)` blocks. Use a template-based Type Registry system instead.
- [ ] **Extended Data Type Support:** Add support for `BFloat16`, `Int8`, and `Complex` types without repeating code.

## 4. Observability and Profiling
- [ ] **Chrome Tracing Support:** Add a feature to export operation timelines in `.json` format.
- [ ] **Hardware Profiler Integration:** Integrate kernel naming and annotations (like ITT API) for Intel VTune and NVIDIA Nsight.

## 5. Deployment and Integration (Deployment)
- [ ] **CMake Export:** Add `install(EXPORT ...)` commands so other projects can use `find_package(SushiBLAS)`.
- [ ] **Header-Only Wrapper Option:** Evaluate a header-only layer for fast integration when needed.
