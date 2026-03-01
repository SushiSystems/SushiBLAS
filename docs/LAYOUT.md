# SushiBLAS Operation Architecture Guide

This document defines the universal implementation pattern for all operations (BLAS, Math, Logic, Signal, etc.) in SushiBLAS.

## 1. Directory Structure Standards

- **Reference Kernels**: `kernels/reference/<category>/<sub-category>/<op>.hpp`
- **SYCL Kernels**: `kernels/sycl/<category>/<sub-category>/<op>.hpp`
- **Engine Logic**: `src/engine/<category>/<sub-category>/<op>.cpp`
- **Tests**: `tests/<category>/<sub-category>/test_<op>.cpp`

Categories include `blas`, `math`, `logic`, `signal`, etc.

## 2. Kernel Implementation Rules

### Reference Kernels
- **Purity**: Must be pure C++ with no external library dependencies (MKL, etc.).
- **Role**: Serves as the mathematical "Ground Truth".
- **Layout Support**: Must correctly handle all `Core::Layout` types (Row-Major/Column-Major).
- **Templates**: Must be templated by type `T` to support all `DataType` variants.

### SYCL Kernels
- **Abstraction**: Acts as the optimized hardware dispatcher.
- **Asynchrony**: Must return `sycl::event` and accept `const std::vector<sycl::event>& deps`.
- **Resources**: Uses the provided `sycl::queue` for all operations.
- **Hardware Limitations**: If a hardware backend (like oneMKL) does not support a specific `DataType` (e.g., `sycl::half` for SYRK), the SYCL kernel must throw a descriptive error or provide a fallback.

## 3. Engine Driver (Implementation)

The driver in `src/engine/` is the orchestrator between the SushiRuntime TaskGraph and the Kernels.

### Input Validation
Every operation must perform strict validation using `SB_THROW_IF`:
- Verify tensor ranks and dimensions.
- Check for data type (`DataType`) compatibility.
- Validate storage availability.

### Data Capture
- Safe capture of raw pointers from `Tensor` objects for asynchronous execution.
- Calculation of auxiliary parameters (strides, offsets, batch sizes).
- Setting operation-specific parameters in `SushiRuntime::Graph::TaskMetadata`.

### Execution Dispatch
The TaskGraph lambda must handle `ExecutionMode` logic:
- `FORCE_REFERENCE`: Wait for deps, run Reference kernel on CPU.
- `FORCE_SYCL` / `AUTO`: Run SYCL kernel on device.
- `VERIFY`: Run SYCL, wait for event, run Reference, and cross-check results.

## 4. Universal Testing Standards

Tests must ensure reliability across the entire library:
- **Typed Tests**: Every test suite must run for all supported types (`float`, `double`, `half`).
- **Precision Management**: Use `SB_ASSERT_TENSOR_NEAR` with type-appropriate tolerances.
- **Verification Mode**: Tests should primarily use `VERIFY` mode or manual comparison between `FORCE_SYCL` and `FORCE_REFERENCE`.
- **Comprehensive Coverage**: Test different shapes (square, rectangular, vector) and all supported memory layouts.
