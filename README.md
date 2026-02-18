# SushiBLAS

SushiBLAS is a high-performance Basic Linear Algebra Subprograms (BLAS) library. It is designed to work as a mathematics layer for SushiStack.

## Features
- Tensor engine with a high-level C++ API.
- Integrated with SushiRuntime for hardware management and execution.
- Zero-copy data sharing using SYCL Unified Shared Memory (USM).
- Object-oriented engine interface.
- Memory alignment and NUMA topology awareness.

## Architecture
- **Engine Layer**: The main interface for BLAS operations.
- **Tensor Objects**: Metadata containers for managing shapes and data ownership.
- **Storage Layer**: Memory management via USM allocators provided by SushiRuntime.

## Project Status
SushiBLAS is currently under development. Internal APIs and features are subject to change.

## Requirements
- Intel oneAPI (DPC++ Compiler)
- SushiRuntime (pre-compiled binaries included in lib folder)
- HWLOC

## License
This project is licensed under the MIT License. See the LICENSE file for details.
