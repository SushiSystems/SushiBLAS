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
SushiBLAS is currently a personal project in active development. To maintain architecture consistency and performance, external Pull Requests are not being accepted at this time.

## Contributing & Style
If you find bugs or have suggestions, please open an [Issue](https://github.com/SushiSystems/SushiBLAS/issues). You can find more details in our [Contributing Policy](docs/CONTRIBUTING.md) and [Style Guide](docs/STYLE_GUIDE.md).


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

