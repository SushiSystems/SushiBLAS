cmake_minimum_required(VERSION 3.20)

# --- Define SushiBLAS Library ---
add_library(sushiblas STATIC
    # Engine
    engine.cpp

    # IO & Utilities
    ops/io.cpp

    # BLAS: Level 1
    ops/blas/level1/axpy.cpp
    ops/blas/level1/asum.cpp
    ops/blas/level1/copy.cpp
    ops/blas/level1/dot.cpp
    ops/blas/level1/iamax.cpp
    ops/blas/level1/nrm2.cpp
    ops/blas/level1/rot.cpp
    ops/blas/level1/scal.cpp
    ops/blas/level1/swap.cpp

    # BLAS: Level 2
    ops/blas/level2/gemv.cpp
    ops/blas/level2/ger.cpp
    ops/blas/level2/symv.cpp
    ops/blas/level2/trmv.cpp
    ops/blas/level2/trsv.cpp
    ops/blas/level2/syr.cpp
    ops/blas/level2/syr2.cpp

    # BLAS: Level 3
    ops/blas/level3/gemm.cpp
    ops/blas/level3/syrk.cpp
    ops/blas/level3/trsm.cpp

    # Math: Random
    # ops/math/random/random.cpp

    # Math: Nonlinear
    ops/math/nonlinear/relu.cpp
    ops/math/nonlinear/leaky_relu.cpp
    ops/math/nonlinear/sigmoid.cpp
    ops/math/nonlinear/tanh.cpp
    ops/math/nonlinear/elu.cpp
    ops/math/nonlinear/silu.cpp
    ops/math/nonlinear/gelu.cpp
    ops/math/nonlinear/softplus.cpp
)


# --- Include Directories ---
target_include_directories(sushiblas PUBLIC 
    "${CMAKE_SOURCE_DIR}/include"
)

# --- Link Directories ---
target_link_directories(sushiblas PUBLIC "${CMAKE_SOURCE_DIR}/lib")

# --- Link Libraries ---
# Link against dependencies. PUBLIC so consumers (like the app) inherit them.
target_link_libraries(sushiblas PUBLIC
    sushiruntime
    IntelSYCL::SYCL_CXX
    OpenCL::OpenCL
)

# --- Properties ---
set_target_properties(sushiblas PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Use oneMKL through compiler flags (Recommended for IntelLLVM)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    if(WIN32)
        target_compile_options(sushiblas PUBLIC "/Qmkl")
        target_link_options(sushiblas PUBLIC "/Qmkl")
    else()
        target_compile_options(sushiblas PUBLIC "-qmkl")
        target_link_options(sushiblas PUBLIC "-qmkl")
    endif()
endif()


