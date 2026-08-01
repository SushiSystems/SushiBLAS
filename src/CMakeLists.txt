cmake_minimum_required(VERSION 3.20)

# --- Define SushiBLAS Library ---
add_library(sushiblas STATIC
    # Engine
    engine.cpp

    # IO & Utilities
    io.cpp

    # BLAS: Level 1
    # engine/blas/level1/axpy.cpp
    # engine/blas/level1/asum.cpp
    # engine/blas/level1/copy.cpp
    # engine/blas/level1/dot.cpp
    # engine/blas/level1/iamax.cpp
    # engine/blas/level1/nrm2.cpp
    # engine/blas/level1/rot.cpp
    # engine/blas/level1/scal.cpp
    # engine/blas/level1/swap.cpp

    # BLAS: Level 2
    # engine/blas/level2/gemv.cpp
    # engine/blas/level2/ger.cpp
    # engine/blas/level2/symv.cpp
    # engine/blas/level2/trmv.cpp
    # engine/blas/level2/trsv.cpp
    # engine/blas/level2/syr.cpp
    # engine/blas/level2/syr2.cpp

    # BLAS: Level 3
    engine/blas/level3/gemm.cpp
    engine/blas/level3/syrk.cpp
    # engine/blas/level3/trsm.cpp

    # Math: Random
    # engine/math/random/bernoulli.cpp
    # engine/math/random/constant.cpp
    # engine/math/random/discrete_uniform.cpp
    # engine/math/random/exponential.cpp
    # engine/math/random/he_normal.cpp
    # engine/math/random/he_uniform.cpp
    # engine/math/random/log_normal.cpp
    # engine/math/random/normal.cpp
    # engine/math/random/orthogonal.cpp
    # engine/math/random/poisson.cpp
    # engine/math/random/set_seed.cpp
    # engine/math/random/shuffle.cpp
    # engine/math/random/truncated_normal.cpp
    # engine/math/random/uniform.cpp
    # engine/math/random/xavier_normal.cpp
    # engine/math/random/xavier_uniform.cpp

    # Math: Nonlinear
    # engine/math/nonlinear/relu.cpp
    # engine/math/nonlinear/leaky_relu.cpp
    # engine/math/nonlinear/sigmoid.cpp
    # engine/math/nonlinear/tanh.cpp
    # engine/math/nonlinear/elu.cpp
    # engine/math/nonlinear/silu.cpp
    # engine/math/nonlinear/gelu.cpp
    # engine/math/nonlinear/softplus.cpp

    # Math: Elementwise
    # engine/math/elementwise/add.cpp
    # engine/math/elementwise/sub.cpp
    # engine/math/elementwise/mul.cpp
    # engine/math/elementwise/div.cpp
    # engine/math/elementwise/sqrt.cpp
    # engine/math/elementwise/square.cpp
    # engine/math/elementwise/reciprocal.cpp
    # engine/math/elementwise/neg.cpp
    # engine/math/elementwise/exp.cpp
    # engine/math/elementwise/log.cpp
    # engine/math/elementwise/abs.cpp
    # engine/math/elementwise/pow.cpp
    # engine/math/elementwise/sin.cpp
    # engine/math/elementwise/cos.cpp
    # engine/math/elementwise/tan.cpp
    # engine/math/elementwise/asin.cpp
    # engine/math/elementwise/acos.cpp
    # engine/math/elementwise/atan.cpp
    # engine/math/elementwise/sinh.cpp
    # engine/math/elementwise/cosh.cpp
    # engine/math/elementwise/asinh.cpp
    # engine/math/elementwise/acosh.cpp
    # engine/math/elementwise/atanh.cpp
    # engine/math/elementwise/ceil.cpp
    # engine/math/elementwise/floor.cpp
    # engine/math/elementwise/round.cpp
    # engine/math/elementwise/clamp.cpp
    # engine/math/elementwise/min.cpp
    # engine/math/elementwise/max.cpp
    # engine/math/elementwise/fmod.cpp
    # engine/math/elementwise/remainder.cpp

    # Math: Reductions
    # engine/math/reductions/sum.cpp
    # engine/math/reductions/mean.cpp
    # engine/math/reductions/max.cpp
    # engine/math/reductions/min.cpp
    # engine/math/reductions/argmax.cpp
    # engine/math/reductions/argmin.cpp
    # engine/math/reductions/var.cpp
    # engine/math/reductions/std.cpp
    # engine/math/reductions/prod.cpp
    # engine/math/reductions/logsumexp.cpp
    # engine/math/reductions/asum.cpp
    # engine/math/reductions/nrm2.cpp

    # Logic
    # engine/logic/equal.cpp
    # engine/logic/not_equal.cpp
    # engine/logic/greater.cpp
    # engine/logic/less.cpp
    # engine/logic/greater_equal.cpp
    # engine/logic/less_equal.cpp
    # engine/logic/logical_and.cpp
    # engine/logic/logical_or.cpp
    # engine/logic/logical_xor.cpp
    # engine/logic/logical_not.cpp
    # engine/logic/where.cpp
    # engine/logic/all.cpp
    # engine/logic/any.cpp
)


# --- Include Directories ---
target_include_directories(sushiblas PUBLIC 
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}"
)

# --- Link Libraries ---
# Link against dependencies. PUBLIC so consumers (like the app) inherit them.
# SushiRuntime is always linked via its namespaced alias (D3): whichever of the
# three resolution modes in cmake/Runtime.cmake fired, the link line here is
# identical, and it never varies with how the dependency was actually found.
target_link_libraries(sushiblas PUBLIC
    SushiRuntime::SushiRuntime
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


