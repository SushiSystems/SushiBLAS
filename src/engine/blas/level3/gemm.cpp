/****************************************************************/
/* gemm.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file gemm.cpp
 * @brief Defines Level3::gemm, which records the BLAS GEMM task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/dispatch.hpp>
#include <SushiBLAS/engine/blas/level3.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // Plain GEMM is the fused GEMM with the identity epilogue; see src/README.md.
    sycl::event Level3::gemm(const Tensor& A, const Tensor& B, Tensor& C,
                            bool transA, bool transB,
                            double alpha, double beta)
    {
        // REAL_AND_COMPLEX is named because the seam defaults to REAL; see src/README.md.
        return Dispatch::execute_gemm<Dispatch::Domain::REAL_AND_COMPLEX>(
            engine_, A, B, C, transA, transB, alpha, beta,
            "blas.lvl3.gemm", "blas.lvl3.gemm"_op,
            Kernels::GemmIdentityEpilogue{});
    }
} // namespace SushiBLAS
