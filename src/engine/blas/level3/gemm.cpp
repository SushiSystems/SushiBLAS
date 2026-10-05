/****************************************************************/
/* gemm.cpp                                                     */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas/dispatch.hpp>
#include <SushiBLAS/engine/blas/level3.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // Plain GEMM is the fused GEMM with nothing fused into it, so it is spelled that
    // way rather than as a second copy of the shape derivation, the access set, the
    // dtype switch and the ExecutionMode triad. GemmIdentityEpilogue inlines to its
    // argument, so the kernel this records is byte for byte the one this file used to
    // record directly - see <SushiBLAS/engine/blas/dispatch.hpp>.
    sycl::event Level3::gemm(const Tensor& A, const Tensor& B, Tensor& C,
                            bool transA, bool transB,
                            double alpha, double beta)
    {
        // REAL_AND_COMPLEX, explicitly: the seam defaults to REAL because a
        // caller-written epilogue usually is, but plain GEMM has supported all five
        // dtypes since SB-2 and the identity epilogue is well-formed for every one.
        return Dispatch::execute_gemm<Dispatch::Domain::REAL_AND_COMPLEX>(
            engine_, A, B, C, transA, transB, alpha, beta,
            "blas.lvl3.gemm", "blas.lvl3.gemm"_op,
            Kernels::GemmIdentityEpilogue{});
    }
} // namespace SushiBLAS
