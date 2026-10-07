/**************************************************************************/
/* gemm.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file gemm.cpp
 * @brief Defines Level3::gemm, which records the BLAS GEMM task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/engine/blas/dispatch.hpp>
#include <SushiBLAS/engine/blas/level3.hpp>
#include <SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // Plain GEMM is the fused GEMM with the identity epilogue; see src/README.md.
    TaskHandle Level3::gemm(const Tensor& A, const Tensor& B, Tensor& C,
                            bool transA, bool transB,
                            double alpha, double beta)
    {
        // REAL_AND_COMPLEX is named because the seam defaults to REAL; see src/README.md.
        return Dispatch::execute_gemm<Dispatch::Domain::REAL_AND_COMPLEX>(
            recorder_, A, B, C, transA, transB, alpha, beta,
            "blas.lvl3.gemm", "blas.lvl3.gemm"_op,
            Kernels::GemmIdentityEpilogue{});
    }
} // namespace SushiBLAS
