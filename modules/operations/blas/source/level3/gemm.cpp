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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file gemm.cpp
 * @brief Defines Level3::gemm, which records the BLAS GEMM task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level3.hpp>

#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    // Plain GEMM is the fused GEMM with the identity epilogue; see modules/operations/README.md.
    TaskHandle Level3::gemm(const Tensor& A, const Tensor& B, Tensor& C,
                            Core::Transpose transA, Core::Transpose transB,
                            BLASScalar alpha, BLASScalar beta)
    {
        // REAL_AND_COMPLEX is named because the seam defaults to REAL; see modules/operations/README.md.
        return Dispatch::execute_gemm<Dispatch::Domain::RealAndComplex>(
            recorder_, A, B, C, transA, transB, alpha, beta,
            "blas.lvl3.gemm", "blas.lvl3.gemm"_op,
            Kernels::GemmIdentityEpilogue{});
    }
} // namespace SushiBLAS
