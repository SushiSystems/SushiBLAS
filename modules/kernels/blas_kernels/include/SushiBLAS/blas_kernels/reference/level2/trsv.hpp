/**************************************************************************/
/* trsv.hpp                                                               */
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
 * @file trsv.hpp
 * @brief Defines the reference kernel of TRSV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/triangular_solve.hpp>
#include <SushiBLAS/blas_kernels/triangular_solve_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes TRSV: solves op(A) * x = b in place for a dense triangular A, where
             *        x holds b on entry. Only the stored triangle of A is read.
             * @see kernels/README.md
             */
            template<typename T>
            void trsv(Core::Layout layout, Core::Uplo uplo, Core::Transpose trans, Core::Diag diag,
                      int64_t n, const T* a, int64_t lda, T* x, int64_t incx)
            {
                const bool op_upper = Indexing::effective_upper(Core::is_transposed(trans),
                                                                uplo == Core::Uplo::Upper);
                Detail::triangular_solve(operand_access(layout, trans, lda), op_upper,
                                         diag == Core::Diag::Unit, n, a, x, incx);
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
