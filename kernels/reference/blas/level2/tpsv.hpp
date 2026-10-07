/**************************************************************************/
/* tpsv.hpp                                                               */
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
 * @file tpsv.hpp
 * @brief Defines the reference kernel of TPSV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/kernels/blas/packed_access.hpp>
#include "kernels/reference/blas/level2/triangular_solve.hpp"

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes TPSV: solves op(A) * x = b in place for a triangular matrix A
             *        packed in ap, where x holds b on entry.
             * @see kernels/README.md
             */
            template<typename T>
            void tpsv(Core::Layout layout, Core::Uplo uplo, Core::Transpose trans, Core::Diag diag,
                      int64_t n, const T* ap, T* x, int64_t incx)
            {
                const PackedAccess access = packed_access(layout, uplo, trans, n);
                const bool unit_diag = (diag == Core::Diag::UNIT);
                Detail::triangular_solve(access, access.is_upper(), unit_diag, n, ap, x, incx);
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
