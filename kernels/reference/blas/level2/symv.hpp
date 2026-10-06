/**************************************************************************/
/* symv.hpp                                                               */
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
 * @file symv.hpp
 * @brief Defines the reference kernel of SYMV.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
            * @brief Pure C++ reference SYMV: y = alpha*A*x + beta*y, A symmetric
            * with only the requested triangle physically read.
            */
            template<typename T>
            void symv(Core::Layout layout, bool upper, int64_t n,
                      T alpha, const T* a, int64_t lda,
                      const T* x, int64_t incx,
                      T beta, T* y, int64_t incy)
            {
                auto A_at = [&](int64_t r, int64_t c) -> T
                {
                    return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                };
                auto sym_at = [&](int64_t i, int64_t j) -> T
                {
                    if (upper) return (i <= j) ? A_at(i, j) : A_at(j, i);
                    return (i >= j) ? A_at(i, j) : A_at(j, i);
                };

                for (int64_t i = 0; i < n; ++i)
                {
                    T sum(0);
                    for (int64_t j = 0; j < n; ++j) sum += sym_at(i, j) * x[j * incx];
                    y[i * incy] = (beta == T(0)) ? alpha * sum : alpha * sum + beta * y[i * incy];
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
