/**************************************************************************/
/* trmv.hpp                                                               */
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
 * @file trmv.hpp
 * @brief Defines the reference kernel of TRMV.
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
             * @brief Computes TRMV, x = op(A)*x, in place without a copy of x.
             * @see kernels/README.md
             */
            template<typename T>
            void trmv(Core::Layout layout, bool upper, bool transA, bool unit_diag,
                      int64_t n, const T* a, int64_t lda, T* x, int64_t incx)
            {
                auto A_at = [&](int64_t r, int64_t c) -> T
                {
                    return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                };
                auto A_eff = [&](int64_t i, int64_t k) -> T
                {
                    return transA ? A_at(k, i) : A_at(i, k);
                };
                bool eff_upper = transA ? !upper : upper;

                if (eff_upper)
                {
                    for (int64_t i = 0; i < n; ++i)
                    {
                        T sum(0);
                        for (int64_t k = i; k < n; ++k)
                        {
                            T aval = (k == i && unit_diag) ? T(1) : A_eff(i, k);
                            sum += aval * x[k * incx];
                        }
                        x[i * incx] = sum;
                    }
                }
                else
                {
                    for (int64_t i = n - 1; i >= 0; --i)
                    {
                        T sum(0);
                        for (int64_t k = 0; k <= i; ++k)
                        {
                            T aval = (k == i && unit_diag) ? T(1) : A_eff(i, k);
                            sum += aval * x[k * incx];
                        }
                        x[i * incx] = sum;
                    }
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
