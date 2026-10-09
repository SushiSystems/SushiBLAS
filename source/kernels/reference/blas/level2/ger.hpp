/**************************************************************************/
/* ger.hpp                                                                */
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
 * @file ger.hpp
 * @brief Defines the reference kernel of GER, GERU and GERC.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/kernels/blas/conjugation.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes the rank-1 update A += alpha * x * Conj(y)^T.
             * @tparam Conj Unconjugated for ger and geru, Conjugated for gerc.
             */
            template<typename T, typename Conj = Unconjugated>
            void ger(Core::Layout layout, int64_t m, int64_t n,
                     T alpha, const T* x, int64_t incx,
                     const T* y, int64_t incy,
                     T* a, int64_t lda)
            {
                auto A_ref = [&](int64_t r, int64_t c) -> T&
                {
                    return (layout == Core::Layout::RowMajor) ? a[r * lda + c] : a[r + c * lda];
                };

                using W = Accumulator<T>;
                for (int64_t i = 0; i < m; ++i)
                    for (int64_t j = 0; j < n; ++j)
                        A_ref(i, j) = T(W(A_ref(i, j)) +
                                        W(alpha) * W(x[i * incx]) * Conj::apply(W(y[j * incy])));
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
