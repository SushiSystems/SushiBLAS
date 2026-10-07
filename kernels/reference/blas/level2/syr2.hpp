/**************************************************************************/
/* syr2.hpp                                                               */
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
 * @file syr2.hpp
 * @brief Defines the reference kernel of SYR2 and HER2.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/kernels/blas/conjugation.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            namespace Detail
            {
                /** @brief Returns the SYR2 term alpha * (x_i * y_j + y_i * x_j). */
                template<typename W>
                [[nodiscard]] inline W rank2_term(Unconjugated, W alpha, W xi, W yj, W yi, W xj)
                {
                    return alpha * (xi * yj + yi * xj);
                }

                /** @brief Returns the HER2 term, which conjugates y_j, x_j and the second alpha. */
                template<typename W>
                [[nodiscard]] inline W rank2_term(Conjugated, W alpha, W xi, W yj, W yi, W xj)
                {
                    return alpha * xi * Conjugated::apply(yj) +
                           Conjugated::apply(alpha) * yi * Conjugated::apply(xj);
                }
            } // namespace Detail

            /**
             * @brief Computes SYR2 or HER2, A(i, j) += rank2_term, in the triangle uplo names.
             * @tparam Conj Unconjugated (syr2) or Conjugated (her2, which stores a real diagonal).
             */
            template<typename T, typename Conj = Unconjugated>
            void syr2(Core::Layout layout, Core::Uplo uplo, int64_t n,
                      T alpha, const T* x, int64_t incx,
                      const T* y, int64_t incy,
                      T* a, int64_t lda)
            {
                using W = Accumulator<T>;

                auto A_ref = [&](int64_t r, int64_t c) -> T&
                {
                    return (layout == Core::Layout::ROW_MAJOR) ? a[r * lda + c] : a[r + c * lda];
                };
                const bool upper = (uplo == Core::Uplo::UPPER);

                for (int64_t i = 0; i < n; ++i)
                {
                    int64_t jstart = upper ? i : 0;
                    int64_t jend = upper ? n : i + 1;
                    for (int64_t j = jstart; j < jend; ++j)
                    {
                        const W term = Detail::rank2_term(Conj{}, W(alpha),
                                                          W(x[i * incx]), W(y[j * incy]),
                                                          W(y[i * incy]), W(x[j * incx]));
                        const W updated = W(A_ref(i, j)) + term;
                        A_ref(i, j) = T((i == j) ? Conj::diagonal(updated) : updated);
                    }
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
