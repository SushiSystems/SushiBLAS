/**************************************************************************/
/* syr.hpp                                                                */
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
 * @file syr.hpp
 * @brief Defines the reference kernel of SYR and HER.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Computes SYR or HER, A += alpha * x * Conj(x)^T, in the triangle uplo names.
             * @tparam Conj Unconjugated (syr) or Conjugated (her, which stores a real diagonal).
             * @tparam Scalar The type of alpha: T for syr, the real component type of T for her.
             */
            template<typename T, typename Conj = Unconjugated, typename Scalar = T>
            void syr(Core::Layout layout, Core::Uplo uplo, int64_t n,
                     Scalar alpha, const T* x, int64_t incx,
                     T* a, int64_t lda)
            {
                static_assert(std::is_same_v<Scalar, T> || std::is_same_v<Scalar, RealType<T>>,
                              "alpha is a T for SYR and the real component type of T for HER.");

                using W = Accumulator<T>;
                using WS = Accumulator<Scalar>;

                auto A_ref = [&](int64_t r, int64_t c) -> T&
                {
                    return (layout == Core::Layout::RowMajor) ? a[r * lda + c] : a[r + c * lda];
                };
                const bool upper = (uplo == Core::Uplo::Upper);

                for (int64_t i = 0; i < n; ++i)
                {
                    int64_t jstart = upper ? i : 0;
                    int64_t jend = upper ? n : i + 1;
                    for (int64_t j = jstart; j < jend; ++j)
                    {
                        const W updated = W(A_ref(i, j)) +
                            WS(alpha) * W(x[i * incx]) * Conj::apply(W(x[j * incx]));
                        A_ref(i, j) = T((i == j) ? Conj::diagonal(updated) : updated);
                    }
                }
            }

        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
