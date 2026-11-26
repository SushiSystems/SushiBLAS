/**************************************************************************/
/* gemm_store.hpp                                                         */
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
 * @file gemm_store.hpp
 * @brief Declares GemmStore, the store that writes one element of C from its sum.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Stores `epilogue(alpha * sum + beta * C)` for each element of a packed
                 *        GEMM.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T, typename Epilogue>
                class GemmStore final
                {
                    public:
                        /** @brief Binds the scalars, the addressing of C and the epilogue. */
                        GemmStore(T alpha, T beta, OperandAccess c_access,
                                  Epilogue epilogue) noexcept
                            : alpha_(alpha), beta_(beta), c_access_(c_access), epilogue_(epilogue)
                        {

                        }

                        /** @brief Returns true: a GEMM writes every macro-block of C. */
                        [[nodiscard]] constexpr bool
                        wants([[maybe_unused]] int64_t row0, [[maybe_unused]] int64_t rows,
                              [[maybe_unused]] int64_t col0,
                              [[maybe_unused]] int64_t cols) const noexcept
                        {
                            return true;
                        }

                        /**
                         * @brief Writes element (row, col) of one matrix of C from its sum.
                         * @param c The first element of the matrix.
                         */
                        void operator()(T* c, int64_t row, int64_t col, Accumulator<T> sum) const
                        {
                            using W = Accumulator<T>;

                            const int64_t c_idx = c_access_.index(row, col);

                            // A zero beta must not read C, whose prior value may be uninitialised.
                            const T value = (beta_ == T(0))
                                ? T(multiply(W(alpha_), sum))
                                : T(multiply(W(alpha_), sum) + multiply(W(beta_), W(c[c_idx])));

                            c[c_idx] = epilogue_(value, row, col);
                        }

                    private:
                        /// @brief The scale of the product.
                        T alpha_;

                        /// @brief The scale of the prior C; exactly zero skips reading it.
                        T beta_;

                        /// @brief How element (row, col) of C is addressed.
                        OperandAccess c_access_;

                        /// @brief The functor applied to each element before it is stored.
                        Epilogue epilogue_;
                };
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
