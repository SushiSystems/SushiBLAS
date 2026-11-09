/**************************************************************************/
/* memory_store.hpp                                                       */
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
 * @file memory_store.hpp
 * @brief Declares the store of the packed product that keeps each sum in a row-major array.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/arithmetic/accumulator.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Writes each sum of a packed product into a row-major C and counts the
                 *        writes, for the macro-blocks that start in the first rows.
                 * @see modules/kernels/blas_seam_kernels/README.md
                 */
                template<typename T>
                class MemoryStore final
                {
                    public:
                        /**
                         * @param width Columns of C.
                         * @param calls One counter per element of C.
                         * @param wanted_rows The macro-blocks that start below this row are
                         *        written; the others are skipped.
                         */
                        MemoryStore(int64_t width, unsigned* calls, int64_t wanted_rows) noexcept
                            : width_(width), calls_(calls), wanted_rows_(wanted_rows)
                        {

                        }

                        /** @brief Returns true when the macro-block starts below wanted_rows. */
                        [[nodiscard]] bool wants(int64_t row0,
                                                 [[maybe_unused]] int64_t rows,
                                                 [[maybe_unused]] int64_t col0,
                                                 [[maybe_unused]] int64_t cols) const noexcept
                        {
                            return row0 < wanted_rows_;
                        }

                        /** @brief Stores sum at (row, col) of the matrix at c and counts it. */
                        void operator()(T* c, int64_t row, int64_t col, Accumulator<T> sum) const
                        {
                            c[row * width_ + col] = T(sum);
                            calls_[row * width_ + col] += 1;
                        }

                    private:
                        /// @brief Columns of C.
                        int64_t width_;

                        /// @brief One counter per element of C.
                        unsigned* calls_;

                        /// @brief The first row no macro-block may start at.
                        int64_t wanted_rows_;
                };
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
