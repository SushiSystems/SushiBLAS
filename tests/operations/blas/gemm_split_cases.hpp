/**************************************************************************/
/* gemm_split_cases.hpp                                                   */
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
 * @file gemm_split_cases.hpp
 * @brief Declares the products and the data the GEMM split tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/blas/gemm_split.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace GemmSplitCases
        {
            using Core::Layout;
            using Core::Transpose;
            using Dispatch::GemmGeometry;

            /** @brief Adds the position of an element to its value, so a wrong offset shows. */
            struct PositionEpilogue
            {
                /** @brief Returns value plus 100 per row and 1 per column. */
                float operator()(float value, int64_t row, int64_t column) const noexcept
                {
                    return value + static_cast<float>(100 * row + column);
                }
            };

            /** @brief Holds one product: its layout, its flags and its extents. */
            struct SplitCase
            {
                Layout layout;      /**< The layout of A, B and C. */
                Transpose trans_a;  /**< How A is read. */
                Transpose trans_b;  /**< How B is read. */
                int64_t m;          /**< Rows of C. */
                int64_t n;          /**< Columns of C. */
                int64_t k;          /**< The summed extent. */
            };

            /** @brief Returns the geometry of a contiguous, unbatched product. */
            [[nodiscard]] GemmGeometry geometry_of(const SplitCase& test_case)
            {
                const bool row_major = test_case.layout == Layout::RowMajor;
                const bool a_swapped = Core::is_transposed(test_case.trans_a);
                const bool b_swapped = Core::is_transposed(test_case.trans_b);
                GemmGeometry geometry;
                geometry.m = test_case.m;
                geometry.n = test_case.n;
                geometry.k = test_case.k;
                geometry.a.leading_dimension = row_major ? (a_swapped ? test_case.m : test_case.k)
                                                         : (a_swapped ? test_case.k : test_case.m);
                geometry.b.leading_dimension = row_major ? (b_swapped ? test_case.k : test_case.n)
                                                         : (b_swapped ? test_case.n : test_case.k);
                geometry.c.leading_dimension = row_major ? test_case.n : test_case.m;
                return geometry;
            }

            /** @brief Returns count small integers that differ from one element to the next. */
            [[nodiscard]] std::vector<float> values(std::size_t count, int seed)
            {
                std::vector<float> result(count);
                for (std::size_t i = 0; i < count; ++i)
                    result[i] = static_cast<float>((static_cast<int>(i) * 7 + seed) % 11 - 5);
                return result;
            }
        } // namespace GemmSplitCases
    } // namespace Test
} // namespace SushiBLAS
