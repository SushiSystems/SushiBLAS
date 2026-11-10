/**************************************************************************/
/* gemm_seam_fixtures.hpp                                                 */
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
 * @file gemm_seam_fixtures.hpp
 * @brief Declares the epilogues and the operand data the GEMM seam tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief A per-column bias add, the shape a fused Linear layer uses. */
        struct BiasColumnEpilogue
        {
            /** @brief One value per column of C, broadcast down its rows. */
            const float* bias;

            /**
             * @brief Adds the bias of the column to value.
             * @return value + bias[col].
             */
            template<typename T>
            [[nodiscard]] T operator()(T value, int64_t, int64_t col) const noexcept
            {
                return value + static_cast<T>(bias[col]);
            }
        };

        /** @brief Adds a bias and also writes its result to an auxiliary buffer. */
        struct BiasSpillEpilogue
        {
            /** @brief One value per column of C. */
            const float* bias;

            /** @brief Row-major spill buffer, the same extents as C. */
            float* aux;

            /** @brief Column count of C, so the epilogue can address aux. */
            int64_t columns;

            /**
             * @brief Adds the bias, spills the result and returns it.
             * @return value + bias[col].
             */
            template<typename T>
            T operator()(T value, int64_t row, int64_t col) const
            {
                const T pre = value + static_cast<T>(bias[col]);
                aux[row * columns + col] = static_cast<float>(pre);
                return pre;
            }
        };

        /** @brief Holds the operands and the hand-computed result of the fused seam tests. */
        class GemmSeamData
        {
            protected:
                /** @brief A = 3x2, chosen so A*B is square and not symmetric. */
                static std::vector<float> a_values() { return {1, 0, 0, 1, 1, 1}; }

                /** @brief B = 2x3. */
                static std::vector<float> b_values() { return {1, 2, 3, 4, 5, 6}; }

                /** @brief Returns the hand-computed result of the fused call. */
                static std::vector<float> expected()
                {
                    return {2, 0, 6,
                            0, 0, 3,
                            11, 4, 17};
                }
        };
    } // namespace Test
} // namespace SushiBLAS
