/**************************************************************************/
/* sparse_test_sweeps.hpp                                                 */
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
 * @file sparse_test_sweeps.hpp
 * @brief Declares the index-base sweeps the sparse operation tests run a check over, beside the
 *        forced-mode sweep of forced_mode_checks.hpp.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>
#include "support/forced_mode_checks.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Returns the name a failure message gives an index base. */
        inline std::string trace_of(Core::IndexBase base)
        {
            return base == Core::IndexBase::Zero ? "zero-based" : "one-based";
        }

        /** @brief Runs check once for each index base, under a trace that names the base. */
        template<typename Check>
        void in_each_index_base(Check&& check)
        {
            for (const Core::IndexBase base : K_BOTH_BASES)
            {
                SCOPED_TRACE(trace_of(base));
                check(base);
            }
        }

        /** @brief Returns the name a failure message gives a transposition. */
        inline std::string trace_of(Core::Transpose trans)
        {
            switch (trans)
            {
                case Core::Transpose::NoTrans:
                    return "no transpose";
                case Core::Transpose::Trans:
                    return "transpose";
                case Core::Transpose::ConjTrans:
                    return "conjugate transpose";
            }
            return "unknown transpose";
        }

        /** @brief Returns the name a failure message gives a triangle. */
        inline std::string trace_of(Core::Uplo uplo)
        {
            return uplo == Core::Uplo::Lower ? "lower triangle" : "upper triangle";
        }

        /// @brief The three values of Core::Transpose.
        inline constexpr Core::Transpose K_EVERY_TRANSPOSE[3] = {
            Core::Transpose::NoTrans, Core::Transpose::Trans, Core::Transpose::ConjTrans};

        /// @brief The two values of Core::Transpose that leave a matrix unconjugated.
        inline constexpr Core::Transpose K_PLAIN_AND_TRANSPOSED[2] = {Core::Transpose::NoTrans,
                                                                      Core::Transpose::Trans};

        /// @brief The two triangles a one-triangle product reads.
        inline constexpr Core::Uplo K_BOTH_TRIANGLES[2] = {Core::Uplo::Lower, Core::Uplo::Upper};

        /** @brief Runs check once for each of the three transpositions, under a trace that names it. */
        template<typename Check>
        void in_each_transpose(Check&& check)
        {
            for (const Core::Transpose trans : K_EVERY_TRANSPOSE)
            {
                SCOPED_TRACE(trace_of(trans));
                check(trans);
            }
        }

        /** @brief Runs check once for NoTrans and once for Trans, under a trace that names it. */
        template<typename Check>
        void in_plain_and_transposed(Check&& check)
        {
            for (const Core::Transpose trans : K_PLAIN_AND_TRANSPOSED)
            {
                SCOPED_TRACE(trace_of(trans));
                check(trans);
            }
        }

        /** @brief Runs check once for each triangle, under a trace that names it. */
        template<typename Check>
        void in_each_triangle(Check&& check)
        {
            for (const Core::Uplo uplo : K_BOTH_TRIANGLES)
            {
                SCOPED_TRACE(trace_of(uplo));
                check(uplo);
            }
        }

        /**
         * @brief Runs check once for each forced mode and index base, under traces that name them.
         * @tparam Check Callable `(ExecutionMode, Core::IndexBase) -> void`.
         */
        template<typename Check>
        void in_each_mode_and_base(Check&& check)
        {
            in_each_forced_mode([&](ExecutionMode mode) {
                in_each_index_base([&](Core::IndexBase base) { check(mode, base); });
            });
        }
    } // namespace Test
} // namespace SushiBLAS
