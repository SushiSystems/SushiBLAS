/**************************************************************************/
/* test_level2_reference.cpp                                              */
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
 * @file test_level2_reference.cpp
 * @brief Tests the BLAS Level 2 reference kernels against hand-computed results.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/trsm.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Fills the triangle a call must neither read nor write.
            constexpr float K_POISON = 999.0f;

            /** @brief Holds the triangle and the form of A a unit-diagonal solve runs on. */
            struct UnitSolveRow
            {
                const char* name;       /**< The row name a failure shows. */
                Core::Uplo uplo;        /**< The triangle of A that is read. */
                Core::Transpose trans;  /**< The form of A the solve uses. */
                std::vector<float> a;   /**< A, row-major, with the other triangle poisoned. */
            };

            /** @brief Returns U = [[1,2,3],[0,4,5],[0,0,6]] with its lower part poisoned. */
            std::vector<float> upper_matrix()
            {
                return {1.0f, 2.0f, 3.0f,
                        K_POISON, 4.0f, 5.0f,
                        K_POISON, K_POISON, 6.0f};
            }

            /** @brief Returns L = U^T with its upper part poisoned. */
            std::vector<float> lower_matrix()
            {
                return {1.0f, K_POISON, K_POISON,
                        2.0f, 4.0f, K_POISON,
                        3.0f, 5.0f, 6.0f};
            }

            /** @brief Returns the rows the reference solve is run on. */
            std::vector<UnitSolveRow> unit_solve_rows()
            {
                return {{"UpperNoTranspose", Core::Uplo::Upper, Core::Transpose::NoTrans, upper_matrix()},
                        {"LowerTransposed", Core::Uplo::Lower, Core::Transpose::Trans, lower_matrix()}};
            }

            /** @brief Returns the name of a row for GoogleTest. */
            std::string row_name(const ::testing::TestParamInfo<UnitSolveRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the unit-diagonal solve once per triangle and form of A. */
        class Small_Level2ReferenceUnitSolve : public ::testing::TestWithParam<UnitSolveRow>
        {

        };

        /**
         * @brief Tests that a solve with a unit diagonal ignores the stored diagonal and the
         *        other triangle, on integers the host solves exactly.
         *
         * Both rows solve (I + N) x = [9, 11, 2] to x = [1, 1, 2], where N is the strict triangle
         * [[0,2,3],[0,0,5],[0,0,0]], used as given or transposed.
         */
        TEST_P(Small_Level2ReferenceUnitSolve, TheStoredDiagonalIsNeverRead)
        {
            const UnitSolveRow& row = GetParam();
            const std::vector<float> want = {1.0f, 1.0f, 2.0f};
            std::vector<float> b = {9.0f, 11.0f, 2.0f};

            Kernels::Reference::trsm<float>(Core::Layout::RowMajor, Core::Side::Left, row.uplo,
                                            row.trans, Core::Diag::Unit, 3, 1, 1.0f, row.a.data(), 3,
                                            b.data(), 1);

            EXPECT_EQ(b, want);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_Level2ReferenceUnitSolve, ::testing::ValuesIn(unit_solve_rows()), row_name);
    } // namespace Test
} // namespace SushiBLAS
