/**************************************************************************/
/* test_trsv.cpp                                                          */
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
 * @file test_trsv.cpp
 * @brief Tests that the reference trsv recovers x from op(A) * x for every flag, layout and stride.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/trsv.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The order of the system every row solves.
        constexpr int64_t K_ORDER = 3;

        /// @brief The value the matrix holds outside the stored triangle; never read.
        constexpr double K_OUTSIDE = 99.0;

        /// @brief The value a unit-diagonal matrix holds on its diagonal; never read.
        constexpr double K_UNREAD_DIAGONAL = 7.0;

        /// @brief The value the vector holds between strided elements; the kernel never writes it.
        constexpr double K_GAP = 55.0;

        /// @brief The upper triangle, row by row; its transpose is K_LOWER.
        const std::vector<double> K_UPPER{2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 0.0, 0.0, 5.0};

        /// @brief The lower triangle, row by row; its transpose is K_UPPER.
        const std::vector<double> K_LOWER{2.0, 0.0, 0.0, 1.0, 4.0, 0.0, 3.0, 1.0, 5.0};

        /// @brief The solution every row expects.
        const std::vector<double> K_SOLUTION{1.0, 2.0, 3.0};

        /// @brief One solve: the flags, the stride and the right-hand side op(A) * K_SOLUTION.
        struct TrsvRow
        {
            Core::Layout layout;
            Core::Uplo uplo;
            Core::Transpose trans;
            Core::Diag diag;
            int64_t incx;
            std::vector<double> b;
        };

        /**
         * @brief Returns the 3 x 3 matrix a row stores: the triangle uplo names, K_OUTSIDE in the
         *        other one, K_UNREAD_DIAGONAL on a unit diagonal, laid out as layout says.
         */
        static std::vector<double> stored_matrix(Core::Layout layout, Core::Uplo uplo,
                                                 Core::Diag diag)
        {
            const bool upper = (uplo == Core::Uplo::Upper);
            const std::vector<double>& triangle = upper ? K_UPPER : K_LOWER;
            std::vector<double> stored(static_cast<std::size_t>(K_ORDER * K_ORDER));
            for (int64_t i = 0; i < K_ORDER; ++i)
            {
                for (int64_t j = 0; j < K_ORDER; ++j)
                {
                    const bool inside = upper ? (j >= i) : (j <= i);
                    double value = inside ? triangle[static_cast<std::size_t>(i * K_ORDER + j)]
                                          : K_OUTSIDE;
                    if (i == j && diag == Core::Diag::Unit)
                        value = K_UNREAD_DIAGONAL;
                    const int64_t offset =
                        (layout == Core::Layout::RowMajor) ? i * K_ORDER + j : j * K_ORDER + i;
                    stored[static_cast<std::size_t>(offset)] = value;
                }
            }
            return stored;
        }

        /** @brief Returns values spread over a vector of stride incx, with K_GAP between them. */
        static std::vector<double> spread(const std::vector<double>& values, int64_t incx)
        {
            std::vector<double> out(static_cast<std::size_t>((K_ORDER - 1) * incx + 1), K_GAP);
            for (int64_t i = 0; i < K_ORDER; ++i)
                out[static_cast<std::size_t>(i * incx)] = values[static_cast<std::size_t>(i)];
            return out;
        }

        /**
         * @brief Returns the rows. Each b is op(A) * {1, 2, 3} worked by hand; with a unit
         *        diagonal the diagonal counts as 1.
         */
        static std::vector<TrsvRow> trsv_rows()
        {
            using Core::Diag;
            using Core::Layout;
            using Core::Transpose;
            using Core::Uplo;
            return {
                // Non-unit diagonal: U x = {13, 11, 15}, L x = {2, 9, 20}.
                {Layout::RowMajor, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, 1,
                 {13.0, 11.0, 15.0}},
                {Layout::RowMajor, Uplo::Upper, Transpose::Trans, Diag::NonUnit, 1,
                 {2.0, 9.0, 20.0}},
                {Layout::RowMajor, Uplo::Upper, Transpose::ConjTrans, Diag::NonUnit, 1,
                 {2.0, 9.0, 20.0}},
                {Layout::RowMajor, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 1,
                 {2.0, 9.0, 20.0}},
                {Layout::RowMajor, Uplo::Lower, Transpose::Trans, Diag::NonUnit, 1,
                 {13.0, 11.0, 15.0}},
                {Layout::ColumnMajor, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, 1,
                 {13.0, 11.0, 15.0}},
                {Layout::ColumnMajor, Uplo::Upper, Transpose::Trans, Diag::NonUnit, 1,
                 {2.0, 9.0, 20.0}},
                {Layout::ColumnMajor, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 1,
                 {2.0, 9.0, 20.0}},
                {Layout::ColumnMajor, Uplo::Lower, Transpose::Trans, Diag::NonUnit, 1,
                 {13.0, 11.0, 15.0}},
                // Unit diagonal: U1 x = {12, 5, 3}, L1 x = {1, 3, 8}.
                {Layout::RowMajor, Uplo::Upper, Transpose::NoTrans, Diag::Unit, 1,
                 {12.0, 5.0, 3.0}},
                {Layout::RowMajor, Uplo::Upper, Transpose::Trans, Diag::Unit, 1,
                 {1.0, 3.0, 8.0}},
                {Layout::RowMajor, Uplo::Lower, Transpose::NoTrans, Diag::Unit, 1,
                 {1.0, 3.0, 8.0}},
                {Layout::RowMajor, Uplo::Lower, Transpose::Trans, Diag::Unit, 1,
                 {12.0, 5.0, 3.0}},
                {Layout::ColumnMajor, Uplo::Upper, Transpose::NoTrans, Diag::Unit, 1,
                 {12.0, 5.0, 3.0}},
                {Layout::ColumnMajor, Uplo::Upper, Transpose::Trans, Diag::Unit, 1,
                 {1.0, 3.0, 8.0}},
                {Layout::ColumnMajor, Uplo::Lower, Transpose::NoTrans, Diag::Unit, 1,
                 {1.0, 3.0, 8.0}},
                {Layout::ColumnMajor, Uplo::Lower, Transpose::Trans, Diag::Unit, 1,
                 {12.0, 5.0, 3.0}},
                // A stride other than one: the gaps between the elements stay untouched.
                {Layout::RowMajor, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, 2,
                 {13.0, 11.0, 15.0}},
                {Layout::ColumnMajor, Uplo::Lower, Transpose::Trans, Diag::Unit, 3,
                 {12.0, 5.0, 3.0}},
            };
        }

        /// @brief Fixture of the rows of trsv.
        class Small_ReferenceTrsv : public ::testing::TestWithParam<TrsvRow>
        {
        };

        TEST_P(Small_ReferenceTrsv, RecoversTheSolutionTheRightHandSideWasBuiltFrom)
        {
            const TrsvRow& row = GetParam();
            const std::vector<double> a = stored_matrix(row.layout, row.uplo, row.diag);
            std::vector<double> x = spread(row.b, row.incx);

            Kernels::Reference::trsv(row.layout, row.uplo, row.trans, row.diag, K_ORDER, a.data(),
                                     K_ORDER, x.data(), row.incx);

            EXPECT_EQ(x, spread(K_SOLUTION, row.incx));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReferenceTrsv, ::testing::ValuesIn(trsv_rows()));

        TEST(Small_ReferenceTrsvEmpty, LeavesTheVectorUntouchedForOrderZero)
        {
            const std::vector<double> a{K_OUTSIDE};
            std::vector<double> x{K_GAP};

            Kernels::Reference::trsv(Core::Layout::RowMajor, Core::Uplo::Upper,
                                     Core::Transpose::NoTrans, Core::Diag::NonUnit, 0, a.data(), 1,
                                     x.data(), 1);

            EXPECT_EQ(x, std::vector<double>{K_GAP});
        }
    } // namespace Test
} // namespace SushiBLAS
