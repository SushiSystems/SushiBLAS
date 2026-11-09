/**************************************************************************/
/* test_symmetric_row.cpp                                                 */
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
 * @file test_symmetric_row.cpp
 * @brief Tests that SymmetricRow visits a row of a band or packed symmetric matrix in order.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/band_access.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/blas_seam_kernels/symmetric_row.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::Layout;
            using Core::Transpose;
            using Core::Uplo;
            using Complex = std::complex<double>;

            /** @brief Returns storage whose element at offset p is (p + 1, -(p + 1)). */
            [[nodiscard]] std::vector<Complex> numbered(int64_t elements)
            {
                std::vector<Complex> storage(static_cast<std::size_t>(elements));
                for (int64_t p = 0; p < elements; ++p)
                    storage[static_cast<std::size_t>(p)] = Complex(double(p + 1), -double(p + 1));
                return storage;
            }

            /** @brief Returns element (r, c) of the full matrix by one indexed read. */
            template<typename Conj, typename Access>
            [[nodiscard]] Complex indexed(const Access& access, const Complex* a, int64_t r,
                                          int64_t c)
            {
                if (r == c)
                    return Conj::diagonal(access.at(a, r, r));
                return access.holds(r, c) ? access.at(a, r, c) : Conj::apply(access.at(a, c, r));
            }

            /** @brief Expects every row of the matrix to be visited in order, by value. */
            template<typename Conj, typename Access>
            void expect_rows_in_order(const Access& access, const Complex* a, int64_t n)
            {
                const Access mirror = access.transposed();
                const Kernels::SymmetricRow<Conj, Access> rows =
                    Kernels::symmetric_row<Conj>(access, n);

                for (int64_t i = 0; i < n; ++i)
                {
                    const int64_t first =
                        std::min(access.first_column(i), mirror.first_column(i));
                    const int64_t end =
                        std::max(access.end_column(i, n), mirror.end_column(i, n));
                    EXPECT_EQ(first, rows.first_column(i)) << "row " << i;
                    EXPECT_EQ(end, rows.end_column(i)) << "row " << i;

                    int64_t expected = first;
                    rows.for_each(a, i, [&](Complex value, int64_t j)
                    {
                        EXPECT_EQ(expected, j) << "row " << i;
                        EXPECT_EQ(indexed<Conj>(access, a, i, j), value)
                            << "row " << i << ", column " << j;
                        ++expected;
                    });
                    EXPECT_EQ(end, expected) << "row " << i;
                }
            }

            /** @brief Runs the symmetric and the Hermitian walk over one stored triangle. */
            template<typename Access>
            void expect_both_policies(const Access& access, const Complex* a, int64_t n)
            {
                expect_rows_in_order<Kernels::Unconjugated>(access, a, n);
                expect_rows_in_order<Kernels::Conjugated>(access, a, n);
            }

            /** @brief Holds one storage, order and band width the rows are walked at. */
            struct RowCase
            {
                Layout layout; /**< The storage order of the lines. */
                Uplo uplo;     /**< The stored triangle. */
                int64_t order; /**< The order of the matrix. */
                int64_t band;  /**< The band width; unused by the packed storage. */
            };

            /** @brief Returns both layouts and triangles crossed with the orders and bands. */
            [[nodiscard]] std::vector<RowCase> row_cases(std::vector<int64_t> orders,
                                                         std::vector<int64_t> bands)
            {
                std::vector<RowCase> cases;
                for (const Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
                    for (const Uplo uplo : {Uplo::Upper, Uplo::Lower})
                        for (const int64_t n : orders)
                            for (const int64_t k : bands)
                                cases.push_back({layout, uplo, n, k});
                return cases;
            }

            /** @brief Names a case for the test name. */
            std::string row_case_name(const ::testing::TestParamInfo<RowCase>& info)
            {
                const RowCase& row = info.param;
                return std::string(row.layout == Layout::RowMajor ? "RowMajor" : "ColumnMajor") +
                       (row.uplo == Uplo::Upper ? "Upper" : "Lower") + "Order" +
                       std::to_string(row.order) + "Band" + std::to_string(row.band);
            }
        } // namespace

        /** @brief Walks the rows of a packed symmetric matrix once per case. */
        class Small_SymmetricRowPacked : public ::testing::TestWithParam<RowCase>
        {

        };

        /**
         * @brief Tests that a row of a packed symmetric or Hermitian matrix is visited from its
         *        first column to its last, each element equal to its indexed read.
         */
        TEST_P(Small_SymmetricRowPacked, APackedRowIsVisitedInColumnOrder)
        {
            const RowCase& row = GetParam();
            const std::vector<Complex> a = numbered(row.order * (row.order + 1) / 2);

            expect_both_policies(
                Kernels::packed_access(row.layout, row.uplo, Transpose::NoTrans, row.order),
                a.data(), row.order);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SymmetricRowPacked,
                                 ::testing::ValuesIn(row_cases({1, 2, 7}, {0})), row_case_name);

        /** @brief Walks the rows of a band-stored symmetric matrix once per case. */
        class Small_SymmetricRowBand : public ::testing::TestWithParam<RowCase>
        {

        };

        /**
         * @brief Tests that a row of a band-stored symmetric or Hermitian matrix is visited
         *        from its first column inside the band to its last.
         */
        TEST_P(Small_SymmetricRowBand, ABandRowIsVisitedInColumnOrder)
        {
            const RowCase& row = GetParam();
            const std::vector<Complex> a = numbered((row.band + 1) * row.order);

            expect_both_policies(
                Kernels::triangle_band_access(row.layout, row.uplo, Transpose::NoTrans, row.band,
                                              row.band + 1),
                a.data(), row.order);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SymmetricRowBand,
                                 ::testing::ValuesIn(row_cases({1, 2, 7}, {0, 1, 3})),
                                 row_case_name);
    } // namespace Test
} // namespace SushiBLAS
