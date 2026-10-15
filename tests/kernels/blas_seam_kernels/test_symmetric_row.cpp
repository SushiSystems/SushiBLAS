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
 * @file test_symmetric_row.cpp
 * @brief Tests that SymmetricRow visits a row of a band or packed symmetric matrix in order.
 * @author Mustafa Garip
 */

#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <initializer_list>
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

            /// @brief Holds the orders the rows are walked at.
            constexpr int64_t K_ORDERS[] = {1, 2, 7};

            /// @brief Holds the band widths the rows are walked at.
            constexpr int64_t K_BANDS[] = {0, 1, 3};

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
        } // namespace

        /**
         * @brief Tests that a row of a packed symmetric or Hermitian matrix is visited from its
         *        first column to its last, each element equal to its indexed read.
         */
        TEST(Small_SymmetricRow, APackedRowIsVisitedInColumnOrder)
        {
            for (const Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
            {
                for (const Uplo uplo : {Uplo::Upper, Uplo::Lower})
                {
                    for (const int64_t n : K_ORDERS)
                    {
                        SCOPED_TRACE(::testing::Message()
                                     << (layout == Layout::RowMajor ? "row-major" : "column-major")
                                     << ", " << Core::to_string(uplo) << ", order " << n);

                        const std::vector<Complex> a = numbered(n * (n + 1) / 2);
                        expect_both_policies(
                            Kernels::packed_access(layout, uplo, Transpose::NoTrans, n),
                            a.data(), n);
                    }
                }
            }
        }

        /**
         * @brief Tests that a row of a band-stored symmetric or Hermitian matrix is visited
         *        from its first column inside the band to its last.
         */
        TEST(Small_SymmetricRow, ABandRowIsVisitedInColumnOrder)
        {
            for (const Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
            {
                for (const Uplo uplo : {Uplo::Upper, Uplo::Lower})
                {
                    for (const int64_t n : K_ORDERS)
                    {
                        for (const int64_t k : K_BANDS)
                        {
                            SCOPED_TRACE(::testing::Message()
                                         << (layout == Layout::RowMajor ? "row-major"
                                                                        : "column-major")
                                         << ", " << Core::to_string(uplo) << ", order " << n
                                         << ", band " << k);

                            const std::vector<Complex> a = numbered((k + 1) * n);
                            expect_both_policies(
                                Kernels::triangle_band_access(layout, uplo, Transpose::NoTrans,
                                                              k, k + 1),
                                a.data(), n);
                        }
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
