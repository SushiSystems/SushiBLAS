/**************************************************************************/
/* test_band_access.cpp                                                   */
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
 * @file test_band_access.cpp
 * @brief Tests the offsets, the band test and the column ranges of BandAccess.
 * @author Mustafa Garip
 */

#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/band_access.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::Layout;
            using Core::Transpose;
            using Core::Uplo;

            /// @brief Holds the two band storages every case runs under.
            constexpr Layout K_LAYOUTS[] = {Layout::ColumnMajor, Layout::RowMajor};

            /// @brief Holds the rows of the worked matrix of the design, section 2.1.
            constexpr int64_t K_WORKED_ROWS = 4;

            /// @brief Holds the columns of the worked matrix.
            constexpr int64_t K_WORKED_COLS = 5;

            /// @brief Holds the sub-diagonals of the worked matrix.
            constexpr int64_t K_WORKED_LOWER = 1;

            /// @brief Holds the super-diagonals of the worked matrix.
            constexpr int64_t K_WORKED_UPPER = 2;

            /// @brief Holds the elements per stored line of the worked matrix.
            constexpr int64_t K_WORKED_LINE = 4;

            /// @brief Holds the elements of the worked matrix its band stores.
            constexpr int K_WORKED_STORED = 14;

            /// @brief Holds the value of a slot that stores no element of the band.
            constexpr double K_POISON = 999.0;

            /// @brief Holds the worked matrix as CBLAS stores it by columns, shape (4, 5).
            constexpr double K_WORKED_COLUMN_MAJOR[] = {
                K_POISON, K_POISON, 11.0, 21.0,
                K_POISON, 12.0, 22.0, 32.0,
                13.0, 23.0, 33.0, 43.0,
                24.0, 34.0, 44.0, K_POISON,
                35.0, 45.0, K_POISON, K_POISON};

            /// @brief Holds the worked matrix as CBLAS stores it by rows, shape (4, 4).
            constexpr double K_WORKED_ROW_MAJOR[] = {
                K_POISON, 11.0, 12.0, 13.0,
                21.0, 22.0, 23.0, 24.0,
                32.0, 33.0, 34.0, 35.0,
                43.0, 44.0, 45.0, K_POISON};

            /** @brief Holds one band shape the column ranges are scanned for. */
            struct ScannedBand
            {
                int64_t rows;  /**< The rows of the matrix. */
                int64_t cols;  /**< The columns of the matrix. */
                int64_t lower; /**< The sub-diagonals, kl. */
                int64_t upper; /**< The super-diagonals, ku. */
            };

            /** @brief Holds where one triangle band storage puts the diagonal of a line. */
            struct DiagonalLine
            {
                Layout layout;  /**< The band storage. */
                Uplo uplo;      /**< The stored triangle. */
                int64_t offset; /**< The expected offset of the diagonal inside its line. */
            };

            /** @brief Returns the name of a layout for a failure message. */
            [[nodiscard]] constexpr const char* layout_name(Layout layout) noexcept
            {
                return (layout == Layout::RowMajor) ? "row-major" : "column-major";
            }

            /** @brief Returns the band storage of the worked matrix under a layout. */
            [[nodiscard]] constexpr const double* worked_storage(Layout layout) noexcept
            {
                return (layout == Layout::ColumnMajor) ? K_WORKED_COLUMN_MAJOR
                                                        : K_WORKED_ROW_MAJOR;
            }

            /** @brief Returns how many slots the worked matrix occupies under a layout. */
            [[nodiscard]] constexpr int64_t worked_slots(Layout layout) noexcept
            {
                return K_WORKED_LINE *
                       ((layout == Layout::ColumnMajor) ? K_WORKED_COLS : K_WORKED_ROWS);
            }

            /** @brief Returns element (i, j) of the worked matrix, which names its own place. */
            [[nodiscard]] constexpr double worked_element(int64_t i, int64_t j) noexcept
            {
                return static_cast<double>(10 * (i + 1) + (j + 1));
            }

            /** @brief Returns whether (i, j) lies in a band of kl sub- and ku super-diagonals. */
            [[nodiscard]] constexpr bool is_in_band(int64_t i, int64_t j, int64_t kl,
                                                    int64_t ku) noexcept
            {
                return (j - i <= ku) && (i - j <= kl);
            }

            /** @brief Returns the access of the worked matrix under a layout and a Transpose. */
            [[nodiscard]] constexpr Kernels::BandAccess worked_access(Layout layout,
                                                                      Transpose trans) noexcept
            {
                return Kernels::band_access(layout, trans, K_WORKED_LOWER, K_WORKED_UPPER,
                                            K_WORKED_LINE);
            }

            /**
             * @brief Returns whether the band mapping of a layout is the CBLAS formula, which is
             *        written here without the brick.
             */
            [[nodiscard]] constexpr bool band_matches_cblas(Layout layout)
            {
                const int64_t m = 5, n = 7, kl = 1, ku = 2, ld = 5;
                const Kernels::BandAccess a =
                    Kernels::band_access(layout, Transpose::NoTrans, kl, ku, ld);
                const Kernels::BandAccess t =
                    Kernels::band_access(layout, Transpose::Trans, kl, ku, ld);
                const Kernels::BandAccess h =
                    Kernels::band_access(layout, Transpose::ConjTrans, kl, ku, ld);
                if (a.conjugated || t.conjugated || !h.conjugated)
                    return false;

                for (int64_t i = 0; i < m; ++i)
                {
                    int64_t first = -1, end = -1;
                    for (int64_t j = 0; j < n; ++j)
                    {
                        const bool in_band = (j - i <= ku) && (i - j <= kl);
                        if (a.holds(i, j) != in_band || t.holds(j, i) != in_band)
                            return false;
                        if (!in_band)
                            continue;
                        if (first < 0)
                            first = j;
                        end = j + 1;

                        const int64_t expected = (layout == Layout::ColumnMajor)
                                                     ? j * ld + (ku + i - j)
                                                     : i * ld + (kl + j - i);
                        if (a.index(i, j) != expected || t.index(j, i) != expected ||
                            h.index(j, i) != expected)
                            return false;
                    }
                    if (first >= 0 && (a.first_column(i) != first || a.end_column(i, n) != end))
                        return false;
                    if (first < 0 && a.first_column(i) < a.end_column(i, n))
                        return false;
                }

                for (int64_t j = 0; j < n; ++j)
                {
                    int64_t first = -1, end = -1;
                    for (int64_t i = 0; i < m; ++i)
                    {
                        if ((j - i <= ku) && (i - j <= kl))
                        {
                            if (first < 0)
                                first = i;
                            end = i + 1;
                        }
                    }
                    if (first >= 0 && (t.first_column(j) != first || t.end_column(j, m) != end))
                        return false;
                    if (first < 0 && t.first_column(j) < t.end_column(j, m))
                        return false;
                }
                return true;
            }

            /**
             * @brief Returns whether the band storage of one triangle is the LAPACK formula,
             *        which is written here without the brick.
             */
            [[nodiscard]] constexpr bool triangle_band_matches_cblas(Layout layout, Uplo uplo)
            {
                const int64_t n = 6, k = 2, ld = 3;
                const Kernels::BandAccess a =
                    Kernels::triangle_band_access(layout, uplo, Transpose::NoTrans, k, ld);
                for (int64_t i = 0; i < n; ++i)
                {
                    for (int64_t j = 0; j < n; ++j)
                    {
                        const bool stored = (uplo == Uplo::Upper) ? (j >= i && j - i <= k)
                                                                  : (i >= j && i - j <= k);
                        if (a.holds(i, j) != stored)
                            return false;
                        if (!stored)
                            continue;

                        int64_t expected = 0;
                        if (layout == Layout::ColumnMajor)
                            expected = (uplo == Uplo::Upper) ? j * ld + (k + i - j)
                                                             : j * ld + (i - j);
                        else
                            expected = (uplo == Uplo::Upper) ? i * ld + (j - i)
                                                             : i * ld + (k + j - i);
                        if (a.index(i, j) != expected)
                            return false;
                    }
                }
                return true;
            }

            /**
             * @brief Expects that the column range of every row of op(X) equals a scan of the
             *        band.
             * @param lower Sub-diagonals of op(X), not of X; upper counts its super-diagonals.
             */
            void expect_column_ranges_match_a_scan(const Kernels::BandAccess& access,
                                                   int64_t rows, int64_t cols, int64_t lower,
                                                   int64_t upper)
            {
                for (int64_t r = 0; r < rows; ++r)
                {
                    SCOPED_TRACE(::testing::Message() << "row " << r);

                    int64_t first = -1;
                    int64_t end = -1;
                    for (int64_t c = 0; c < cols; ++c)
                    {
                        if (!is_in_band(r, c, lower, upper))
                            continue;
                        if (first < 0)
                            first = c;
                        end = c + 1;
                    }

                    if (first < 0)
                    {
                        EXPECT_GE(access.first_column(r), access.end_column(r, cols));
                        continue;
                    }
                    EXPECT_EQ(first, access.first_column(r));
                    EXPECT_EQ(end, access.end_column(r, cols));
                }
            }
        } // namespace

        /**
         * @brief Tests at compile time that both band storages follow the CBLAS formulas.
         */
        TEST(Small_BandAccess, TheMappingIsTheCblasFormula)
        {
            static_assert(band_matches_cblas(Layout::ColumnMajor),
                          "The column-major band mapping is the CBLAS one.");
            static_assert(band_matches_cblas(Layout::RowMajor),
                          "The row-major band mapping is the CBLAS one.");

            static_assert(triangle_band_matches_cblas(Layout::ColumnMajor, Uplo::Upper),
                          "A column-major upper triangle band is the LAPACK one.");
            static_assert(triangle_band_matches_cblas(Layout::ColumnMajor, Uplo::Lower),
                          "A column-major lower triangle band is the LAPACK one.");
            static_assert(triangle_band_matches_cblas(Layout::RowMajor, Uplo::Upper),
                          "A row-major upper triangle band is the CBLAS one.");
            static_assert(triangle_band_matches_cblas(Layout::RowMajor, Uplo::Lower),
                          "A row-major lower triangle band is the CBLAS one.");

            SUCCEED();
        }

        /**
         * @brief Tests that every stored element of the worked matrix is read from its slot.
         */
        TEST(Small_BandAccess, TheWorkedMatrixReadsBack)
        {
            for (const Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(layout_name(layout));

                const Kernels::BandAccess access = worked_access(layout, Transpose::NoTrans);
                const double* p = worked_storage(layout);

                int stored = 0;
                for (int64_t i = 0; i < K_WORKED_ROWS; ++i)
                {
                    for (int64_t j = 0; j < K_WORKED_COLS; ++j)
                    {
                        const bool in_band = is_in_band(i, j, K_WORKED_LOWER, K_WORKED_UPPER);
                        EXPECT_EQ(in_band, access.holds(i, j)) << "at (" << i << ", " << j << ")";
                        if (!in_band)
                            continue;

                        ++stored;
                        ASSERT_GE(access.index(i, j), 0);
                        ASSERT_LT(access.index(i, j), worked_slots(layout));
                        EXPECT_EQ(worked_element(i, j), p[access.index(i, j)])
                            << "at (" << i << ", " << j << ")";
                        EXPECT_EQ(worked_element(i, j), access.at(p, i, j))
                            << "at (" << i << ", " << j << ")";
                    }
                }
                EXPECT_EQ(K_WORKED_STORED, stored);
            }
        }

        /**
         * @brief Tests that TRANS and CONJ_TRANS address A(i, j) as (j, i) and exchange the
         *        band widths.
         */
        TEST(Small_BandAccess, ATransposedAccessAddressesTheMirror)
        {
            for (const Layout layout : K_LAYOUTS)
            {
                const Kernels::BandAccess plain = worked_access(layout, Transpose::NoTrans);
                const double* p = worked_storage(layout);

                EXPECT_EQ(K_WORKED_LOWER, plain.lower);
                EXPECT_EQ(K_WORKED_UPPER, plain.upper);
                EXPECT_FALSE(plain.conjugated);

                for (const Transpose trans : {Transpose::Trans, Transpose::ConjTrans})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << layout_name(layout) << ", " << Core::to_string(trans));

                    const Kernels::BandAccess op = worked_access(layout, trans);
                    EXPECT_EQ(K_WORKED_UPPER, op.lower);
                    EXPECT_EQ(K_WORKED_LOWER, op.upper);
                    EXPECT_EQ(trans == Transpose::ConjTrans, op.conjugated);

                    for (int64_t i = 0; i < K_WORKED_ROWS; ++i)
                    {
                        for (int64_t j = 0; j < K_WORKED_COLS; ++j)
                        {
                            const bool in_band =
                                is_in_band(i, j, K_WORKED_LOWER, K_WORKED_UPPER);
                            EXPECT_EQ(in_band, op.holds(j, i))
                                << "at (" << i << ", " << j << ")";
                            if (!in_band)
                                continue;

                            EXPECT_EQ(plain.index(i, j), op.index(j, i))
                                << "at (" << i << ", " << j << ")";
                            EXPECT_EQ(worked_element(i, j), p[op.index(j, i)])
                                << "at (" << i << ", " << j << ")";
                        }
                    }
                }
            }
        }

        /**
         * @brief Tests that at conjugates a complex element under CONJ_TRANS and no other read.
         */
        TEST(Small_BandAccess, AtConjugatesOnlyUnderConjTrans)
        {
            for (const Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(layout_name(layout));

                const double* real_storage = worked_storage(layout);
                const std::size_t slots = static_cast<std::size_t>(worked_slots(layout));

                // The imaginary part of a slot is one more than its offset, so none is zero.
                std::vector<std::complex<double>> complex_storage(slots);
                for (std::size_t slot = 0; slot < slots; ++slot)
                    complex_storage[slot] = std::complex<double>(real_storage[slot],
                                                                 static_cast<double>(slot + 1));

                const Kernels::BandAccess plain = worked_access(layout, Transpose::NoTrans);
                const Kernels::BandAccess trans = worked_access(layout, Transpose::Trans);
                const Kernels::BandAccess conj = worked_access(layout, Transpose::ConjTrans);
                const std::complex<double>* p = complex_storage.data();

                for (int64_t i = 0; i < K_WORKED_ROWS; ++i)
                {
                    for (int64_t j = 0; j < K_WORKED_COLS; ++j)
                    {
                        if (!is_in_band(i, j, K_WORKED_LOWER, K_WORKED_UPPER))
                            continue;

                        const std::complex<double> stored =
                            complex_storage[static_cast<std::size_t>(plain.index(i, j))];
                        ASSERT_EQ(worked_element(i, j), stored.real());

                        EXPECT_EQ(stored, plain.at(p, i, j));
                        EXPECT_EQ(stored, trans.at(p, j, i));
                        EXPECT_EQ(std::conj(stored), conj.at(p, j, i));
                        EXPECT_EQ(worked_element(i, j), conj.at(real_storage, j, i));
                    }
                }
            }
        }

        /**
         * @brief Tests first_column and end_column against a scan, rows without an element
         *        included.
         */
        TEST(Small_BandAccess, TheColumnRangeEqualsAScan)
        {
            const ScannedBand bands[] = {
                {7, 3, 0, 0}, {7, 3, 1, 0}, {7, 3, 0, 2}, {7, 3, 9, 9},
                {3, 7, 0, 0}, {3, 7, 1, 0}, {3, 7, 0, 2}, {3, 7, 9, 9}};

            for (const Layout layout : K_LAYOUTS)
            {
                for (const ScannedBand& band : bands)
                {
                    SCOPED_TRACE(::testing::Message()
                                 << layout_name(layout) << ", " << band.rows << " x "
                                 << band.cols << ", kl " << band.lower << ", ku " << band.upper);

                    const int64_t ld = band.lower + band.upper + 1;
                    expect_column_ranges_match_a_scan(
                        Kernels::band_access(layout, Transpose::NoTrans, band.lower, band.upper,
                                             ld),
                        band.rows, band.cols, band.lower, band.upper);
                    expect_column_ranges_match_a_scan(
                        Kernels::band_access(layout, Transpose::Trans, band.lower, band.upper, ld),
                        band.cols, band.rows, band.upper, band.lower);
                }
            }
        }

        /**
         * @brief Tests that a cursor walked along a row, forward and back, stands on the offset
         *        index gives for every stored column.
         */
        TEST(Small_BandAccess, ARowCursorReachesTheOffsetOfEveryColumn)
        {
            constexpr int64_t rows = 7;
            constexpr int64_t cols = 6;
            constexpr int64_t lower = 2;
            constexpr int64_t upper = 1;

            for (const Layout layout : K_LAYOUTS)
            {
                for (const Transpose trans : {Transpose::NoTrans, Transpose::Trans})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << layout_name(layout) << ", " << Core::to_string(trans));

                    const Kernels::BandAccess access =
                        Kernels::band_access(layout, trans, lower, upper, lower + upper + 1);
                    const bool transposed = Core::is_transposed(trans);
                    const int64_t op_rows = transposed ? cols : rows;
                    const int64_t op_cols = transposed ? rows : cols;

                    for (int64_t r = 0; r < op_rows; ++r)
                    {
                        const int64_t first = access.first_column(r);
                        const int64_t end = access.end_column(r, op_cols);
                        if (first >= end)
                            continue;

                        Kernels::RowCursor cursor = access.row_cursor(r, first);
                        for (int64_t c = first; c < end; ++c, cursor.advance())
                            EXPECT_EQ(access.index(r, c), cursor.offset)
                                << "forward, row " << r << ", column " << c;

                        cursor = access.row_cursor(r, end - 1);
                        for (int64_t c = end - 1; c >= first; --c, cursor.retreat())
                            EXPECT_EQ(access.index(r, c), cursor.offset)
                                << "backward, row " << r << ", column " << c;
                    }
                }
            }
        }

        /**
         * @brief Tests on which slot of a stored line each triangle band storage puts the
         *        diagonal.
         */
        TEST(Small_BandAccess, ATriangleBandPutsTheDiagonalOnItsLine)
        {
            constexpr int64_t n = 6;
            constexpr int64_t k = 2;
            constexpr int64_t ld = 4;

            const DiagonalLine lines[] = {{Layout::ColumnMajor, Uplo::Upper, k},
                                          {Layout::ColumnMajor, Uplo::Lower, 0},
                                          {Layout::RowMajor, Uplo::Upper, 0},
                                          {Layout::RowMajor, Uplo::Lower, k}};

            for (const DiagonalLine& line : lines)
            {
                SCOPED_TRACE(::testing::Message() << layout_name(line.layout) << ", "
                                                  << Core::to_string(line.uplo));

                const Kernels::BandAccess access = Kernels::triangle_band_access(
                    line.layout, line.uplo, Transpose::NoTrans, k, ld);
                for (int64_t i = 0; i < n; ++i)
                    EXPECT_EQ(i * ld + line.offset, access.index(i, i)) << "at diagonal " << i;
            }
        }

        /**
         * @brief Tests that no index reaches the slots a line holds past its band.
         */
        TEST(Small_BandAccess, SpareSlotsAreNeverAddressed)
        {
            constexpr int64_t band_slots = K_WORKED_LOWER + K_WORKED_UPPER + 1;
            constexpr int64_t ld = band_slots + 2;

            for (const Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(layout_name(layout));

                const Kernels::BandAccess access = Kernels::band_access(
                    layout, Transpose::NoTrans, K_WORKED_LOWER, K_WORKED_UPPER, ld);

                for (int64_t i = 0; i < K_WORKED_ROWS; ++i)
                {
                    for (int64_t j = 0; j < K_WORKED_COLS; ++j)
                    {
                        if (!is_in_band(i, j, K_WORKED_LOWER, K_WORKED_UPPER))
                            continue;

                        const int64_t line = (layout == Layout::ColumnMajor) ? j : i;
                        const int64_t index = access.index(i, j);
                        EXPECT_EQ(line, index / ld) << "at (" << i << ", " << j << ")";
                        EXPECT_GE(index % ld, 0) << "at (" << i << ", " << j << ")";
                        EXPECT_LT(index % ld, band_slots) << "at (" << i << ", " << j << ")";
                    }
                }
            }
        }

        /**
         * @brief Tests that a band of no off-diagonal and one slot per line is the diagonal.
         */
        TEST(Small_BandAccess, ADiagonalMatrixIsStoredAsItsDiagonal)
        {
            constexpr int64_t n = 5;

            for (const Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(layout_name(layout));

                const Kernels::BandAccess access =
                    Kernels::band_access(layout, Transpose::NoTrans, 0, 0, 1);
                for (int64_t i = 0; i < n; ++i)
                {
                    EXPECT_EQ(i, access.index(i, i));
                    EXPECT_TRUE(access.holds(i, i));
                    EXPECT_FALSE(access.holds(i, i + 1));
                    EXPECT_FALSE(access.holds(i + 1, i));
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
