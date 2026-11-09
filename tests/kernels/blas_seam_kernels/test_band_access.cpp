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
 * @file test_band_access.cpp
 * @brief Tests the offsets, the band test and the column ranges of BandAccess.
 * @author Mustafa Garip
 */

#include <array>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
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

            /** @brief Holds one layout and one Transpose of an access under test. */
            struct LayoutTranspose
            {
                Layout layout;    /**< The band storage. */
                Transpose trans;  /**< The transpose applied to the stored matrix. */
            };

            /** @brief Holds one layout and the band shape its column ranges are scanned for. */
            struct LayoutBand
            {
                Layout layout;   /**< The band storage. */
                ScannedBand band; /**< The shape of the band. */
            };

            /** @brief Returns both layouts crossed with TRANS and CONJ_TRANS. */
            [[nodiscard]] std::array<LayoutTranspose, 4> mirror_rows()
            {
                return {{{Layout::ColumnMajor, Transpose::Trans},
                         {Layout::ColumnMajor, Transpose::ConjTrans},
                         {Layout::RowMajor, Transpose::Trans},
                         {Layout::RowMajor, Transpose::ConjTrans}}};
            }

            /** @brief Returns both layouts crossed with NO_TRANS and TRANS. */
            [[nodiscard]] std::array<LayoutTranspose, 4> cursor_rows()
            {
                return {{{Layout::ColumnMajor, Transpose::NoTrans},
                         {Layout::ColumnMajor, Transpose::Trans},
                         {Layout::RowMajor, Transpose::NoTrans},
                         {Layout::RowMajor, Transpose::Trans}}};
            }

            /** @brief Returns both layouts crossed with the eight scanned band shapes. */
            [[nodiscard]] std::vector<LayoutBand> scanned_rows()
            {
                const ScannedBand bands[] = {
                    {7, 3, 0, 0}, {7, 3, 1, 0}, {7, 3, 0, 2}, {7, 3, 9, 9},
                    {3, 7, 0, 0}, {3, 7, 1, 0}, {3, 7, 0, 2}, {3, 7, 9, 9}};

                std::vector<LayoutBand> rows;
                for (const Layout layout : {Layout::ColumnMajor, Layout::RowMajor})
                    for (const ScannedBand& band : bands)
                        rows.push_back({layout, band});
                return rows;
            }

            /** @brief Returns the four diagonal lines of the triangle band storages. */
            [[nodiscard]] std::array<DiagonalLine, 4> diagonal_lines()
            {
                constexpr int64_t k = 2;
                return {{{Layout::ColumnMajor, Uplo::Upper, k},
                         {Layout::ColumnMajor, Uplo::Lower, 0},
                         {Layout::RowMajor, Uplo::Upper, 0},
                         {Layout::RowMajor, Uplo::Lower, k}}};
            }

            /** @brief Returns the CamelCase word of a Transpose, which a test name can hold. */
            [[nodiscard]] const char* transpose_word(Transpose trans) noexcept
            {
                if (trans == Transpose::NoTrans)
                    return "NoTrans";
                return (trans == Transpose::Trans) ? "Trans" : "ConjTrans";
            }

            /** @brief Names a layout for the test name. */
            std::string layout_param_name(const ::testing::TestParamInfo<Layout>& info)
            {
                return info.param == Layout::RowMajor ? "RowMajor" : "ColumnMajor";
            }

            /** @brief Names a layout and a Transpose for the test name. */
            std::string layout_transpose_name(const ::testing::TestParamInfo<LayoutTranspose>& info)
            {
                return std::string(info.param.layout == Layout::RowMajor ? "RowMajor" : "ColumnMajor") +
                       transpose_word(info.param.trans);
            }

            /** @brief Names a layout and a band shape for the test name. */
            std::string layout_band_name(const ::testing::TestParamInfo<LayoutBand>& info)
            {
                const LayoutBand& row = info.param;
                return std::string(row.layout == Layout::RowMajor ? "RowMajor" : "ColumnMajor") +
                       "Rows" + std::to_string(row.band.rows) + "Cols" +
                       std::to_string(row.band.cols) + "Lower" + std::to_string(row.band.lower) +
                       "Upper" + std::to_string(row.band.upper);
            }

            /** @brief Names a triangle band storage for the test name. */
            std::string diagonal_line_name(const ::testing::TestParamInfo<DiagonalLine>& info)
            {
                return std::string(info.param.layout == Layout::RowMajor ? "RowMajor" : "ColumnMajor") +
                       (info.param.uplo == Uplo::Upper ? "Upper" : "Lower");
            }

            /** @brief Expects that every element of the worked matrix reads from its slot. */
            void expect_worked_matrix_reads_back(Layout layout)
            {
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

            /** @brief Expects that op(X) addresses A(i, j) as (j, i) over the worked matrix. */
            void expect_mirror_is_addressed(Layout layout, Transpose trans)
            {
                const Kernels::BandAccess plain = worked_access(layout, Transpose::NoTrans);
                const Kernels::BandAccess op = worked_access(layout, trans);
                const double* p = worked_storage(layout);

                for (int64_t i = 0; i < K_WORKED_ROWS; ++i)
                {
                    for (int64_t j = 0; j < K_WORKED_COLS; ++j)
                    {
                        const bool in_band = is_in_band(i, j, K_WORKED_LOWER, K_WORKED_UPPER);
                        EXPECT_EQ(in_band, op.holds(j, i)) << "at (" << i << ", " << j << ")";
                        if (!in_band)
                            continue;

                        EXPECT_EQ(plain.index(i, j), op.index(j, i))
                            << "at (" << i << ", " << j << ")";
                        EXPECT_EQ(worked_element(i, j), p[op.index(j, i)])
                            << "at (" << i << ", " << j << ")";
                    }
                }
            }

            /** @brief Returns the worked storage of a layout as complex slots with imaginary parts. */
            [[nodiscard]] std::vector<std::complex<double>> complex_worked_storage(Layout layout)
            {
                const double* real_storage = worked_storage(layout);
                const std::size_t slots = static_cast<std::size_t>(worked_slots(layout));

                // The imaginary part of a slot is one more than its offset, so none is zero.
                std::vector<std::complex<double>> complex_storage(slots);
                for (std::size_t slot = 0; slot < slots; ++slot)
                    complex_storage[slot] = std::complex<double>(real_storage[slot],
                                                                 static_cast<double>(slot + 1));
                return complex_storage;
            }

            /** @brief Expects that at conjugates only under CONJ_TRANS over the worked matrix. */
            void expect_at_conjugates_only_under_conj_trans(Layout layout)
            {
                const std::vector<std::complex<double>> complex_storage =
                    complex_worked_storage(layout);
                const double* real_storage = worked_storage(layout);
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

            /** @brief Expects that a cursor walked along each row stands on the offset of index. */
            void expect_row_cursors_reach_every_column(Layout layout, Transpose trans)
            {
                constexpr int64_t rows = 7;
                constexpr int64_t cols = 6;
                constexpr int64_t lower = 2;
                constexpr int64_t upper = 1;

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

            /** @brief Expects that no index reaches the slots a line holds past its band. */
            void expect_spare_slots_are_never_addressed(Layout layout)
            {
                constexpr int64_t band_slots = K_WORKED_LOWER + K_WORKED_UPPER + 1;
                constexpr int64_t ld = band_slots + 2;

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

            /** @brief Expects that a band of one slot per line stores the diagonal alone. */
            void expect_diagonal_matrix_is_its_diagonal(Layout layout)
            {
                constexpr int64_t n = 5;

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

            /** @brief Expects that a triangle band puts the diagonal of row i on its line. */
            void expect_diagonal_sits_on_its_line(const DiagonalLine& line)
            {
                constexpr int64_t n = 6;
                constexpr int64_t k = 2;
                constexpr int64_t ld = 4;

                const Kernels::BandAccess access = Kernels::triangle_band_access(
                    line.layout, line.uplo, Transpose::NoTrans, k, ld);
                for (int64_t i = 0; i < n; ++i)
                    EXPECT_EQ(i * ld + line.offset, access.index(i, i)) << "at diagonal " << i;
            }
        } // namespace

        /** @brief Runs a case of the band access once per band storage. */
        class Small_BandAccessLayouts : public ::testing::TestWithParam<Layout>
        {

        };

        TEST_P(Small_BandAccessLayouts, TheMappingIsTheCblasFormula)
        {
            EXPECT_TRUE(band_matches_cblas(GetParam()));
        }

        TEST_P(Small_BandAccessLayouts, TheWorkedMatrixReadsBack)
        {
            expect_worked_matrix_reads_back(GetParam());
        }

        TEST_P(Small_BandAccessLayouts, AtConjugatesOnlyUnderConjTrans)
        {
            expect_at_conjugates_only_under_conj_trans(GetParam());
        }

        TEST_P(Small_BandAccessLayouts, SpareSlotsAreNeverAddressed)
        {
            expect_spare_slots_are_never_addressed(GetParam());
        }

        TEST_P(Small_BandAccessLayouts, ADiagonalMatrixIsStoredAsItsDiagonal)
        {
            expect_diagonal_matrix_is_its_diagonal(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_BandAccessLayouts,
                                 ::testing::Values(Layout::ColumnMajor, Layout::RowMajor),
                                 layout_param_name);

        /** @brief Runs a case of the triangle band storage once per layout and triangle. */
        class Small_BandAccessTriangles : public ::testing::TestWithParam<DiagonalLine>
        {

        };

        TEST_P(Small_BandAccessTriangles, TheMappingIsTheCblasFormula)
        {
            EXPECT_TRUE(triangle_band_matches_cblas(GetParam().layout, GetParam().uplo));
        }

        TEST_P(Small_BandAccessTriangles, ATriangleBandPutsTheDiagonalOnItsLine)
        {
            expect_diagonal_sits_on_its_line(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_BandAccessTriangles, ::testing::ValuesIn(diagonal_lines()),
                                 diagonal_line_name);

        /** @brief Runs a case of a transposed access once per layout and Transpose. */
        class Small_BandAccessMirrors : public ::testing::TestWithParam<LayoutTranspose>
        {

        };

        TEST_P(Small_BandAccessMirrors, ATransposedAccessExchangesTheBandWidths)
        {
            const Kernels::BandAccess plain = worked_access(GetParam().layout, Transpose::NoTrans);
            const Kernels::BandAccess op = worked_access(GetParam().layout, GetParam().trans);

            EXPECT_EQ(K_WORKED_LOWER, plain.lower);
            EXPECT_EQ(K_WORKED_UPPER, plain.upper);
            EXPECT_FALSE(plain.conjugated);
            EXPECT_EQ(K_WORKED_UPPER, op.lower);
            EXPECT_EQ(K_WORKED_LOWER, op.upper);
            EXPECT_EQ(GetParam().trans == Transpose::ConjTrans, op.conjugated);
        }

        TEST_P(Small_BandAccessMirrors, ATransposedAccessAddressesTheMirror)
        {
            expect_mirror_is_addressed(GetParam().layout, GetParam().trans);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_BandAccessMirrors, ::testing::ValuesIn(mirror_rows()),
                                 layout_transpose_name);

        /** @brief Runs a row cursor case once per layout and Transpose. */
        class Small_BandAccessCursors : public ::testing::TestWithParam<LayoutTranspose>
        {

        };

        TEST_P(Small_BandAccessCursors, ARowCursorReachesTheOffsetOfEveryColumn)
        {
            expect_row_cursors_reach_every_column(GetParam().layout, GetParam().trans);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_BandAccessCursors, ::testing::ValuesIn(cursor_rows()),
                                 layout_transpose_name);

        /** @brief Runs the column range case once per layout and band shape. */
        class Small_BandAccessScans : public ::testing::TestWithParam<LayoutBand>
        {

        };

        TEST_P(Small_BandAccessScans, TheColumnRangeEqualsAScan)
        {
            const LayoutBand& row = GetParam();
            const int64_t ld = row.band.lower + row.band.upper + 1;

            expect_column_ranges_match_a_scan(
                Kernels::band_access(row.layout, Transpose::NoTrans, row.band.lower,
                                     row.band.upper, ld),
                row.band.rows, row.band.cols, row.band.lower, row.band.upper);
            expect_column_ranges_match_a_scan(
                Kernels::band_access(row.layout, Transpose::Trans, row.band.lower, row.band.upper,
                                     ld),
                row.band.cols, row.band.rows, row.band.upper, row.band.lower);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_BandAccessScans, ::testing::ValuesIn(scanned_rows()),
                                 layout_band_name);
    } // namespace Test
} // namespace SushiBLAS
