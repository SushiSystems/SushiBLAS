/**************************************************************************/
/* test_packed_access.cpp                                                 */
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
 * @file test_packed_access.cpp
 * @brief Tests the offsets, the triangle test and the column ranges of PackedAccess.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::Layout;
            using Core::Transpose;
            using Core::Uplo;

            /// @brief Holds the order of the worked matrices of the design, section 2.2.
            constexpr int64_t K_WORKED_ORDER = 3;

            /// @brief Holds the elements one triangle of a worked matrix stores.
            constexpr int64_t K_WORKED_STORED = 6;

            /// @brief Holds a worked triangle whose stored lines grow: 1 | 2 4 | 3 5 6.
            constexpr double K_WORKED_GROWING[] = {1.0, 2.0, 4.0, 3.0, 5.0, 6.0};

            /// @brief Holds a worked triangle whose stored lines shrink: 1 2 3 | 4 5 | 6.
            constexpr double K_WORKED_SHRINKING[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

            /** @brief Holds one packed storage: the order of its lines and the triangle held. */
            struct PackedStorage
            {
                Layout layout; /**< COLUMN_MAJOR packs by columns, ROW_MAJOR by rows. */
                Uplo uplo;     /**< The triangle of the matrix the storage holds. */
            };

            /// @brief Holds the four packed storages every case runs under.
            constexpr PackedStorage K_STORAGES[] = {{Layout::ColumnMajor, Uplo::Upper},
                                                    {Layout::ColumnMajor, Uplo::Lower},
                                                    {Layout::RowMajor, Uplo::Upper},
                                                    {Layout::RowMajor, Uplo::Lower}};

            /** @brief Returns the row of the element a CBLAS walk visits at (line, along). */
            [[nodiscard]] constexpr int64_t walked_row(Layout layout, int64_t line,
                                                       int64_t along) noexcept
            {
                return (layout == Layout::RowMajor) ? line : along;
            }

            /** @brief Returns the column of the element a CBLAS walk visits at (line, along). */
            [[nodiscard]] constexpr int64_t walked_col(Layout layout, int64_t line,
                                                       int64_t along) noexcept
            {
                return (layout == Layout::RowMajor) ? along : line;
            }

            /** @brief Returns whether (i, j) lies in a triangle of a square matrix. */
            [[nodiscard]] constexpr bool is_in_triangle(Uplo uplo, int64_t i, int64_t j) noexcept
            {
                return (uplo == Uplo::Upper) ? (i <= j) : (i >= j);
            }

            /**
             * @brief Returns element (i, j) of the worked U for UPPER and of L = U^T for LOWER.
             * @pre (i, j) lies in the triangle uplo names.
             */
            [[nodiscard]] constexpr double worked_element(Uplo uplo, int64_t i, int64_t j) noexcept
            {
                constexpr double upper[3][3] = {{1.0, 2.0, 3.0}, {0.0, 4.0, 5.0}, {0.0, 0.0, 6.0}};
                return (uplo == Uplo::Upper) ? upper[i][j] : upper[j][i];
            }

            /** @brief Returns the packed array of the worked triangle of one storage. */
            [[nodiscard]] constexpr const double* worked_storage(const PackedStorage& s) noexcept
            {
                const bool by_rows = (s.layout == Layout::RowMajor);
                const bool growing = (s.uplo == Uplo::Upper) != by_rows;
                return growing ? K_WORKED_GROWING : K_WORKED_SHRINKING;
            }

            /**
             * @brief Returns whether the packed mapping of one storage numbers the triangle in
             *        the order CBLAS lays it down, which is walked here without the brick.
             */
            [[nodiscard]] constexpr bool packed_matches_cblas(Layout layout, Uplo uplo)
            {
                const int64_t n = 5;
                const Kernels::PackedAccess a =
                    Kernels::packed_access(layout, uplo, Transpose::NoTrans, n);
                const Kernels::PackedAccess t =
                    Kernels::packed_access(layout, uplo, Transpose::Trans, n);
                const bool upper = (uplo == Uplo::Upper);
                if (a.is_upper() != upper || t.is_upper() == upper)
                    return false;

                int64_t next = 0;
                for (int64_t line = 0; line < n; ++line)
                {
                    for (int64_t along = 0; along < n; ++along)
                    {
                        const int64_t i = (layout == Layout::RowMajor) ? line : along;
                        const int64_t j = (layout == Layout::RowMajor) ? along : line;
                        const bool stored = upper ? (i <= j) : (i >= j);
                        if (a.holds(i, j) != stored || t.holds(j, i) != stored)
                            return false;
                        if (!stored)
                            continue;
                        if (a.index(i, j) != next || t.index(j, i) != next)
                            return false;
                        ++next;
                    }
                }
                if (next != n * (n + 1) / 2)
                    return false;

                for (int64_t i = 0; i < n; ++i)
                {
                    if (a.first_column(i) != (upper ? i : 0) ||
                        a.end_column(i, n) != (upper ? n : i + 1))
                        return false;
                }
                return true;
            }

            /** @brief Holds one packed storage, a Transpose and an order of one case. */
            struct PackedCase
            {
                PackedStorage storage; /**< The packed storage of the case. */
                Transpose trans;       /**< The transpose applied to the stored triangle. */
                int64_t order;         /**< The order of the matrix. */
            };

            /** @brief Returns the four storages crossed with the given transposes and orders. */
            [[nodiscard]] std::vector<PackedCase> packed_cases(std::vector<Transpose> transposes,
                                                               std::vector<int64_t> orders)
            {
                std::vector<PackedCase> cases;
                for (const PackedStorage& storage : K_STORAGES)
                    for (const Transpose trans : transposes)
                        for (const int64_t order : orders)
                            cases.push_back({storage, trans, order});
                return cases;
            }

            /** @brief Returns the four storages with no transpose and the worked order. */
            [[nodiscard]] std::vector<PackedCase> worked_cases()
            {
                return packed_cases({Transpose::NoTrans}, {K_WORKED_ORDER});
            }

            /** @brief Returns the CamelCase word of a Transpose, which a test name can hold. */
            [[nodiscard]] const char* transpose_word(Transpose trans) noexcept
            {
                if (trans == Transpose::NoTrans)
                    return "NoTrans";
                return (trans == Transpose::Trans) ? "Trans" : "ConjTrans";
            }

            /** @brief Names a case for the test name. */
            std::string packed_case_name(const ::testing::TestParamInfo<PackedCase>& info)
            {
                const PackedCase& row = info.param;
                return std::string(row.storage.layout == Layout::RowMajor ? "RowMajor"
                                                                          : "ColumnMajor") +
                       (row.storage.uplo == Uplo::Upper ? "Upper" : "Lower") +
                       transpose_word(row.trans) + "Order" + std::to_string(row.order);
            }

            /** @brief Expects that a walk of the triangle in CBLAS order counts without a gap. */
            void expect_the_walk_counts_without_a_gap(const PackedStorage& storage, int64_t n)
            {
                const Kernels::PackedAccess access =
                    Kernels::packed_access(storage.layout, storage.uplo, Transpose::NoTrans, n);
                EXPECT_EQ(n, access.order);

                int64_t next = 0;
                for (int64_t line = 0; line < n; ++line)
                {
                    for (int64_t along = 0; along < n; ++along)
                    {
                        const int64_t i = walked_row(storage.layout, line, along);
                        const int64_t j = walked_col(storage.layout, line, along);
                        const bool stored = is_in_triangle(storage.uplo, i, j);
                        EXPECT_EQ(stored, access.holds(i, j)) << "at (" << i << ", " << j << ")";
                        if (!stored)
                            continue;

                        EXPECT_EQ(next, access.index(i, j)) << "at (" << i << ", " << j << ")";
                        ++next;
                    }
                }
                EXPECT_EQ(n * (n + 1) / 2, next);
            }

            /** @brief Expects that every element of the worked U and L is read from its slot. */
            void expect_the_worked_triangle_reads_back(const PackedStorage& storage)
            {
                const Kernels::PackedAccess access = Kernels::packed_access(
                    storage.layout, storage.uplo, Transpose::NoTrans, K_WORKED_ORDER);
                const double* p = worked_storage(storage);

                for (int64_t i = 0; i < K_WORKED_ORDER; ++i)
                {
                    for (int64_t j = 0; j < K_WORKED_ORDER; ++j)
                    {
                        if (!is_in_triangle(storage.uplo, i, j))
                            continue;

                        ASSERT_GE(access.index(i, j), 0);
                        ASSERT_LT(access.index(i, j), K_WORKED_STORED);
                        EXPECT_EQ(worked_element(storage.uplo, i, j), p[access.index(i, j)])
                            << "at (" << i << ", " << j << ")";
                        EXPECT_EQ(worked_element(storage.uplo, i, j), access.at(p, i, j))
                            << "at (" << i << ", " << j << ")";
                    }
                }
            }

            /** @brief Expects that op(X) addresses A(i, j) as (j, i) over the worked triangle. */
            void expect_the_mirror_is_addressed(const PackedStorage& storage, Transpose trans)
            {
                const Kernels::PackedAccess plain = Kernels::packed_access(
                    storage.layout, storage.uplo, Transpose::NoTrans, K_WORKED_ORDER);
                const Kernels::PackedAccess op = Kernels::packed_access(
                    storage.layout, storage.uplo, trans, K_WORKED_ORDER);
                const double* p = worked_storage(storage);

                for (int64_t i = 0; i < K_WORKED_ORDER; ++i)
                {
                    for (int64_t j = 0; j < K_WORKED_ORDER; ++j)
                    {
                        const bool stored = is_in_triangle(storage.uplo, i, j);
                        EXPECT_EQ(stored, op.holds(j, i)) << "at (" << i << ", " << j << ")";
                        if (!stored)
                            continue;

                        EXPECT_EQ(plain.index(i, j), op.index(j, i))
                            << "at (" << i << ", " << j << ")";
                        EXPECT_EQ(worked_element(storage.uplo, i, j), p[op.index(j, i)])
                            << "at (" << i << ", " << j << ")";
                    }
                }
            }

            /** @brief Expects that a cursor walked along each row stands on the offset of index. */
            void expect_row_cursors_reach_every_column(const PackedCase& row)
            {
                const Kernels::PackedAccess access = Kernels::packed_access(
                    row.storage.layout, row.storage.uplo, row.trans, row.order);

                for (int64_t r = 0; r < row.order; ++r)
                {
                    const int64_t first = access.first_column(r);
                    const int64_t end = access.end_column(r, row.order);

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

            /** @brief Expects that every row of an access runs over the range its triangle names. */
            void expect_the_column_range_follows_the_triangle(const PackedCase& row)
            {
                const Kernels::PackedAccess access = Kernels::packed_access(
                    row.storage.layout, row.storage.uplo, row.trans, row.order);
                const bool upper =
                    (row.storage.uplo == Uplo::Upper) != Core::is_transposed(row.trans);

                for (int64_t i = 0; i < row.order; ++i)
                {
                    EXPECT_EQ(upper ? i : 0, access.first_column(i)) << "row " << i;
                    EXPECT_EQ(upper ? row.order : i + 1, access.end_column(i, row.order))
                        << "row " << i;
                }
            }

            /** @brief Expects that at conjugates only under CONJ_TRANS over the worked triangle. */
            void expect_at_conjugates_only_under_conj_trans(const PackedStorage& storage)
            {
                const double* real_storage = worked_storage(storage);

                // The imaginary part of a slot is one more than its offset, so none is zero.
                std::complex<double> complex_storage[K_WORKED_STORED];
                for (int64_t slot = 0; slot < K_WORKED_STORED; ++slot)
                    complex_storage[slot] =
                        std::complex<double>(real_storage[slot], static_cast<double>(slot + 1));

                const Kernels::PackedAccess plain = Kernels::packed_access(
                    storage.layout, storage.uplo, Transpose::NoTrans, K_WORKED_ORDER);
                const Kernels::PackedAccess trans = Kernels::packed_access(
                    storage.layout, storage.uplo, Transpose::Trans, K_WORKED_ORDER);
                const Kernels::PackedAccess conj = Kernels::packed_access(
                    storage.layout, storage.uplo, Transpose::ConjTrans, K_WORKED_ORDER);
                const std::complex<double>* p = complex_storage;

                for (int64_t i = 0; i < K_WORKED_ORDER; ++i)
                {
                    for (int64_t j = 0; j < K_WORKED_ORDER; ++j)
                    {
                        if (!is_in_triangle(storage.uplo, i, j))
                            continue;

                        const std::complex<double> stored = complex_storage[plain.index(i, j)];
                        ASSERT_EQ(worked_element(storage.uplo, i, j), stored.real());

                        EXPECT_EQ(stored, plain.at(p, i, j));
                        EXPECT_EQ(stored, trans.at(p, j, i));
                        EXPECT_EQ(std::conj(stored), conj.at(p, j, i));
                        EXPECT_EQ(worked_element(storage.uplo, i, j),
                                  conj.at(real_storage, j, i));
                    }
                }
            }

            /** @brief Returns how many elements the loop a kernel runs visits over an access. */
            [[nodiscard]] int64_t elements_visited(const Kernels::PackedAccess& access)
            {
                int64_t visited = 0;
                for (int64_t r = 0; r < access.order; ++r)
                    for (int64_t c = access.first_column(r); c < access.end_column(r, access.order);
                         ++c)
                        ++visited;
                return visited;
            }
        } // namespace

        /** @brief Runs a packed access case once per storage, Transpose and order. */
        class Small_PackedAccess : public ::testing::TestWithParam<PackedCase>
        {

        };

        TEST_P(Small_PackedAccess, TheMappingIsTheCblasOrder)
        {
            EXPECT_TRUE(packed_matches_cblas(GetParam().storage.layout, GetParam().storage.uplo));
        }

        TEST_P(Small_PackedAccess, TheWorkedTriangleReadsBack)
        {
            expect_the_worked_triangle_reads_back(GetParam().storage);
        }

        TEST_P(Small_PackedAccess, AtConjugatesOnlyUnderConjTrans)
        {
            expect_at_conjugates_only_under_conj_trans(GetParam().storage);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_PackedAccess, ::testing::ValuesIn(worked_cases()),
                                 packed_case_name);

        /** @brief Walks the triangle of one storage at one order. */
        class Small_PackedAccessWalks : public ::testing::TestWithParam<PackedCase>
        {

        };

        TEST_P(Small_PackedAccessWalks, AWalkInCblasOrderCountsWithoutAGap)
        {
            expect_the_walk_counts_without_a_gap(GetParam().storage, GetParam().order);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_PackedAccessWalks,
                                 ::testing::ValuesIn(packed_cases({Transpose::NoTrans},
                                                                  {1, 2, 5})),
                                 packed_case_name);

        /** @brief Reads the mirror of the worked triangle once per storage and transpose. */
        class Small_PackedAccessMirrors : public ::testing::TestWithParam<PackedCase>
        {

        };

        TEST_P(Small_PackedAccessMirrors, ATransposedAccessReportsTheOtherTriangle)
        {
            const PackedCase& row = GetParam();
            const Kernels::PackedAccess plain = Kernels::packed_access(
                row.storage.layout, row.storage.uplo, Transpose::NoTrans, K_WORKED_ORDER);
            const Kernels::PackedAccess op = Kernels::packed_access(
                row.storage.layout, row.storage.uplo, row.trans, K_WORKED_ORDER);

            EXPECT_EQ(row.storage.uplo == Uplo::Upper, plain.is_upper());
            EXPECT_FALSE(plain.conjugated);
            EXPECT_EQ(row.storage.uplo != Uplo::Upper, op.is_upper());
            EXPECT_EQ(row.trans == Transpose::ConjTrans, op.conjugated);
            EXPECT_EQ(K_WORKED_ORDER, op.order);
        }

        TEST_P(Small_PackedAccessMirrors, ATransposedAccessAddressesTheMirror)
        {
            expect_the_mirror_is_addressed(GetParam().storage, GetParam().trans);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_PackedAccessMirrors,
            ::testing::ValuesIn(packed_cases({Transpose::Trans, Transpose::ConjTrans},
                                             {K_WORKED_ORDER})),
            packed_case_name);

        /** @brief Walks a row cursor once per storage, transpose and order. */
        class Small_PackedAccessCursors : public ::testing::TestWithParam<PackedCase>
        {

        };

        TEST_P(Small_PackedAccessCursors, ARowCursorReachesTheOffsetOfEveryColumn)
        {
            expect_row_cursors_reach_every_column(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_PackedAccessCursors,
            ::testing::ValuesIn(packed_cases({Transpose::NoTrans, Transpose::Trans}, {1, 2, 7})),
            packed_case_name);

        /** @brief Reads the column range of a row once per storage and transpose. */
        class Small_PackedAccessRanges : public ::testing::TestWithParam<PackedCase>
        {

        };

        TEST_P(Small_PackedAccessRanges, TheColumnRangeFollowsTheTriangle)
        {
            expect_the_column_range_follows_the_triangle(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_PackedAccessRanges,
            ::testing::ValuesIn(packed_cases({Transpose::NoTrans, Transpose::Trans}, {5})),
            packed_case_name);

        /** @brief Resolves an access of order zero once per storage and transpose. */
        class Small_PackedAccessEmpty : public ::testing::TestWithParam<PackedCase>
        {

        };

        TEST_P(Small_PackedAccessEmpty, AnOrderOfZeroHoldsNoElement)
        {
            const PackedCase& row = GetParam();

            const Kernels::PackedAccess access = Kernels::packed_access(
                row.storage.layout, row.storage.uplo, row.trans, 0);

            EXPECT_EQ(0, access.order);
            EXPECT_EQ(0, elements_visited(access));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_PackedAccessEmpty,
            ::testing::ValuesIn(packed_cases({Transpose::NoTrans, Transpose::ConjTrans}, {0})),
            packed_case_name);
    } // namespace Test
} // namespace SushiBLAS
