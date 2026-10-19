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
 * @file test_packed_access.cpp
 * @brief Tests the offsets, the triangle test and the column ranges of PackedAccess.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstdint>
#include <initializer_list>
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

            /** @brief Returns the name of a layout for a failure message. */
            [[nodiscard]] constexpr const char* layout_name(Layout layout) noexcept
            {
                return (layout == Layout::RowMajor) ? "row-major" : "column-major";
            }

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
        } // namespace

        /**
         * @brief Tests at compile time that the four packed storages follow the CBLAS order.
         */
        TEST(Medium_PackedAccess, TheMappingIsTheCblasOrder)
        {
            static_assert(packed_matches_cblas(Layout::ColumnMajor, Uplo::Upper),
                          "A column-major upper packed triangle is the CBLAS one.");
            static_assert(packed_matches_cblas(Layout::ColumnMajor, Uplo::Lower),
                          "A column-major lower packed triangle is the CBLAS one.");
            static_assert(packed_matches_cblas(Layout::RowMajor, Uplo::Upper),
                          "A row-major upper packed triangle is the CBLAS one.");
            static_assert(packed_matches_cblas(Layout::RowMajor, Uplo::Lower),
                          "A row-major lower packed triangle is the CBLAS one.");

            SUCCEED();
        }

        /**
         * @brief Tests that a walk of the triangle in CBLAS order meets the offsets 0, 1, 2
         *        and on, and ends at n * (n + 1) / 2.
         */
        TEST(Medium_PackedAccess, AWalkInCblasOrderCountsWithoutAGap)
        {
            for (const PackedStorage& storage : K_STORAGES)
            {
                for (const int64_t n : {int64_t{1}, int64_t{2}, int64_t{5}})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << layout_name(storage.layout) << ", "
                                 << Core::to_string(storage.uplo) << ", order " << n);

                    const Kernels::PackedAccess access = Kernels::packed_access(
                        storage.layout, storage.uplo, Transpose::NoTrans, n);
                    EXPECT_EQ(n, access.order);

                    int64_t next = 0;
                    for (int64_t line = 0; line < n; ++line)
                    {
                        for (int64_t along = 0; along < n; ++along)
                        {
                            const int64_t i = walked_row(storage.layout, line, along);
                            const int64_t j = walked_col(storage.layout, line, along);
                            const bool stored = is_in_triangle(storage.uplo, i, j);
                            EXPECT_EQ(stored, access.holds(i, j))
                                << "at (" << i << ", " << j << ")";
                            if (!stored)
                                continue;

                            EXPECT_EQ(next, access.index(i, j))
                                << "at (" << i << ", " << j << ")";
                            ++next;
                        }
                    }
                    EXPECT_EQ(n * (n + 1) / 2, next);
                }
            }
        }

        /**
         * @brief Tests that every element of the worked U and L is read from its slot.
         */
        TEST(Medium_PackedAccess, TheWorkedTrianglesReadBack)
        {
            for (const PackedStorage& storage : K_STORAGES)
            {
                SCOPED_TRACE(::testing::Message() << layout_name(storage.layout) << ", "
                                                  << Core::to_string(storage.uplo));

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
        }

        /**
         * @brief Tests that TRANS and CONJ_TRANS address A(i, j) as (j, i) and report the
         *        other triangle.
         */
        TEST(Medium_PackedAccess, ATransposedAccessAddressesTheMirror)
        {
            for (const PackedStorage& storage : K_STORAGES)
            {
                const Kernels::PackedAccess plain = Kernels::packed_access(
                    storage.layout, storage.uplo, Transpose::NoTrans, K_WORKED_ORDER);
                const double* p = worked_storage(storage);

                EXPECT_EQ(storage.uplo == Uplo::Upper, plain.is_upper());
                EXPECT_FALSE(plain.conjugated);

                for (const Transpose trans : {Transpose::Trans, Transpose::ConjTrans})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << layout_name(storage.layout) << ", "
                                 << Core::to_string(storage.uplo) << ", "
                                 << Core::to_string(trans));

                    const Kernels::PackedAccess op = Kernels::packed_access(
                        storage.layout, storage.uplo, trans, K_WORKED_ORDER);
                    EXPECT_EQ(storage.uplo != Uplo::Upper, op.is_upper());
                    EXPECT_EQ(trans == Transpose::ConjTrans, op.conjugated);
                    EXPECT_EQ(K_WORKED_ORDER, op.order);

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
            }
        }

        /**
         * @brief Tests that a cursor walked along a row, forward and back, stands on the offset
         *        index gives for every stored column.
         */
        TEST(Medium_PackedAccess, ARowCursorReachesTheOffsetOfEveryColumn)
        {
            for (const PackedStorage& storage : K_STORAGES)
            {
                for (const Transpose trans : {Transpose::NoTrans, Transpose::Trans})
                {
                    for (const int64_t n : {int64_t{1}, int64_t{2}, int64_t{7}})
                    {
                        SCOPED_TRACE(::testing::Message()
                                     << layout_name(storage.layout) << ", "
                                     << Core::to_string(storage.uplo) << ", "
                                     << Core::to_string(trans) << ", order " << n);

                        const Kernels::PackedAccess access =
                            Kernels::packed_access(storage.layout, storage.uplo, trans, n);

                        for (int64_t r = 0; r < n; ++r)
                        {
                            const int64_t first = access.first_column(r);
                            const int64_t end = access.end_column(r, n);

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
        }

        /**
         * @brief Tests that a row of an upper triangle runs from its diagonal to the order and
         *        a row of a lower one from zero through its diagonal.
         */
        TEST(Medium_PackedAccess, TheColumnRangeFollowsTheTriangle)
        {
            constexpr int64_t n = 5;

            for (const PackedStorage& storage : K_STORAGES)
            {
                for (const Transpose trans : {Transpose::NoTrans, Transpose::Trans})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << layout_name(storage.layout) << ", "
                                 << Core::to_string(storage.uplo) << ", "
                                 << Core::to_string(trans));

                    const Kernels::PackedAccess access =
                        Kernels::packed_access(storage.layout, storage.uplo, trans, n);
                    const bool upper = (storage.uplo == Uplo::Upper) != Core::is_transposed(trans);

                    for (int64_t i = 0; i < n; ++i)
                    {
                        EXPECT_EQ(upper ? i : 0, access.first_column(i)) << "row " << i;
                        EXPECT_EQ(upper ? n : i + 1, access.end_column(i, n)) << "row " << i;
                    }
                }
            }
        }

        /**
         * @brief Tests that at conjugates a complex element under CONJ_TRANS and no other read.
         */
        TEST(Medium_PackedAccess, AtConjugatesOnlyUnderConjTrans)
        {
            for (const PackedStorage& storage : K_STORAGES)
            {
                SCOPED_TRACE(::testing::Message() << layout_name(storage.layout) << ", "
                                                  << Core::to_string(storage.uplo));

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
        }

        /**
         * @brief Tests that a triangle of order 0 has no row, so no element is ever asked for.
         */
        TEST(Medium_PackedAccess, AnOrderOfZeroHoldsNoElement)
        {
            for (const PackedStorage& storage : K_STORAGES)
            {
                for (const Transpose trans : {Transpose::NoTrans, Transpose::ConjTrans})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << layout_name(storage.layout) << ", "
                                 << Core::to_string(storage.uplo) << ", "
                                 << Core::to_string(trans));

                    const Kernels::PackedAccess access =
                        Kernels::packed_access(storage.layout, storage.uplo, trans, 0);
                    EXPECT_EQ(0, access.order);

                    // The loop is the one a kernel runs; with no row its body never executes.
                    int64_t visited = 0;
                    for (int64_t r = 0; r < access.order; ++r)
                    {
                        for (int64_t c = access.first_column(r);
                             c < access.end_column(r, access.order); ++c)
                        {
                            EXPECT_TRUE(access.holds(r, c));
                            ++visited;
                        }
                    }
                    EXPECT_EQ(0, visited);
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
