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
#include <SushiBLAS/kernels/blas/packed_access.hpp>

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
            constexpr PackedStorage K_STORAGES[] = {{Layout::COLUMN_MAJOR, Uplo::UPPER},
                                                    {Layout::COLUMN_MAJOR, Uplo::LOWER},
                                                    {Layout::ROW_MAJOR, Uplo::UPPER},
                                                    {Layout::ROW_MAJOR, Uplo::LOWER}};

            /** @brief Returns the name of a layout for a failure message. */
            [[nodiscard]] constexpr const char* layout_name(Layout layout) noexcept
            {
                return (layout == Layout::ROW_MAJOR) ? "row-major" : "column-major";
            }

            /** @brief Returns the row of the element a CBLAS walk visits at (line, along). */
            [[nodiscard]] constexpr int64_t walked_row(Layout layout, int64_t line,
                                                       int64_t along) noexcept
            {
                return (layout == Layout::ROW_MAJOR) ? line : along;
            }

            /** @brief Returns the column of the element a CBLAS walk visits at (line, along). */
            [[nodiscard]] constexpr int64_t walked_col(Layout layout, int64_t line,
                                                       int64_t along) noexcept
            {
                return (layout == Layout::ROW_MAJOR) ? along : line;
            }

            /** @brief Returns whether (i, j) lies in a triangle of a square matrix. */
            [[nodiscard]] constexpr bool is_in_triangle(Uplo uplo, int64_t i, int64_t j) noexcept
            {
                return (uplo == Uplo::UPPER) ? (i <= j) : (i >= j);
            }

            /**
             * @brief Returns element (i, j) of the worked U for UPPER and of L = U^T for LOWER.
             * @pre (i, j) lies in the triangle uplo names.
             */
            [[nodiscard]] constexpr double worked_element(Uplo uplo, int64_t i, int64_t j) noexcept
            {
                constexpr double upper[3][3] = {{1.0, 2.0, 3.0}, {0.0, 4.0, 5.0}, {0.0, 0.0, 6.0}};
                return (uplo == Uplo::UPPER) ? upper[i][j] : upper[j][i];
            }

            /** @brief Returns the packed array of the worked triangle of one storage. */
            [[nodiscard]] constexpr const double* worked_storage(const PackedStorage& s) noexcept
            {
                const bool by_rows = (s.layout == Layout::ROW_MAJOR);
                const bool growing = (s.uplo == Uplo::UPPER) != by_rows;
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
                    Kernels::packed_access(layout, uplo, Transpose::NO_TRANS, n);
                const Kernels::PackedAccess t =
                    Kernels::packed_access(layout, uplo, Transpose::TRANS, n);
                const bool upper = (uplo == Uplo::UPPER);
                if (a.is_upper() != upper || t.is_upper() == upper)
                    return false;

                int64_t next = 0;
                for (int64_t line = 0; line < n; ++line)
                {
                    for (int64_t along = 0; along < n; ++along)
                    {
                        const int64_t i = (layout == Layout::ROW_MAJOR) ? line : along;
                        const int64_t j = (layout == Layout::ROW_MAJOR) ? along : line;
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
        TEST(Unit_PackedAccess, TheMappingIsTheCblasOrder)
        {
            static_assert(packed_matches_cblas(Layout::COLUMN_MAJOR, Uplo::UPPER),
                          "A column-major upper packed triangle is the CBLAS one.");
            static_assert(packed_matches_cblas(Layout::COLUMN_MAJOR, Uplo::LOWER),
                          "A column-major lower packed triangle is the CBLAS one.");
            static_assert(packed_matches_cblas(Layout::ROW_MAJOR, Uplo::UPPER),
                          "A row-major upper packed triangle is the CBLAS one.");
            static_assert(packed_matches_cblas(Layout::ROW_MAJOR, Uplo::LOWER),
                          "A row-major lower packed triangle is the CBLAS one.");

            SUCCEED();
        }

        /**
         * @brief Tests that a walk of the triangle in CBLAS order meets the offsets 0, 1, 2
         *        and on, and ends at n * (n + 1) / 2.
         */
        TEST(Unit_PackedAccess, AWalkInCblasOrderCountsWithoutAGap)
        {
            for (const PackedStorage& storage : K_STORAGES)
            {
                for (const int64_t n : {int64_t{1}, int64_t{2}, int64_t{5}})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << layout_name(storage.layout) << ", "
                                 << Core::to_string(storage.uplo) << ", order " << n);

                    const Kernels::PackedAccess access = Kernels::packed_access(
                        storage.layout, storage.uplo, Transpose::NO_TRANS, n);
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
        TEST(Unit_PackedAccess, TheWorkedTrianglesReadBack)
        {
            for (const PackedStorage& storage : K_STORAGES)
            {
                SCOPED_TRACE(::testing::Message() << layout_name(storage.layout) << ", "
                                                  << Core::to_string(storage.uplo));

                const Kernels::PackedAccess access = Kernels::packed_access(
                    storage.layout, storage.uplo, Transpose::NO_TRANS, K_WORKED_ORDER);
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
        TEST(Unit_PackedAccess, ATransposedAccessAddressesTheMirror)
        {
            for (const PackedStorage& storage : K_STORAGES)
            {
                const Kernels::PackedAccess plain = Kernels::packed_access(
                    storage.layout, storage.uplo, Transpose::NO_TRANS, K_WORKED_ORDER);
                const double* p = worked_storage(storage);

                EXPECT_EQ(storage.uplo == Uplo::UPPER, plain.is_upper());
                EXPECT_FALSE(plain.conjugated);

                for (const Transpose trans : {Transpose::TRANS, Transpose::CONJ_TRANS})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << layout_name(storage.layout) << ", "
                                 << Core::to_string(storage.uplo) << ", "
                                 << Core::to_string(trans));

                    const Kernels::PackedAccess op = Kernels::packed_access(
                        storage.layout, storage.uplo, trans, K_WORKED_ORDER);
                    EXPECT_EQ(storage.uplo != Uplo::UPPER, op.is_upper());
                    EXPECT_EQ(trans == Transpose::CONJ_TRANS, op.conjugated);
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
         * @brief Tests that a row of an upper triangle runs from its diagonal to the order and
         *        a row of a lower one from zero through its diagonal.
         */
        TEST(Unit_PackedAccess, TheColumnRangeFollowsTheTriangle)
        {
            constexpr int64_t n = 5;

            for (const PackedStorage& storage : K_STORAGES)
            {
                for (const Transpose trans : {Transpose::NO_TRANS, Transpose::TRANS})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << layout_name(storage.layout) << ", "
                                 << Core::to_string(storage.uplo) << ", "
                                 << Core::to_string(trans));

                    const Kernels::PackedAccess access =
                        Kernels::packed_access(storage.layout, storage.uplo, trans, n);
                    const bool upper = (storage.uplo == Uplo::UPPER) != Core::is_transposed(trans);

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
        TEST(Unit_PackedAccess, AtConjugatesOnlyUnderConjTrans)
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
                    storage.layout, storage.uplo, Transpose::NO_TRANS, K_WORKED_ORDER);
                const Kernels::PackedAccess trans = Kernels::packed_access(
                    storage.layout, storage.uplo, Transpose::TRANS, K_WORKED_ORDER);
                const Kernels::PackedAccess conj = Kernels::packed_access(
                    storage.layout, storage.uplo, Transpose::CONJ_TRANS, K_WORKED_ORDER);
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
        TEST(Unit_PackedAccess, AnOrderOfZeroHoldsNoElement)
        {
            for (const PackedStorage& storage : K_STORAGES)
            {
                for (const Transpose trans : {Transpose::NO_TRANS, Transpose::CONJ_TRANS})
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
