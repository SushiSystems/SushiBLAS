/**************************************************************************/
/* test_packed_panel.cpp                                                  */
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
 * @file test_packed_panel.cpp
 * @brief Tests pack_panel, which copies a block of op(X) into unit-stride micro-panels.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_panel.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Kernels::SYCL::Detail::PanelLines;

            /// @brief Holds the rows of the source matrix.
            constexpr int64_t K_SOURCE_ROWS = 7;

            /// @brief Holds the columns of the source matrix.
            constexpr int64_t K_SOURCE_COLS = 6;

            /// @brief Holds the rows of the source a count of two whole micro-panels reads.
            constexpr int64_t K_WHOLE_PANEL_ROWS = 8;

            /// @brief Holds the lines of one micro-panel.
            constexpr int K_WIDTH = 4;

            /// @brief Holds the first line the common panel packs.
            constexpr int64_t K_FIRST = 1;

            /// @brief Holds the lines the common panel packs: one micro-panel and one line.
            constexpr int64_t K_COUNT = 5;

            /// @brief Holds the first term the common panel packs.
            constexpr int64_t K_TERM_FIRST = 2;

            /// @brief Holds the terms the common panel packs.
            constexpr int64_t K_DEPTH = 3;

            /// @brief Holds the value a real has before pack_panel runs.
            constexpr float K_SENTINEL = -777.0f;

            /** @brief Returns the reals one packed element of accumulation type W takes. */
            template<typename W>
            [[nodiscard]] constexpr int64_t parts_of() noexcept
            {
                return Core::is_complex_v<W> ? 2 : 1;
            }

            /**
             * @brief Returns the real part of a value for plane 0 and its imaginary part for
             *        plane 1.
             * @param plane Must be 0 for a real W.
             */
            template<typename W>
            [[nodiscard]] Kernels::RealType<W> part_of(W value, int64_t plane) noexcept
            {
                if constexpr (Core::is_complex_v<W>)
                    return (plane == 0) ? value.real() : value.imag();
                else
                    return value;
            }

            /**
             * @brief Returns element (row, col) of the source in the accumulation type of T:
             *        10 row + col, and for a complex type 100 more in the imaginary part.
             */
            template<typename T>
            [[nodiscard]] Kernels::Accumulator<T> source_value(int64_t row, int64_t col) noexcept
            {
                using W = Kernels::Accumulator<T>;
                using R = Kernels::RealType<W>;

                const R real = static_cast<R>(10 * row + col);
                if constexpr (Core::is_complex_v<W>)
                    return W(real, static_cast<R>(100) + real);
                else
                    return real;
            }

            /** @brief Returns element (row, col) of the source in its stored type. */
            template<typename T>
            [[nodiscard]] T source_element(int64_t row, int64_t col) noexcept
            {
                return static_cast<T>(source_value<T>(row, col));
            }

            /** @brief Holds one stored copy of the source and the access that reads it. */
            template<typename T>
            struct StoredSource
            {
                std::vector<T> elements;        /**< The source in one storage order. */
                Kernels::OperandAccess access;  /**< Reads element (row, col) of elements. */
            };

            /** @brief Returns the source of the given rows stored row by row. */
            template<typename T>
            [[nodiscard]] StoredSource<T> row_major_source(int64_t rows)
            {
                StoredSource<T> source;
                source.elements.resize(static_cast<std::size_t>(rows * K_SOURCE_COLS));
                source.access = Kernels::OperandAccess{K_SOURCE_COLS, 1, false};
                for (int64_t r = 0; r < rows; ++r)
                    for (int64_t c = 0; c < K_SOURCE_COLS; ++c)
                        source.elements[static_cast<std::size_t>(source.access.index(r, c))] =
                            source_element<T>(r, c);
                return source;
            }

            /** @brief Returns the source of the given rows stored column by column. */
            template<typename T>
            [[nodiscard]] StoredSource<T> column_major_source(int64_t rows)
            {
                StoredSource<T> source;
                source.elements.resize(static_cast<std::size_t>(rows * K_SOURCE_COLS));
                source.access = Kernels::OperandAccess{1, rows, false};
                for (int64_t r = 0; r < rows; ++r)
                    for (int64_t c = 0; c < K_SOURCE_COLS; ++c)
                        source.elements[static_cast<std::size_t>(source.access.index(r, c))] =
                            source_element<T>(r, c);
                return source;
            }

            /** @brief Returns the source of the given rows in both storage orders. */
            template<typename T>
            [[nodiscard]] std::array<StoredSource<T>, 2> both_storage_orders(int64_t rows)
            {
                return {row_major_source<T>(rows), column_major_source<T>(rows)};
            }

            /** @brief Reads a row-major source of K_SOURCE_COLS columns and offers only `at`. */
            struct AtOnlyAccess
            {
                /** @brief Returns element (r, c) of the row-major source. */
                template<typename T>
                [[nodiscard]] T at(const T* p, int64_t r, int64_t c) const noexcept
                {
                    return p[r * K_SOURCE_COLS + c];
                }
            };

            /** @brief Returns the reals a panel of count lines and depth terms takes. */
            template<typename T>
            [[nodiscard]] int64_t panel_reals(int64_t count, int64_t depth) noexcept
            {
                const int64_t groups = (count + K_WIDTH - 1) / K_WIDTH;
                return groups * depth * parts_of<Kernels::Accumulator<T>>() * K_WIDTH;
            }

            /** @brief Returns where one part of line `line`, term `term` of a panel is stored. */
            template<typename T>
            [[nodiscard]] std::size_t packed_offset(int64_t line, int64_t term, int64_t depth,
                                                    int64_t plane) noexcept
            {
                const int64_t slice_reals = parts_of<Kernels::Accumulator<T>>() * K_WIDTH;
                return static_cast<std::size_t>((line / K_WIDTH) * depth * slice_reals +
                                                term * slice_reals + plane * K_WIDTH +
                                                line % K_WIDTH);
            }

            /**
             * @brief Packs a panel into storage filled with K_SENTINEL.
             * @return The panel followed by one real that pack_panel must leave alone.
             */
            template<typename T, PanelLines Lines, typename Access>
            [[nodiscard]] std::vector<Kernels::RealType<Kernels::Accumulator<T>>>
            pack_guarded(const std::vector<T>& x, const Access& access, int64_t first,
                         int64_t count, int64_t term_first, int64_t depth)
            {
                using R = Kernels::RealType<Kernels::Accumulator<T>>;

                std::vector<R> panel(static_cast<std::size_t>(panel_reals<T>(count, depth)) + 1,
                                     static_cast<R>(K_SENTINEL));
                Kernels::SYCL::Detail::pack_panel<T, K_WIDTH, Lines>(
                    x.data(), access, first, count, term_first, depth, panel.data());
                return panel;
            }

            /** @brief Packs the common panel: K_COUNT lines from K_FIRST over K_DEPTH terms. */
            template<typename T, PanelLines Lines, typename Access>
            [[nodiscard]] std::vector<Kernels::RealType<Kernels::Accumulator<T>>>
            pack_common(const std::vector<T>& x, const Access& access)
            {
                return pack_guarded<T, Lines>(x, access, K_FIRST, K_COUNT, K_TERM_FIRST, K_DEPTH);
            }

            /**
             * @brief Expects that the common panel holds op(X) line by line and zeroes the
             *        lines its last micro-panel lacks.
             */
            template<typename T, PanelLines Lines>
            void expect_common_panel(
                const std::vector<Kernels::RealType<Kernels::Accumulator<T>>>& panel)
            {
                using W = Kernels::Accumulator<T>;
                using R = Kernels::RealType<W>;

                const int64_t padded = (K_COUNT + K_WIDTH - 1) / K_WIDTH * K_WIDTH;
                for (int64_t l = 0; l < K_DEPTH; ++l)
                {
                    for (int64_t plane = 0; plane < parts_of<W>(); ++plane)
                    {
                        for (int64_t i = 0; i < K_COUNT; ++i)
                        {
                            const W expected = (Lines == PanelLines::Rows)
                                ? source_value<T>(K_FIRST + i, K_TERM_FIRST + l)
                                : source_value<T>(K_TERM_FIRST + l, K_FIRST + i);
                            EXPECT_EQ(panel[packed_offset<T>(i, l, K_DEPTH, plane)],
                                      part_of(expected, plane))
                                << "line " << i << ", term " << l << ", plane " << plane;
                        }
                        for (int64_t i = K_COUNT; i < padded; ++i)
                            EXPECT_EQ(panel[packed_offset<T>(i, l, K_DEPTH, plane)], R(0))
                                << "padding line " << i << ", term " << l << ", plane " << plane;
                    }
                }
            }
        }

        /**
         * @brief Names the element type a panel is packed from.
         * @tparam T The stored element type; the panel holds reals of Accumulator<T>.
         */
        template<typename T>
        class Unit_PackedPanel : public ::testing::Test
        {

        };

        using PackedPanelTypes =
            ::testing::Types<sycl::half, float, std::complex<float>, std::complex<double>>;
        SB_TYPED_TEST_SUITE(Unit_PackedPanel, PackedPanelTypes);

        /** @brief Tests that lines are grouped into micro-panels and terms into slices. */
        TYPED_TEST(Unit_PackedPanel, LinesAreGroupedAndTermsAreSlices)
        {
            using T = TypeParam;

            const StoredSource<T> source = row_major_source<T>(K_SOURCE_ROWS);
            const auto panel = pack_common<T, PanelLines::Rows>(source.elements, source.access);

            ASSERT_EQ(panel.size(), static_cast<std::size_t>(panel_reals<T>(K_COUNT, K_DEPTH)) + 1);
            expect_common_panel<T, PanelLines::Rows>(panel);
        }

        /** @brief Tests that the storage order of the source does not change the panel. */
        TYPED_TEST(Unit_PackedPanel, BothStorageOrdersGiveTheSamePanel)
        {
            using T = TypeParam;

            const StoredSource<T> by_rows = row_major_source<T>(K_SOURCE_ROWS);
            const StoredSource<T> by_columns = column_major_source<T>(K_SOURCE_ROWS);

            const auto rows_of_rows =
                pack_common<T, PanelLines::Rows>(by_rows.elements, by_rows.access);
            const auto rows_of_columns =
                pack_common<T, PanelLines::Rows>(by_columns.elements, by_columns.access);
            const auto columns_of_rows =
                pack_common<T, PanelLines::Columns>(by_rows.elements, by_rows.access);
            const auto columns_of_columns =
                pack_common<T, PanelLines::Columns>(by_columns.elements, by_columns.access);

            EXPECT_EQ(rows_of_rows, rows_of_columns);
            EXPECT_EQ(columns_of_rows, columns_of_columns);
        }

        /** @brief Tests that a panel of columns holds element (term, line) of the source. */
        TYPED_TEST(Unit_PackedPanel, ColumnLinesReadTheSecondFactor)
        {
            using T = TypeParam;

            for (const StoredSource<T>& source : both_storage_orders<T>(K_SOURCE_ROWS))
                expect_common_panel<T, PanelLines::Columns>(
                    pack_common<T, PanelLines::Columns>(source.elements, source.access));
        }

        /** @brief Tests that an access brick whose only member is `at` is packed. */
        TYPED_TEST(Unit_PackedPanel, ABrickWithOnlyAtIsPacked)
        {
            using T = TypeParam;

            const StoredSource<T> source = row_major_source<T>(K_SOURCE_ROWS);
            const AtOnlyAccess at_only;

            const auto rows_by_at = pack_common<T, PanelLines::Rows>(source.elements, at_only);
            const auto rows_by_access =
                pack_common<T, PanelLines::Rows>(source.elements, source.access);
            const auto columns_by_at =
                pack_common<T, PanelLines::Columns>(source.elements, at_only);
            const auto columns_by_access =
                pack_common<T, PanelLines::Columns>(source.elements, source.access);

            EXPECT_EQ(rows_by_at, rows_by_access);
            EXPECT_EQ(columns_by_at, columns_by_access);
        }

        /** @brief Tests that a conjugated access negates the imaginary plane and no other. */
        TYPED_TEST(Unit_PackedPanel, AConjugatedAccessNegatesTheImaginaryPlane)
        {
            using T = TypeParam;
            using W = Kernels::Accumulator<T>;

            for (const StoredSource<T>& source : both_storage_orders<T>(K_SOURCE_ROWS))
            {
                Kernels::OperandAccess conjugated = source.access;
                conjugated.conjugated = true;

                const auto plain = pack_common<T, PanelLines::Rows>(source.elements, source.access);
                const auto flipped = pack_common<T, PanelLines::Rows>(source.elements, conjugated);

                ASSERT_EQ(plain.size(), flipped.size());
                for (std::size_t i = 0; i + 1 < plain.size(); ++i)
                {
                    const bool imaginary = (i / K_WIDTH) % parts_of<W>() == 1;
                    EXPECT_EQ(flipped[i], imaginary ? -plain[i] : plain[i]) << "real " << i;
                }
            }
        }

        /** @brief Tests that a slice is Width real parts followed by Width imaginary parts. */
        TYPED_TEST(Unit_PackedPanel, ComplexPartsAreStoredInPlanes)
        {
            using T = TypeParam;
            using W = Kernels::Accumulator<T>;

            const int64_t depth = 2;
            const StoredSource<T> source = row_major_source<T>(K_SOURCE_ROWS);
            const auto panel = pack_guarded<T, PanelLines::Rows>(source.elements, source.access, 0,
                                                                 K_WIDTH, 0, depth);

            ASSERT_EQ(panel.size(), static_cast<std::size_t>(depth * parts_of<W>() * K_WIDTH) + 1);
            for (int64_t l = 0; l < depth; ++l)
                for (int64_t plane = 0; plane < parts_of<W>(); ++plane)
                    for (int64_t i = 0; i < K_WIDTH; ++i)
                        EXPECT_EQ(panel[static_cast<std::size_t>(l * parts_of<W>() * K_WIDTH +
                                                                 plane * K_WIDTH + i)],
                                  part_of(source_value<T>(i, l), plane))
                            << "line " << i << ", term " << l << ", plane " << plane;
        }

        /** @brief Tests that the panel holds reals of Accumulator<T> and converts each exactly. */
        TYPED_TEST(Unit_PackedPanel, HalfIsPackedAsFloat)
        {
            using T = TypeParam;
            using R = Kernels::RealType<Kernels::Accumulator<T>>;

            using HalfReal = Kernels::RealType<Kernels::Accumulator<sycl::half>>;

            static_assert(std::is_same_v<HalfReal, float>, "A HALF panel must hold FLOAT32 reals.");

            const StoredSource<T> source = row_major_source<T>(K_SOURCE_ROWS);
            const std::vector<R> panel =
                pack_guarded<T, PanelLines::Rows>(source.elements, source.access, 0, K_WIDTH, 0, 1);

            for (int64_t i = 0; i < K_WIDTH; ++i)
                EXPECT_EQ(panel[static_cast<std::size_t>(i)], static_cast<R>(10 * i))
                    << "line " << i;
        }

        /** @brief Tests that the real after the last micro-panel is left alone. */
        TYPED_TEST(Unit_PackedPanel, NothingIsWrittenPastThePanel)
        {
            using T = TypeParam;
            using R = Kernels::RealType<Kernels::Accumulator<T>>;

            for (const StoredSource<T>& source : both_storage_orders<T>(K_WHOLE_PANEL_ROWS))
            {
                for (const int64_t count : {K_COUNT, K_WHOLE_PANEL_ROWS})
                {
                    const auto panel = pack_guarded<T, PanelLines::Rows>(
                        source.elements, source.access, 0, count, K_TERM_FIRST, K_DEPTH);

                    ASSERT_EQ(panel.size(),
                              static_cast<std::size_t>(panel_reals<T>(count, K_DEPTH)) + 1);
                    EXPECT_EQ(panel.back(), static_cast<R>(K_SENTINEL)) << "count " << count;
                    EXPECT_NE(panel[panel.size() - 2], static_cast<R>(K_SENTINEL))
                        << "count " << count;
                }
            }
        }

        /** @brief Tests that a panel of no lines, or of no terms, writes no real. */
        TYPED_TEST(Unit_PackedPanel, ACountOrADepthOfZeroWritesNothing)
        {
            using T = TypeParam;
            using R = Kernels::RealType<Kernels::Accumulator<T>>;

            for (const StoredSource<T>& source : both_storage_orders<T>(K_SOURCE_ROWS))
            {
                const auto no_lines = pack_guarded<T, PanelLines::Rows>(
                    source.elements, source.access, K_FIRST, 0, K_TERM_FIRST, K_DEPTH);
                const auto no_terms = pack_guarded<T, PanelLines::Rows>(
                    source.elements, source.access, K_FIRST, K_COUNT, K_TERM_FIRST, 0);

                ASSERT_EQ(no_lines.size(), std::size_t{1});
                ASSERT_EQ(no_terms.size(), std::size_t{1});
                EXPECT_EQ(no_lines[0], static_cast<R>(K_SENTINEL));
                EXPECT_EQ(no_terms[0], static_cast<R>(K_SENTINEL));
            }
        }
    } // namespace Test
} // namespace SushiBLAS
