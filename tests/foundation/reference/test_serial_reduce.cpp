/**************************************************************************/
/* test_serial_reduce.cpp                                                 */
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
 * @file test_serial_reduce.cpp
 * @brief Tests the serial host folds against the launched folds and accumulators they mirror.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/reference/serial_reduce.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds the row count of the grid the axis fold is tested on. */
            inline constexpr std::size_t K_GRID_ROWS = 3;

            /** @brief Holds the column count of the grid the axis fold is tested on. */
            inline constexpr std::size_t K_GRID_COLUMNS = 5;

            /** @brief Returns whether the first n elements of a and b hold the same bits. */
            [[nodiscard]] bool same_bits(const float* a, const float* b, std::size_t n) noexcept
            {
                return std::memcmp(a, b, n * sizeof(float)) == 0;
            }

            /** @brief Returns the host sum of `in[0..n)` as Serial::reduce folds it. */
            [[nodiscard]] float serial_sum(const float* in, std::size_t n)
            {
                float out = -1.0f;
                Serial::reduce<Reduce::CompensatedSum>(
                    in, &out, n, [](float x) { return x; },
                    [](double value, std::size_t) { return value; });
                return out;
            }
        } // namespace

        /** @brief Returns the host maximum of the terms as NaNPropagatingMax folds them. */
        [[nodiscard]] static float serial_max(const std::vector<float>& terms)
        {
            float out = -1.0f;
            Serial::reduce<Reduce::NaNPropagatingMax>(
                terms.data(), &out, terms.size(), [](float x) { return x; },
                [](float value, std::size_t) { return value; });
            return out;
        }

        /** @brief Returns the grid of K_GRID_ROWS by K_GRID_COLUMNS terms, row-major. */
        [[nodiscard]] static std::vector<float> grid_terms()
        {
            std::vector<float> grid(K_GRID_ROWS * K_GRID_COLUMNS);
            for (std::size_t i = 0; i < grid.size(); ++i)
            {
                grid[i] = static_cast<float>((i * 7) % 11) + 0.1f;
            }
            return grid;
        }

        /** @brief Returns the terms of one column of the row-major grid. */
        [[nodiscard]] static std::vector<float> column_of(const std::vector<float>& grid,
                                                          std::size_t column)
        {
            std::vector<float> terms;
            for (std::size_t r = 0; r < K_GRID_ROWS; ++r)
            {
                terms.push_back(grid[r * K_GRID_COLUMNS + column]);
            }
            return terms;
        }

        /** @brief Expects each element of `folded` to hold the bits of the host fold of its segment. */
        static void expect_bits_of_each_fold(const std::vector<float>& folded,
                                             const std::vector<std::vector<float>>& segments)
        {
            ASSERT_EQ(folded.size(), segments.size());
            for (std::size_t s = 0; s < segments.size(); ++s)
            {
                const float want = serial_sum(segments[s].data(), segments[s].size());
                EXPECT_TRUE(same_bits(&folded[s], &want, 1)) << "segment " << s;
            }
        }

        /** @brief Tests that a host sum of HALF terms is added in double, where 2048 + 1 + 1 is 2050. */
        TEST(Small_SerialLoops, ASumOfHalfIsFormedInDouble)
        {
            static_assert(
                std::is_same_v<Serial::Fold<Reduce::CompensatedSum, sycl::half>::Wide, double>,
                "The host sum of HALF terms is formed in double.");

            const std::vector<sycl::half> terms{sycl::half(2048.0f), sycl::half(1.0f),
                                                sycl::half(1.0f)};

            sycl::half out(0.0f);
            Serial::reduce<Reduce::CompensatedSum>(
                terms.data(), &out, terms.size(),
                [](sycl::half x) { return x; },
                [](double value, std::size_t) { return value; });

            // 2049 has no HALF value, so a sum formed in HALF stays at 2048.
            EXPECT_EQ(static_cast<float>(out), 2050.0f);
        }

        /** @brief Tests that a host maximum propagates a NaN term. */
        TEST(Small_SerialLoops, AMaximumPropagatesANaN)
        {
            EXPECT_TRUE(std::isnan(serial_max({1.0f, std::nanf(""), 3.0f})));
        }

        /** @brief Tests that a host maximum of -0 and +0 is the positive zero. */
        TEST(Small_SerialLoops, AMaximumOfSignedZerosIsThePositiveZero)
        {
            const float maximum = serial_max({-0.0f, 0.0f});

            EXPECT_EQ(maximum, 0.0f);
            EXPECT_FALSE(std::signbit(maximum));
        }

        /** @brief Tests that a host argmax keeps the first index of a tie. */
        TEST(Small_SerialLoops, AnArgMaxKeepsTheFirstIndexOfATie)
        {
            const std::vector<float> in{5.0f, 7.0f, 7.0f};
            std::int64_t elected = -2;

            Serial::reduce_indexed<Reduce::RunningArgMax>(
                in.size(),
                [&in](std::size_t i)
                {
                    return Reduce::IndexedValue<float>{in[i], static_cast<std::int64_t>(i)};
                },
                &elected,
                [](Reduce::IndexedValue<float> winner, std::size_t) { return winner.index; });

            EXPECT_EQ(elected, std::int64_t{1});
        }

        /** @brief Tests that a host fold over no element writes the identity of its accumulator. */
        TEST(Small_SerialLoops, AFoldOfNothingFinalizesTheIdentity)
        {
            const float* nothing = nullptr;
            const auto identity = [](float x) { return x; };

            float sum = -1.0f;
            Serial::reduce<Reduce::CompensatedSum>(
                nothing, &sum, std::size_t{0}, identity,
                [](double value, std::size_t) { return value; });
            EXPECT_EQ(sum, 0.0f);

            float product = -1.0f;
            Serial::reduce<Reduce::RunningProduct>(
                nothing, &product, std::size_t{0}, identity,
                [](float value, std::size_t) { return value; });
            EXPECT_EQ(product, 1.0f);
        }

        /** @brief Tests that the host axis fold of a 3 by 5 grid along its rows equals one host fold per row. */
        TEST(Small_SerialLoops, AxisFoldAlongRowsMatchesOneFoldPerRow)
        {
            const std::vector<float> grid = grid_terms();
            Reduce::AxisSpan row_span;
            row_span.segments = K_GRID_ROWS;
            row_span.extent = K_GRID_COLUMNS;
            row_span.segment_stride = K_GRID_COLUMNS;

            std::vector<float> row_sums(K_GRID_ROWS);
            Serial::reduce_axis<Reduce::CompensatedSum>(
                grid.data(), row_sums.data(), row_span, [](float x) { return x; },
                [](double value, std::size_t) { return value; });

            std::vector<std::vector<float>> rows;
            for (std::size_t r = 0; r < K_GRID_ROWS; ++r)
            {
                rows.emplace_back(grid.begin() + r * K_GRID_COLUMNS,
                                  grid.begin() + (r + 1) * K_GRID_COLUMNS);
            }
            expect_bits_of_each_fold(row_sums, rows);
        }

        /** @brief Tests that the host axis fold of a 3 by 5 grid along its columns equals one host fold per column. */
        TEST(Small_SerialLoops, AxisFoldAlongColumnsMatchesOneFoldPerColumn)
        {
            const std::vector<float> grid = grid_terms();
            Reduce::AxisSpan column_span;
            column_span.segments = K_GRID_COLUMNS;
            column_span.extent = K_GRID_ROWS;
            column_span.segment_stride = 1;
            column_span.element_stride = K_GRID_COLUMNS;

            std::vector<float> column_sums(K_GRID_COLUMNS);
            Serial::reduce_axis<Reduce::CompensatedSum>(
                grid.data(), column_sums.data(), column_span, [](float x) { return x; },
                [](double value, std::size_t) { return value; });

            std::vector<std::vector<float>> columns;
            for (std::size_t c = 0; c < K_GRID_COLUMNS; ++c)
            {
                columns.push_back(column_of(grid, c));
            }
            expect_bits_of_each_fold(column_sums, columns);
        }
    } // namespace Test
} // namespace SushiBLAS
