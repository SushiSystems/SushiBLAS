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
 * @file test_serial_reduce.cpp
 * @brief Tests the serial host loops and folds against the launch shapes and accumulators they mirror.
 * @author Mustafa Garip
 */

#include <new>
#include <array>
#include <cmath>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/reference/serial_1d.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/reference/serial_reduce.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds the alignment, in bytes, of every array a test launches over. */
            inline constexpr std::size_t K_TEST_ALIGNMENT = 128;

            /** @brief Holds the element count of the per-element comparisons. */
            inline constexpr std::size_t K_ELEMENT_COUNT = 1000;

            /** @brief Holds the modulus that keeps every generated value an integer a float holds exactly. */
            inline constexpr std::size_t K_VALUE_PERIOD = 1024;

            /** @brief Holds the row count of the grid the axis fold is tested on. */
            inline constexpr std::size_t K_GRID_ROWS = 3;

            /** @brief Holds the column count of the grid the axis fold is tested on. */
            inline constexpr std::size_t K_GRID_COLUMNS = 5;

            /**
             * @brief Owns one 128-byte aligned shared USM array and frees it with `sycl::free`.
             * @tparam T The element type; trivially copyable.
             */
            template<typename T>
            class SharedArray final
            {
                public:
                    /** @brief Allocates count elements on the device of q. */
                    SharedArray(sycl::queue& q, std::size_t count)
                        : queue_(q),
                          data_(sycl::aligned_alloc_shared<T>(K_TEST_ALIGNMENT, count, q))
                    {
                        if (data_ == nullptr)
                            throw std::bad_alloc();
                    }

                    /** @brief Frees the array. */
                    ~SharedArray()
                    {
                        sycl::free(data_, queue_);
                    }

                    SharedArray(const SharedArray&) = delete;
                    SharedArray& operator=(const SharedArray&) = delete;
                    SharedArray(SharedArray&&) = delete;
                    SharedArray& operator=(SharedArray&&) = delete;

                    /** @brief Returns the first element. */
                    [[nodiscard]] T* get() const noexcept
                    {
                        return data_;
                    }

                private:
                    /** @brief The queue whose context owns the array. */
                    sycl::queue& queue_;

                    /** @brief The first element. */
                    T* data_;
            };

            /** @brief Returns the i-th generated input value, an integer below K_VALUE_PERIOD. */
            [[nodiscard]] float value_at(std::size_t i) noexcept
            {
                return static_cast<float>(i % K_VALUE_PERIOD);
            }

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

        /** @brief Gives the serial-loop tests the queue the launched side submits to. */
        class Unit_SerialLoops : public ::testing::Test
        {
            protected:
                /** @brief Returns the queue of the test environment. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return TestEnvironment::queue();
                }
        };

        /** @brief Tests that the serial map writes the bits the launched map writes, for one input and for two. */
        TEST_F(Unit_SerialLoops, MapMatchesTheLaunchedMap)
        {
            sycl::queue& q = queue();
            const std::size_t n = K_ELEMENT_COUNT;

            SharedArray<float> first(q, n);
            SharedArray<float> second(q, n);
            SharedArray<float> by_serial(q, n);
            SharedArray<float> by_launch(q, n);
            for (std::size_t i = 0; i < n; ++i)
            {
                first.get()[i] = value_at(i);
                second.get()[i] = value_at(i + 7);
            }

            const float* x = first.get();
            const float* y = second.get();
            float* serial_out = by_serial.get();
            float* launch_out = by_launch.get();
            const bool aligned = Launch::vector_aligned(x, y, serial_out, launch_out);

            const auto unary = [](float a) { return a * 2.0f + 1.0f; };
            Serial::map(n, unary, serial_out, x);
            q.submit([&](sycl::handler& h)
            {
                Launch::map(h, q, n, aligned, unary, launch_out, x);
            }).wait();
            EXPECT_TRUE(same_bits(serial_out, launch_out, n));

            const auto binary = [](float a, float b) { return a * b - a; };
            Serial::map(n, binary, serial_out, x, y);
            q.submit([&](sycl::handler& h)
            {
                Launch::map(h, q, n, aligned, binary, launch_out, x, y);
            }).wait();
            EXPECT_TRUE(same_bits(serial_out, launch_out, n));
        }

        /** @brief Tests that the serial fused update leaves the bits the launched fused update leaves. */
        TEST_F(Unit_SerialLoops, ZipMatchesTheLaunchedZip)
        {
            sycl::queue& q = queue();
            const std::size_t n = K_ELEMENT_COUNT;

            SharedArray<float> serial_x(q, n);
            SharedArray<float> serial_y(q, n);
            SharedArray<float> launch_x(q, n);
            SharedArray<float> launch_y(q, n);
            for (std::size_t i = 0; i < n; ++i)
            {
                serial_x.get()[i] = value_at(i);
                serial_y.get()[i] = value_at(i + 7);
                launch_x.get()[i] = value_at(i);
                launch_y.get()[i] = value_at(i + 7);
            }

            const auto update = [](float& x, float& y)
            {
                y = y + 0.5f * x;
                x = -x;
            };

            const std::array<float*, 2> serial_ptrs{serial_x.get(), serial_y.get()};
            const std::array<float*, 2> launch_ptrs{launch_x.get(), launch_y.get()};
            const bool aligned = Launch::vector_aligned(launch_ptrs[0], launch_ptrs[1]);

            Serial::zip_inplace(n, serial_ptrs, update);
            q.submit([&](sycl::handler& h)
            {
                Launch::zip_inplace(h, q, n, aligned, launch_ptrs, update);
            }).wait();

            EXPECT_TRUE(same_bits(serial_x.get(), launch_x.get(), n));
            EXPECT_TRUE(same_bits(serial_y.get(), launch_y.get(), n));
        }

        /** @brief Tests that the serial indexed loop calls its operation once per index, ascending. */
        TEST_F(Unit_SerialLoops, IndexedVisitsEveryIndexInOrder)
        {
            const std::size_t n = K_ELEMENT_COUNT;

            std::vector<std::size_t> visited;
            visited.reserve(n);
            Serial::indexed(n, [&visited](std::size_t i) { visited.push_back(i); });

            ASSERT_EQ(visited.size(), n);
            for (std::size_t i = 0; i < n; ++i)
                EXPECT_EQ(visited[i], i);
        }

        /** @brief Tests that a host sum of HALF terms is added in double, where 2048 + 1 + 1 is 2050. */
        TEST_F(Unit_SerialLoops, ASumOfHalfIsFormedInDouble)
        {
            static_assert(
                std::is_same_v<Serial::Fold<Reduce::CompensatedSum, sycl::half>::Wide, double>,
                "The host sum of HALF terms is formed in double.");

            sycl::queue& q = queue();
            SharedArray<sycl::half> terms(q, 3);
            terms.get()[0] = sycl::half(2048.0f);
            terms.get()[1] = sycl::half(1.0f);
            terms.get()[2] = sycl::half(1.0f);

            sycl::half out(0.0f);
            Serial::reduce<Reduce::CompensatedSum>(
                static_cast<const sycl::half*>(terms.get()), &out, std::size_t{3},
                [](sycl::half x) { return x; },
                [](double value, std::size_t) { return value; });

            // 2049 has no HALF value, so a sum formed in HALF stays at 2048.
            EXPECT_EQ(static_cast<float>(out), 2050.0f);
        }

        /** @brief Tests that a host fold keeps the NaN, signed-zero and tie rules of its accumulator. */
        TEST_F(Unit_SerialLoops, AnOrderedFoldKeepsTheAccumulatorsRule)
        {
            sycl::queue& q = queue();
            const auto identity = [](float x) { return x; };
            const auto keep = [](float value, std::size_t) { return value; };

            SharedArray<float> values(q, 3);
            float* v = values.get();
            const float* in = v;

            v[0] = 1.0f;
            v[1] = std::nanf("");
            v[2] = 3.0f;
            float maximum = 0.0f;
            Serial::reduce<Reduce::NaNPropagatingMax>(in, &maximum, std::size_t{3}, identity, keep);
            EXPECT_TRUE(std::isnan(maximum));

            v[0] = -0.0f;
            v[1] = 0.0f;
            maximum = -1.0f;
            Serial::reduce<Reduce::NaNPropagatingMax>(in, &maximum, std::size_t{2}, identity, keep);
            EXPECT_EQ(maximum, 0.0f);
            EXPECT_FALSE(std::signbit(maximum));

            v[0] = 5.0f;
            v[1] = 7.0f;
            v[2] = 7.0f;
            std::int64_t elected = -2;
            Serial::reduce_indexed<Reduce::RunningArgMax>(
                std::size_t{3},
                [in](std::size_t i)
                {
                    return Reduce::IndexedValue<float>{in[i], static_cast<std::int64_t>(i)};
                },
                &elected,
                [](Reduce::IndexedValue<float> winner, std::size_t) { return winner.index; });
            EXPECT_EQ(elected, std::int64_t{1});
        }

        /** @brief Tests that a host fold over no element writes the identity of its accumulator. */
        TEST_F(Unit_SerialLoops, AFoldOfNothingFinalizesTheIdentity)
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

        /** @brief Tests that the host axis fold of a 3 by 5 grid equals one host fold per row and per column. */
        TEST_F(Unit_SerialLoops, AxisFoldMatchesOneFoldPerSegment)
        {
            sycl::queue& q = queue();
            const std::size_t rows = K_GRID_ROWS;
            const std::size_t columns = K_GRID_COLUMNS;

            SharedArray<float> grid(q, rows * columns);
            for (std::size_t i = 0; i < rows * columns; ++i)
                grid.get()[i] = static_cast<float>((i * 7) % 11) + 0.1f;
            const float* in = grid.get();

            const auto identity = [](float x) { return x; };
            const auto keep = [](double value, std::size_t) { return value; };

            Reduce::AxisSpan row_span;
            row_span.segments = rows;
            row_span.extent = columns;
            row_span.segment_stride = columns;

            std::array<float, K_GRID_ROWS> row_sums{};
            Serial::reduce_axis<Reduce::CompensatedSum>(in, row_sums.data(), row_span, identity,
                                                        keep);
            for (std::size_t r = 0; r < rows; ++r)
            {
                const float want = serial_sum(in + r * columns, columns);
                EXPECT_TRUE(same_bits(&row_sums[r], &want, 1)) << "row " << r;
            }

            Reduce::AxisSpan column_span;
            column_span.segments = columns;
            column_span.extent = rows;
            column_span.segment_stride = 1;
            column_span.element_stride = columns;

            std::array<float, K_GRID_COLUMNS> column_sums{};
            Serial::reduce_axis<Reduce::CompensatedSum>(in, column_sums.data(), column_span,
                                                        identity, keep);
            for (std::size_t c = 0; c < columns; ++c)
            {
                std::array<float, K_GRID_ROWS> column{};
                for (std::size_t r = 0; r < rows; ++r)
                    column[r] = in[r * columns + c];

                const float want = serial_sum(column.data(), rows);
                EXPECT_TRUE(same_bits(&column_sums[c], &want, 1)) << "column " << c;
            }
        }

    } // namespace Test
} // namespace SushiBLAS
