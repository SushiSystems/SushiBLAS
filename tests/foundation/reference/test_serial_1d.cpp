/**************************************************************************/
/* test_serial_1d.cpp                                                     */
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
 * @file test_serial_1d.cpp
 * @brief Tests the serial map, fused update and indexed loop against the launched shapes.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/reference/serial_1d.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The element count of the per-element comparisons.
        constexpr std::size_t K_ELEMENT_COUNT = 1000;

        /// @brief The modulus that keeps every generated value an integer a float holds exactly.
        constexpr std::size_t K_VALUE_PERIOD = 1024;

        /// @brief Returns the i-th generated input value, an integer below K_VALUE_PERIOD.
        [[nodiscard]] float value_at(std::size_t i) noexcept
        {
            return static_cast<float>(i % K_VALUE_PERIOD);
        }

        /// @brief Returns the generated values from i = offset to offset + n.
        [[nodiscard]] std::vector<float> values_from(std::size_t offset, std::size_t n)
        {
            std::vector<float> values(n);
            for (std::size_t i = 0; i < n; ++i)
                values[i] = value_at(i + offset);
            return values;
        }

        /// @brief Returns whether the first n elements of a and b hold the same bits.
        [[nodiscard]] bool same_bits(const float* a, const float* b, std::size_t n) noexcept
        {
            return std::memcmp(a, b, n * sizeof(float)) == 0;
        }

        /** @brief Runs the serial loops against the launched shapes on the host CPU queue. */
        class Device_Serial1D : public HostDeviceTest
        {

        };

        TEST_F(Device_Serial1D, MapWritesTheBitsTheLaunchedMapWrites)
        {
            sycl::queue& q = queue();
            const std::size_t n = K_ELEMENT_COUNT;
            const SharedArray<float> first(q, values_from(0, n));
            const SharedArray<float> second(q, values_from(7, n));
            SharedArray<float> by_serial(q, n);
            SharedArray<float> by_launch(q, n);
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

        TEST_F(Device_Serial1D, ZipLeavesTheBitsTheLaunchedZipLeaves)
        {
            sycl::queue& q = queue();
            const std::size_t n = K_ELEMENT_COUNT;
            SharedArray<float> serial_x(q, values_from(0, n));
            SharedArray<float> serial_y(q, values_from(7, n));
            SharedArray<float> launch_x(q, values_from(0, n));
            SharedArray<float> launch_y(q, values_from(7, n));
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

        TEST(Unit_Serial1D, IndexedVisitsEveryIndexInOrder)
        {
            std::vector<std::size_t> visited;
            visited.reserve(K_ELEMENT_COUNT);
            Serial::indexed(K_ELEMENT_COUNT, [&visited](std::size_t i) { visited.push_back(i); });

            ASSERT_EQ(visited.size(), K_ELEMENT_COUNT);
            for (std::size_t i = 0; i < K_ELEMENT_COUNT; ++i)
                EXPECT_EQ(visited[i], i);
        }

        TEST(Unit_Serial1D, AMapOfNothingWritesNothing)
        {
            float out = 7.0f;
            const float in = 1.0f;

            Serial::map(0, [](float a) { return a; }, &out, &in);

            EXPECT_EQ(out, 7.0f);
        }
    } // namespace Test
} // namespace SushiBLAS
