/**************************************************************************/
/* test_level1.cpp                                                        */
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
 * @file test_level1.cpp
 * @brief Tests the SYCL sparse-dense level 1 kernels against the reference ones on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/sparse_kernels/index_list.hpp>
#include <SushiBLAS/sparse_kernels/reference/level1.hpp>
#include <SushiBLAS/sparse_kernels/sycl/level1.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The dense length; the stored positions are every third one.
        constexpr int64_t K_EXTENT = 30000;

        /// @brief The stored entries.
        constexpr int64_t K_COUNT = K_EXTENT / 3;

        /** @brief Runs the level 1 kernels on the host CPU queue. */
        class Device_SYCLSparseLevel1 : public HostDeviceTest
        {
            protected:
                /** @brief Returns positions 0, 3, 6, ... in a shared array. */
                SharedArray<int64_t> positions()
                {
                    std::vector<int64_t> host(static_cast<std::size_t>(K_COUNT));
                    for (int64_t k = 0; k < K_COUNT; ++k)
                        host[static_cast<std::size_t>(k)] = 3 * k;
                    return SharedArray<int64_t>(queue(), host);
                }
        };

        TEST_F(Device_SYCLSparseLevel1, TheUpdatesMatchTheReference)
        {
            const SharedArray<int64_t> stored = positions();
            const Kernels::IndexList<int64_t> list{K_COUNT, K_EXTENT, 0, stored.get()};
            std::vector<double> x = small_integers<double>(K_COUNT, 1);
            std::vector<double> y = small_integers<double>(K_EXTENT, 2);
            SharedArray<double> device_x(queue(), x);
            SharedArray<double> device_y(queue(), y);

            Kernels::SYCL::axpyi(queue(), 3.0, device_x.get(), list, device_y.get(), 1, {}).wait();
            Kernels::SYCL::roti(queue(), device_x.get(), list, device_y.get(), 1, 0.0, 1.0,
                                {}).wait();
            Kernels::SYCL::gthrz(queue(), device_y.get(), 1, list, device_x.get(), {}).wait();
            Kernels::SYCL::sctr(queue(), device_x.get(), list, device_y.get(), 1, {}).wait();
            Kernels::Reference::axpyi(3.0, x.data(), list, y.data(), 1);
            Kernels::Reference::roti(x.data(), list, y.data(), 1, 0.0, 1.0);
            Kernels::Reference::gthrz(y.data(), 1, list, x.data());
            Kernels::Reference::sctr(x.data(), list, y.data(), 1);

            EXPECT_EQ(device_x.values(), x);
            EXPECT_EQ(device_y.values(), y);
        }

        TEST_F(Device_SYCLSparseLevel1, GatherAndDotMatchTheReference)
        {
            const SharedArray<int64_t> stored = positions();
            const Kernels::IndexList<int64_t> list{K_COUNT, K_EXTENT, 0, stored.get()};
            const std::vector<double> x = small_integers<double>(K_COUNT, 3);
            const std::vector<double> y = small_integers<double>(K_EXTENT, 4);
            const SharedArray<double> device_x(queue(), x);
            const SharedArray<double> device_y(queue(), y);
            SharedArray<double> gathered(queue(), static_cast<std::size_t>(K_COUNT));
            SharedArray<double> dot(queue(), 1);
            const std::size_t scratch_bytes = 4096;
            SharedArray<std::byte> scratch(queue(), scratch_bytes);

            Kernels::SYCL::gthr(queue(), device_y.get(), 1, list, gathered.get(), {}).wait();
            Kernels::SYCL::doti(queue(), device_x.get(), list, device_y.get(), 1, dot.get(),
                                scratch.get(), scratch_bytes, {}).wait();
            std::vector<double> expected_gather(static_cast<std::size_t>(K_COUNT));
            double expected_dot = 0.0;
            Kernels::Reference::gthr(y.data(), 1, list, expected_gather.data());
            Kernels::Reference::doti(x.data(), list, y.data(), 1, &expected_dot);

            EXPECT_EQ(gathered.values(), expected_gather);
            EXPECT_EQ(dot.get()[0], expected_dot);
        }
    } // namespace Test
} // namespace SushiBLAS
