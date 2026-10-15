/**************************************************************************/
/* test_shared_array.cpp                                                  */
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
 * @file test_shared_array.cpp
 * @brief Tests that SharedArray aligns and fills its elements and that a kernel reaches them.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the SharedArray tests on the host CPU queue. */
        class Device_SharedArray : public HostDeviceTest
        {

        };

        TEST_F(Device_SharedArray, StartsValueInitialisedAndAligned)
        {
            const SharedArray<float> array(queue(), 5);

            EXPECT_EQ(array.size(), std::size_t{5});
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(array.get()) % K_SHARED_ARRAY_ALIGNMENT,
                      std::uintptr_t{0});
            EXPECT_EQ(array.values(), std::vector<float>(5, 0.0f));
        }

        TEST_F(Device_SharedArray, HoldsTheValuesItWasGiven)
        {
            const std::vector<int> values{3, 1, 4, 1, 5};
            const SharedArray<int> array(queue(), values);

            EXPECT_EQ(array.values(), values);
        }

        TEST_F(Device_SharedArray, AKernelWritesWhatTheHostReads)
        {
            SharedArray<int> array(queue(), 8);
            int* data = array.get();

            queue().parallel_for(sycl::range<1>(8), [=](sycl::id<1> i)
            {
                data[i] = static_cast<int>(i[0]) * 2;
            }).wait();

            EXPECT_EQ(array.values(), (std::vector<int>{0, 2, 4, 6, 8, 10, 12, 14}));
        }

        TEST_F(Device_SharedArray, AnEmptyArrayHasNoElements)
        {
            const SharedArray<double> array(queue(), 0);

            EXPECT_EQ(array.size(), std::size_t{0});
            EXPECT_TRUE(array.values().empty());
        }
    } // namespace Test
} // namespace SushiBLAS
