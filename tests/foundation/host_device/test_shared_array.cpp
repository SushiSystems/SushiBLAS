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
 * @file test_shared_array.cpp
 * @brief Tests that SharedArray fills, copies and aligns its elements through its allocator and that a kernel reaches them.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/host_shared_allocator.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the SharedArray value tests on host memory, with no device and no queue. */
        class Small_SharedArray : public ::testing::Test
        {
            protected:
                /// @brief The memory every array of a test comes from.
                HostSharedAllocator allocator;
        };

        /** @brief Runs the SharedArray tests that reach a kernel on the host CPU queue. */
        class Medium_SharedArray : public HostDeviceTest
        {

        };

        TEST_F(Small_SharedArray, StartsValueInitialised)
        {
            const SharedArray<float> array(allocator, 5);

            EXPECT_EQ(array.size(), std::size_t{5});
            EXPECT_EQ(array.values(), std::vector<float>(5, 0.0f));
        }

        TEST_F(Small_SharedArray, AsksTheAllocatorForTheSharedArrayAlignment)
        {
            const SharedArray<float> array(allocator, 5);

            EXPECT_EQ(allocator.last_alignment(), K_SHARED_ARRAY_ALIGNMENT);
            EXPECT_EQ(allocator.last_bytes(), 5 * sizeof(float));
        }

        TEST_F(Small_SharedArray, HoldsTheValuesItWasGiven)
        {
            const std::vector<int> values{3, 1, 4, 1, 5};
            const SharedArray<int> array(allocator, values);

            EXPECT_EQ(array.values(), values);
        }

        TEST_F(Small_SharedArray, AnEmptyArrayHasNoElements)
        {
            const SharedArray<double> array(allocator, 0);

            EXPECT_EQ(array.size(), std::size_t{0});
            EXPECT_TRUE(array.values().empty());
        }

        TEST_F(Small_SharedArray, AnEmptyArrayStillTakesOneElementOfMemory)
        {
            const SharedArray<double> array(allocator, 0);

            EXPECT_EQ(allocator.last_bytes(), sizeof(double));
        }

        TEST_F(Small_SharedArray, GivesItsMemoryBackWhenItGoesOutOfScope)
        {
            {
                const SharedArray<int> array(allocator, 4);
            }

            EXPECT_EQ(allocator.free_count(), allocator.allocation_count());
            EXPECT_EQ(allocator.allocation_count(), std::size_t{1});
        }

        TEST_F(Small_SharedArray, ThrowsWhenTheAllocatorHasNoMemory)
        {
            allocator.set_failing(true);

            EXPECT_THROW(SharedArray<int>(allocator, 4), std::bad_alloc);
        }

        TEST_F(Medium_SharedArray, IsAlignedToTheSharedArrayAlignmentOnTheQueue)
        {
            const SharedArray<float> array(queue(), 5);

            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(array.get()) % K_SHARED_ARRAY_ALIGNMENT,
                      std::uintptr_t{0});
        }

        TEST_F(Medium_SharedArray, AKernelWritesWhatTheHostReads)
        {
            SharedArray<int> array(queue(), 8);
            int* data = array.get();

            queue().parallel_for(sycl::range<1>(8), [=](sycl::id<1> i)
            {
                data[i] = static_cast<int>(i[0]) * 2;
            }).wait();

            EXPECT_EQ(array.values(), (std::vector<int>{0, 2, 4, 6, 8, 10, 12, 14}));
        }
    } // namespace Test
} // namespace SushiBLAS
