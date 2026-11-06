/**************************************************************************/
/* test_test_environment.cpp                                              */
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
 * @file test_test_environment.cpp
 * @brief Tests that TestEnvironment hands out one runtime, its queues and its allocation count.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_TestEnvironment, HandsOutTheSameRuntimeEveryTime)
        {
            EXPECT_EQ(&TestEnvironment::runtime(), &TestEnvironment::runtime());
        }

        TEST(Medium_TestEnvironment, ReportsAtLeastOneDeviceAndItsQueue)
        {
            ASSERT_GE(TestEnvironment::device_count(), std::size_t{1});

            sycl::queue& first = TestEnvironment::queue();
            EXPECT_EQ(&first, &TestEnvironment::queue(0));
        }

        TEST(Medium_TestEnvironment, CountsALiveAllocationUntilItIsFreed)
        {
            const std::size_t before = TestEnvironment::live_allocation_count();
            {
                auto buffer = TestEnvironment::runtime().buffer<float>(16);
                ASSERT_TRUE(buffer.has_value());
                EXPECT_EQ(TestEnvironment::live_allocation_count(), before + 1);
            }

            EXPECT_EQ(TestEnvironment::live_allocation_count(), before);
        }
    } // namespace Test
} // namespace SushiBLAS
