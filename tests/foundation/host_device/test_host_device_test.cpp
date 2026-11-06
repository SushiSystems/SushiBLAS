/**************************************************************************/
/* test_host_device_test.cpp                                              */
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
 * @file test_host_device_test.cpp
 * @brief Tests that HostDeviceTest hands a test the host CPU queue.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the fixture under its own suite name. */
        class Medium_HostDeviceTest : public HostDeviceTest
        {

        };

        TEST_F(Medium_HostDeviceTest, GivesTheHostQueueToTheTest)
        {
            EXPECT_EQ(&queue(), host_queue());
            EXPECT_TRUE(queue().get_device().is_cpu());
        }
    } // namespace Test
} // namespace SushiBLAS
